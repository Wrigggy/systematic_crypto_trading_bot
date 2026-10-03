from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

from core.models import Order, OrderStatus, OrderType, Side
from data.buffer import LiveBuffer
from execution.executor import BaseExecutor

logger = logging.getLogger(__name__)


class SimExecutor(BaseExecutor):
    """Simulated executor for paper trading.

    Market orders fill immediately at latest price +/- slippage.
    Limit orders fill if the price is favorable.
    """

    def __init__(self, config: dict, buffer: LiveBuffer, now=None):
        self._slippage_bps: float = config.get("slippage_bps", 5.0)
        self._fee_bps: float = config.get("fee_bps", 10.0)
        self._maker_fee_bps = config.get("fee_limit_bps", 5.0)
        self._taker_fee_bps = config.get("fee_market_bps", self._fee_bps)
        self._penetration_bps = config.get("limit_penetration_bps", 2.0)
        self._submitted_candles = {}
        self._buffer = buffer
        self._pending_orders: Dict[str, Order] = {}
        self._now = now or datetime.utcnow

    async def execute(self, order: Order) -> Order:
        candle = await self._buffer.get_latest_candle(order.symbol)
        if candle is None:
            order.status = OrderStatus.REJECTED
            logger.warning("SimExecutor: no price data for %s", order.symbol)
            return order

        price = candle.close
        slippage_mult = self._slippage_bps / 10000.0

        if order.order_type == OrderType.MARKET:
            # Apply slippage
            if order.side == Side.BUY:
                fill_price = price * (1 + slippage_mult)
            else:
                fill_price = price * (1 - slippage_mult)

            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.filled_at = self._now()
            order.status = OrderStatus.FILLED

            logger.info(
                "SIM %s %s qty=%.6f @ %.2f (market=%.2f, slip=%.1fbps)",
                order.side.value,
                order.symbol,
                order.filled_quantity,
                order.filled_price,
                price,
                self._slippage_bps,
            )

        elif order.order_type == OrderType.LIMIT:
            if order.maker_preferred:
                order.status = OrderStatus.SUBMITTED
                self._pending_orders[order.order_id] = order
                self._submitted_candles[order.order_id] = candle.timestamp
                return order  # Never fill from the bar used to decide the order.
            can_fill = False
            if (
                order.side == Side.BUY
                and order.price is not None
                and price <= order.price
            ):
                can_fill = True
            elif (
                order.side == Side.SELL
                and order.price is not None
                and price >= order.price
            ):
                can_fill = True

            if can_fill:
                order.filled_price = order.price
                order.filled_quantity = order.quantity
                order.filled_at = self._now()
                order.status = OrderStatus.FILLED
                logger.info(
                    "SIM LIMIT %s %s qty=%.6f @ %.2f",
                    order.side.value,
                    order.symbol,
                    order.filled_quantity,
                    order.filled_price,
                )
            else:
                order.status = OrderStatus.SUBMITTED
                self._pending_orders[order.order_id] = order
                logger.info(
                    "SIM LIMIT pending: %s %s qty=%.6f limit=%.2f (market=%.2f)",
                    order.side.value,
                    order.symbol,
                    order.quantity,
                    order.price,
                    price,
                )

        if order.status == OrderStatus.FILLED:
            self._set_fee(order, maker=False)
        return order

    def _set_fee(self, order, maker):
        order.fill_time_source = "simulated_execution"
        order.liquidity = "MAKER" if maker else "TAKER"
        order.fee_bps = self._maker_fee_bps if maker else self._taker_fee_bps
        order.commission_asset = order.symbol.split("/")[1]
        order.commission = order.filled_quantity * order.filled_price * order.fee_bps / 10000

    async def cancel(self, order_id: str, symbol: str) -> Order:
        # A fill can race with timeout cancellation.
        if order_id in self._pending_orders:
            reconciled = await self.get_status(order_id, symbol)
            if reconciled.status == OrderStatus.FILLED:
                return reconciled
        order = self._pending_orders.pop(order_id, None)
        self._submitted_candles.pop(order_id, None)
        if order:
            order.status = OrderStatus.CANCELLED
            logger.info("SIM cancelled order %s", order_id)
            return order
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
            status=OrderStatus.CANCELLED,
        )

    async def get_status(self, order_id: str, symbol: str) -> Order:
        order = self._pending_orders.get(order_id)
        if order is None:
            return Order(
                order_id=order_id,
                symbol=symbol,
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=0,
                status=OrderStatus.CANCELLED,
            )

        # Recheck current price against limit price
        candle = await self._buffer.get_latest_candle(order.symbol)
        if candle is not None:
            if order.maker_preferred and candle.timestamp <= self._submitted_candles[order_id]:
                return order
            price = candle.close
            penetration = self._penetration_bps / 10000 if order.maker_preferred else 0
            can_fill = False
            if (
                order.side == Side.BUY
                and order.price is not None
                and price <= order.price * (1 - penetration)
            ):
                can_fill = True
            elif (
                order.side == Side.SELL
                and order.price is not None
                and price >= order.price * (1 + penetration)
            ):
                can_fill = True

            if can_fill:
                order.filled_price = order.price
                order.filled_quantity = order.quantity
                order.filled_at = self._now()
                order.status = OrderStatus.FILLED
                self._set_fee(order, maker=True)
                self._pending_orders.pop(order_id, None)
                self._submitted_candles.pop(order_id, None)
                logger.info(
                    "SIM LIMIT filled on recheck: %s %s qty=%.6f @ %.2f (market=%.2f)",
                    order.side.value,
                    order.symbol,
                    order.filled_quantity,
                    order.filled_price,
                    price,
                )

        return order
