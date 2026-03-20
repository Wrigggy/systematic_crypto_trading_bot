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

    def __init__(self, config: dict, buffer: LiveBuffer):
        self._slippage_bps: float = config.get("slippage_bps", 5.0)
        self._fee_bps: float = config.get("fee_bps", 10.0)
        self._buffer = buffer
        self._pending_orders: Dict[str, Order] = {}
        self._submitted_candle_ts: Dict[str, datetime] = {}

    async def execute(self, order: Order) -> Order:
        candle = await self._buffer.get_latest_candle(order.symbol)
        if candle is None:
            order.status = OrderStatus.REJECTED
            logger.warning("SimExecutor: no price data for %s", order.symbol)
            return order

        order.status = OrderStatus.SUBMITTED
        self._pending_orders[order.order_id] = order
        self._submitted_candle_ts[order.order_id] = candle.timestamp
        logger.info(
            "SIM queued %s %s qty=%.6f after candle %s (next bar execution)",
            order.side.value,
            order.symbol,
            order.quantity,
            candle.timestamp.isoformat(),
        )

        return order

    async def cancel(self, order_id: str, symbol: str) -> Order:
        order = self._pending_orders.pop(order_id, None)
        self._submitted_candle_ts.pop(order_id, None)
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
            submitted_ts = self._submitted_candle_ts.get(order_id)
            if submitted_ts is not None and candle.timestamp <= submitted_ts:
                return order

            price = candle.open
            slippage_mult = self._slippage_bps / 10000.0

            if order.order_type == OrderType.MARKET:
                if order.side == Side.BUY:
                    fill_price = price * (1 + slippage_mult)
                else:
                    fill_price = price * (1 - slippage_mult)
                order.filled_price = round(fill_price, 2)
                order.filled_quantity = order.quantity
                order.filled_at = datetime.utcnow()
                order.status = OrderStatus.FILLED
                self._pending_orders.pop(order_id, None)
                self._submitted_candle_ts.pop(order_id, None)
                logger.info(
                    "SIM %s %s qty=%.6f @ %.2f (next_open=%.2f, slip=%.1fbps)",
                    order.side.value,
                    order.symbol,
                    order.filled_quantity,
                    order.filled_price,
                    price,
                    self._slippage_bps,
                )
            elif order.order_type == OrderType.LIMIT and order.price is not None:
                touched = False
                if order.side == Side.BUY and candle.low <= order.price:
                    touched = True
                elif order.side == Side.SELL and candle.high >= order.price:
                    touched = True

                if touched:
                    order.filled_price = order.price
                    order.filled_quantity = order.quantity
                    order.filled_at = datetime.utcnow()
                    order.status = OrderStatus.FILLED
                    self._pending_orders.pop(order_id, None)
                    self._submitted_candle_ts.pop(order_id, None)
                    logger.info(
                        "SIM LIMIT filled: %s %s qty=%.6f @ %.2f (bar %.2f/%.2f)",
                        order.side.value,
                        order.symbol,
                        order.filled_quantity,
                        order.filled_price,
                        candle.low,
                        candle.high,
                    )

        return order
