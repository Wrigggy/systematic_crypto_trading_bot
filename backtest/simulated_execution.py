from __future__ import annotations

from datetime import datetime

from core.models import Order, OrderStatus, OrderType, Side
from execution.order_manager import OrderManager
from execution.sim_executor import SimExecutor


class BacktestSimExecutor(SimExecutor):
    """SimExecutor variant that timestamps fills with simulated candle time."""

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
                order.filled_at = candle.timestamp
                order.status = OrderStatus.FILLED
                self._pending_orders.pop(order_id, None)
                self._submitted_candle_ts.pop(order_id, None)
            elif order.order_type == OrderType.LIMIT and order.price is not None:
                touched = False
                if order.side == Side.BUY and candle.low <= order.price:
                    touched = True
                elif order.side == Side.SELL and candle.high >= order.price:
                    touched = True

                if touched:
                    order.filled_price = order.price
                    order.filled_quantity = order.quantity
                    order.filled_at = candle.timestamp
                    order.status = OrderStatus.FILLED
                    self._pending_orders.pop(order_id, None)
                    self._submitted_candle_ts.pop(order_id, None)

        return order


class BacktestOrderManager(OrderManager):
    """OrderManager variant that uses simulated time for limit-order ageing."""

    async def submit_at(self, order: Order, current_time: datetime) -> Order:
        order.created_at = current_time
        return await self.submit(order)

    async def check_pending_at(self, now: datetime) -> None:
        to_remove = []
        to_cancel = []

        if self._timeout_seconds > 0:
            for order_id, order in self._active_orders.items():
                age = (now - order.created_at).total_seconds()
                if age > self._timeout_seconds:
                    to_cancel.append(order_id)

        for order_id in to_cancel:
            await self.cancel(order_id)

        for order_id, order in self._active_orders.items():
            try:
                prev_qty = order.filled_quantity or 0.0
                prev_avg = order.filled_price or 0.0
                updated = await self._executor.get_status(order_id, order.symbol)
                if updated.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    self._apply_fill_update(
                        order,
                        cumulative_filled_qty=updated.filled_quantity,
                        cumulative_avg_price=updated.filled_price,
                        status=updated.status,
                        previous_filled_qty=prev_qty,
                        previous_avg_price=prev_avg,
                    )
                    if updated.status == OrderStatus.FILLED:
                        to_remove.append(order_id)
                elif updated.status == OrderStatus.CANCELLED:
                    order.status = OrderStatus.CANCELLED
                    self._notify_callbacks(order)
                    to_remove.append(order_id)
                self._error_counts.pop(order_id, None)
            except Exception:
                self._error_counts[order_id] = self._error_counts.get(order_id, 0) + 1
                if self._error_counts[order_id] >= self._max_errors:
                    order.status = OrderStatus.CANCELLED
                    to_remove.append(order_id)
                    self._notify_callbacks(order)

        for order_id in to_remove:
            self._active_orders.pop(order_id, None)
