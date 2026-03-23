from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.models import Order, Side
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker


class BacktestRiskShield(RiskShield):
    """RiskShield variant that uses simulated timestamps for order rate limiting."""

    def validate_at(
        self,
        order: Order,
        tracker: PortfolioTracker,
        current_time: datetime,
        market_price: float = 0.0,
        is_stop: bool = False,
    ) -> Optional[Order]:
        snapshot = tracker.snapshot()

        if self._circuit_breaker_active and order.side == Side.BUY:
            return None

        if order.side == Side.SELL:
            pos = tracker.get_position(order.symbol)
            if pos.quantity < order.quantity:
                order.quantity = pos.quantity
            if order.quantity <= 0:
                return None

        if not is_stop:
            now = current_time.timestamp()
            while self._order_timestamps and self._order_timestamps[0] < now - 60:
                self._order_timestamps.popleft()
            if len(self._order_timestamps) >= self._max_orders_per_minute:
                return None
            self._order_timestamps.append(now)

        if order.side == Side.BUY:
            price = order.price or market_price or 0.0
            if price <= 0:
                pos = tracker.get_position(order.symbol)
                price = pos.current_price if pos.current_price > 0 else 1.0

            order_value = price * order.quantity
            current_exposure = tracker.get_exposure(order.symbol)
            new_single_exposure = current_exposure + (
                order_value / snapshot.nav if snapshot.nav > 0 else 1.0
            )
            if new_single_exposure > self._max_single_exposure:
                max_value = (self._max_single_exposure - current_exposure) * snapshot.nav
                if max_value <= 0:
                    return None
                order.quantity = max_value / price
                order_value = price * order.quantity

            total_exposure = tracker.get_total_exposure()
            new_total = total_exposure + (
                order_value / snapshot.nav if snapshot.nav > 0 else 1.0
            )
            if new_total > self._max_portfolio_exposure:
                max_value = (self._max_portfolio_exposure - total_exposure) * snapshot.nav
                if max_value <= 0:
                    return None
                order.quantity = min(order.quantity, max_value / price)

            total_cost = price * order.quantity * 1.001
            if total_cost > snapshot.cash:
                order.quantity = (snapshot.cash * 0.999) / price
                if order.quantity <= 0:
                    return None

        return order
