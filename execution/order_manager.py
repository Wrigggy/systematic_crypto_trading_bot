from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

from core.models import Order, OrderStatus, OrderType, Side
from execution.executor import BaseExecutor
from risk.tracker import PortfolioTracker

logger = logging.getLogger(__name__)
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}


class OrderManager:
    """Book cumulative fills once; retain unresolved orders and reservations."""

    def __init__(self, executor: BaseExecutor, tracker: PortfolioTracker,
                 timeout_seconds: float = 0, journal_path: str = "", now=None):
        self._executor = executor
        self._tracker = tracker
        self._active_orders: Dict[str, Order] = {}
        self._fill_callbacks: List[Callable[[Order], None]] = []
        self._timeout_seconds = timeout_seconds
        self._error_counts: Dict[str, int] = {}
        self._booked: dict[str, tuple[float, float, float]] = {}
        self._lock = asyncio.Lock()
        self._now = now or datetime.utcnow
        self._journal = Path(journal_path) if journal_path else None
        if self._journal and self._journal.exists():
            if json.loads(self._journal.read_text()):
                raise RuntimeError("Unresolved order journal: reconcile exchange orders and balances before restart")

    def _persist(self) -> None:
        if self._journal:
            self._journal.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._journal.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: v.model_dump(mode="json")
                                       for k, v in self._active_orders.items()}, indent=2))
            tmp.replace(self._journal)

    def for_symbol(self, symbol: str) -> list[Order]:
        return [o.model_copy(deep=True) for o in self._active_orders.values() if o.symbol == symbol]

    async def submit(self, order: Order) -> Order:
        async with self._lock:
            # Serialize new entries while cash/exposure is reserved by any live order.
            if self.for_symbol(order.symbol) or (order.side == Side.BUY and self.has_pending):
                order.status = OrderStatus.REJECTED
                order.reason = "pending_order_reservation"
                self._notify(order)
                return order
            if order.side == Side.SELL:
                order.quantity = min(order.quantity, self._tracker.get_position(order.symbol).quantity)
                if order.quantity <= 0:
                    order.status = OrderStatus.REJECTED
                    self._notify(order)
                    return order
            local_id = order.order_id
            self._active_orders[local_id] = order.model_copy(deep=True)
            self._persist()  # Persist before the potentially ambiguous network write.
            try:
                updated = await self._executor.execute(order)
            except Exception:
                logger.exception("Submission unknown; blocking replacement for %s", order.symbol)
                updated = order.model_copy(update={"status": OrderStatus.UNKNOWN})
            self._active_orders.pop(local_id, None)
            self._active_orders[updated.order_id] = updated.model_copy(deep=True)
            self._apply(updated)
            return updated

    def _apply(self, updated: Order) -> None:
        previous = self._active_orders[updated.order_id]
        update = previous.model_copy(update={
            key: getattr(updated, key) for key in (
                "status", "filled_quantity", "filled_price", "filled_at", "liquidity",
                "commission", "commission_asset", "fee_bps", "exchange_acknowledged",
                "fill_time_source")
        })
        qty, notional, commission = self._booked.get(update.order_id, (0.0, 0.0, 0.0))
        new_qty = update.filled_quantity
        new_notional = new_qty * (update.filled_price or 0.0)
        new_commission = update.commission if update.commission is not None else commission
        if (not math.isfinite(new_qty) or not 0 <= new_qty <= previous.quantity + 1e-8
                or not math.isfinite(new_notional) or not math.isfinite(new_commission)
                or new_commission < 0):
            raise ValueError("Invalid cumulative fill or commission")
        if commission > 0 and update.commission_asset != previous.commission_asset:
            raise ValueError("Cumulative commission currency changed")
        if update.status == OrderStatus.FILLED and abs(new_qty - previous.quantity) > 1e-8:
            raise ValueError("FILLED order lacks complete execution quantity")
        if new_qty < qty - 1e-10 or new_commission < commission - 1e-10:
            raise ValueError("Exchange cumulative fill or fee regressed")
        delta = new_qty - qty
        if delta <= 1e-10 and abs(new_notional - notional) > 1e-8:
            raise ValueError("Cumulative notional changed without new fills")
        if delta > 1e-10:
            if not update.filled_price or new_notional <= notional:
                raise ValueError("Positive fill requires valid cumulative execution price")
            event = update.model_copy(update={
                "filled_quantity": delta,
                "filled_price": (new_notional - notional) / delta,
                "commission": new_commission - commission if update.commission is not None else None,
            })
            self._tracker.on_fill(event)
            self._booked[update.order_id] = (new_qty, new_notional, new_commission)
        elif new_commission > commission + 1e-10:
            self._tracker.on_fee(update.symbol, new_commission - commission, update.commission_asset)
            self._booked[update.order_id] = (qty, notional, new_commission)
        self._active_orders[update.order_id] = update
        if update.status in TERMINAL:
            self._active_orders.pop(update.order_id, None)
            self._booked.pop(update.order_id, None)
            self._error_counts.pop(update.order_id, None)
        self._persist()
        if delta > 1e-10 or update.status in TERMINAL:
            self._notify(update)

    def _notify(self, order: Order) -> None:
        for cb in self._fill_callbacks:
            try:
                cb(order.model_copy(deep=True))
            except Exception:
                logger.exception("Order event callback failed")

    async def cancel(self, order_id: str) -> None:
        async with self._lock:
            await self._cancel(order_id)

    async def _cancel(self, order_id: str) -> None:
        order = self._active_orders.get(order_id)
        if order is None:
            return
        try:
            self._apply(await self._executor.cancel(order_id, order.symbol))
        except Exception:
            logger.warning("Cancel unresolved for %s; order retained", order_id, exc_info=True)

    async def cancel_symbol(self, symbol: str) -> bool:
        for order in self.for_symbol(symbol):
            await self.cancel(order.order_id)
        return not self.for_symbol(symbol)

    async def check_pending(self) -> None:
        async with self._lock:
            for order_id in list(self._active_orders):
                order = self._active_orders[order_id]
                age = (self._now() - order.created_at).total_seconds()
                if (order.order_type == OrderType.LIMIT and self._timeout_seconds > 0
                        and age >= self._timeout_seconds):
                    await self._cancel(order_id)
                    continue
                try:
                    self._apply(await self._executor.get_status(order_id, order.symbol))
                    self._error_counts.pop(order_id, None)
                except Exception:
                    self._error_counts[order_id] = self._error_counts.get(order_id, 0) + 1
                    logger.warning("Status unresolved for %s; retaining reservation", order_id)

    async def cancel_all(self) -> None:
        for order_id in list(self._active_orders):
            await self.cancel(order_id)

    def register_fill_callback(self, cb: Callable[[Order], None]) -> None:
        self._fill_callbacks.append(cb)

    @property
    def active_orders(self) -> Dict[str, Order]:
        return {key: order.model_copy(deep=True) for key, order in self._active_orders.items()}

    @property
    def has_pending(self) -> bool:
        return bool(self._active_orders)
