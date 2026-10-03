"""Engineering fill model with independent venue evolution and request accounting.

Not a Roostoo matching-engine replica. No queue, volume, partial-fill or latency
calibration is claimed. Maker fills require a later completed close to penetrate
the limit; taker fills use the latest sufficiently fresh observed price.
"""

from datetime import datetime

from core.models import OrderStatus
from execution.executor import BaseExecutor
from execution.request_budget import RequestBudget, RequestBudgetExceeded
from execution.sim_executor import SimExecutor


class ReplayExecutor(BaseExecutor):
    def __init__(self, config, buffer, now):
        self.now, self.buffer = now, buffer
        self.sim = SimExecutor(
            config, buffer, now=lambda: datetime.utcfromtimestamp(now())
        )
        self.budget = RequestBudget(
            config.get("max_requests_per_minute", 5),
            config.get("emergency_request_reserve", 1),
            now,
        )
        self.max_price_age = config.get("max_price_age_seconds", 120)
        self.receipts = {}
        self.cancel_requested = set()

    async def observe(self):
        """Exchange-side fills happen even when the client cannot afford a poll."""
        for order_id, order in list(self.receipts.items()):
            if order.status in {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
                self.receipts[order_id] = (
                    await self.sim.get_status(order_id, order.symbol)
                ).model_copy(deep=True)

    async def execute(self, order):
        candle = await self.buffer.get_latest_candle(order.symbol)
        if (
            candle is None
            or (
                datetime.utcfromtimestamp(self.now()) - candle.timestamp
            ).total_seconds()
            > self.max_price_age
        ):
            return order.model_copy(
                update={
                    "status": OrderStatus.REJECTED,
                    "reason": "stale_execution_price",
                }
            )
        try:
            if order.maker_preferred and not order.urgent:
                self.budget.take()  # Venue quote request, modeled without L2 data.
            self.budget.take(urgent=order.urgent)
        except RequestBudgetExceeded:
            return order.model_copy(
                update={
                    "status": OrderStatus.REJECTED,
                    "reason": "request_budget_deferred",
                }
            )
        result = await self.sim.execute(order)
        self.receipts[result.order_id] = result.model_copy(deep=True)
        return result

    async def get_status(self, order_id, symbol):
        order = self.receipts[order_id]
        self.budget.take(urgent=order.urgent or order_id in self.cancel_requested)
        return order.model_copy(deep=True)

    async def cancel(self, order_id, symbol):
        if order_id not in self.cancel_requested:
            self.budget.take(urgent=True)
            order = self.receipts[order_id]
            if order.status in {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
                order = await self.sim.cancel(order_id, symbol)
                self.receipts[order_id] = order.model_copy(deep=True)
            self.cancel_requested.add(order_id)
        # Cancellation and reconciliation are two independently budgeted requests.
        result = await self.get_status(order_id, symbol)
        self.cancel_requested.discard(order_id)
        return result
