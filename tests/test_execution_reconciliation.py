"""Mock-only execution contract tests; no network requests or credentials."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from core.models import Order, OrderStatus, OrderType, Side
from execution.order_manager import OrderManager
from execution.request_budget import RequestBudget, RequestBudgetExceeded
from plugins.roostoo.executor import RoostooExecutor
from risk.tracker import PortfolioTracker


def entry(**updates):
    data = dict(
        order_id="local",
        symbol="BTC/USDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        price=100,
        maker_preferred=True,
    )
    data.update(updates)
    return Order(**data)


def receipt(order, qty, price=100, fee=0, status=OrderStatus.PARTIALLY_FILLED):
    return order.model_copy(
        update=dict(
            filled_quantity=qty,
            filled_price=price,
            filled_at=datetime.utcnow(),
            commission=fee,
            commission_asset="USDT",
            status=status,
        )
    )


def test_all_endpoints_share_budget_and_urgent_never_bypasses_ceiling():
    clock = [0]
    budget = RequestBudget(5, 1, lambda: clock[0])
    for _ in range(4):
        budget.take()
    with pytest.raises(RequestBudgetExceeded):
        budget.take()
    budget.take(urgent=True)
    with pytest.raises(RequestBudgetExceeded):
        budget.take(urgent=True)
    clock[0] = 60
    budget.take()
    assert budget.total == 6


@pytest.mark.asyncio
async def test_cumulative_partial_and_cancel_race_book_quantity_vwap_and_fees_once():
    executor = AsyncMock()
    tracker = PortfolioTracker(1000)
    manager = OrderManager(executor, tracker)
    original = entry()
    executor.execute.return_value = original.model_copy(
        update={"status": OrderStatus.SUBMITTED}
    )
    await manager.submit(original)
    partial = receipt(original, 1, 100, 0.05)
    executor.get_status.return_value = partial
    await manager.check_pending()
    await manager.check_pending()
    assert tracker.snapshot().cash == pytest.approx(899.95)
    executor.cancel.return_value = receipt(original, 2, 101, 0.101, OrderStatus.FILLED)
    await manager.cancel(original.order_id)
    assert not manager.has_pending
    assert tracker.get_position("BTC/USDT").quantity == 2
    assert tracker.get_position("BTC/USDT").entry_price == 101
    assert tracker.snapshot().cash == pytest.approx(797.899)


@pytest.mark.asyncio
async def test_partial_cancel_leaves_real_inventory_and_unknown_cancel_blocks_replacement():
    executor, tracker = AsyncMock(), PortfolioTracker(1000)
    manager = OrderManager(executor, tracker)
    original = entry()
    executor.execute.return_value = receipt(original, 1, fee=0.05)
    await manager.submit(original)
    executor.cancel.side_effect = TimeoutError("ambiguous cancel")
    assert not await manager.cancel_symbol(original.symbol)
    result = await manager.submit(entry(side=Side.SELL, order_id="second", quantity=1))
    assert result.status == OrderStatus.REJECTED
    assert executor.execute.await_count == 1
    executor.cancel.side_effect = None
    executor.cancel.return_value = receipt(
        original, 1, fee=0.05, status=OrderStatus.CANCELLED
    )
    assert await manager.cancel_symbol(original.symbol)
    assert tracker.get_position(original.symbol).quantity == 1


@pytest.mark.asyncio
async def test_unknown_submission_is_journaled_and_never_retried(tmp_path):
    executor, tracker = AsyncMock(), PortfolioTracker(1000)
    executor.execute.side_effect = TimeoutError("request may have arrived")
    journal = str(tmp_path / "orders.json")
    manager = OrderManager(executor, tracker, journal_path=journal)
    assert (await manager.submit(entry())).status == OrderStatus.UNKNOWN
    assert (
        await manager.submit(entry(order_id="second"))
    ).status == OrderStatus.REJECTED
    assert executor.execute.await_count == 1
    with pytest.raises(RuntimeError, match="reconcile"):
        OrderManager(executor, tracker, journal_path=journal)


def test_quote_and_base_commission_accounting_and_atomic_oversell_guard():
    tracker = PortfolioTracker(1000)
    buy = receipt(entry(), 2, fee=0.001, status=OrderStatus.FILLED)
    buy.commission_asset = "BTC"
    tracker.on_fill(buy)
    assert tracker.get_position(buy.symbol).quantity == pytest.approx(1.999)
    assert tracker.snapshot().cash == 800
    before = tracker.snapshot().model_dump()
    sell = receipt(entry(side=Side.SELL), 2, fee=0.1, status=OrderStatus.FILLED)
    with pytest.raises(ValueError, match="exceeds"):
        tracker.on_fill(sell)
    assert tracker.snapshot().cash == before["cash"]
    assert tracker.get_position(buy.symbol).realized_pnl == 0


@pytest.fixture
def venue():
    executor = RoostooExecutor({})
    executor._pair_info["BTC/USDT"] = dict(
        qty_precision=3, price_precision=2, min_notional=1, can_trade=True
    )
    executor.get_quote = AsyncMock(return_value={"bid": 99.999, "ask": 100.011})
    executor._signed_request = AsyncMock(
        return_value={
            "Success": True,
            "OrderDetail": {"OrderID": 12, "Status": "PENDING", "FilledQuantity": 0},
        }
    )
    return executor


@pytest.mark.asyncio
@pytest.mark.parametrize("side,expected", [(Side.BUY, "99.99"), (Side.SELL, "100.02")])
async def test_passive_quote_rounds_outward_and_acknowledgement_is_not_fill(
    venue, side, expected
):
    order = await venue.execute(entry(side=side, price=None))
    assert order.status == OrderStatus.SUBMITTED and order.filled_quantity == 0
    assert order.exchange_acknowledged and order.order_id == "12"
    params = venue._signed_request.call_args.args[2]
    assert params["type"] == "LIMIT" and params["price"] == expected


@pytest.mark.asyncio
async def test_venue_ambiguous_post_not_retried_or_marked_cancelled(venue):
    venue._signed_request.return_value = None
    result = await venue.execute(entry())
    assert result.status == OrderStatus.UNKNOWN
    venue._signed_request.assert_awaited_once()
    with pytest.raises(RuntimeError, match="explicit reconciliation"):
        await venue.get_status(result.order_id, result.symbol)


@pytest.mark.asyncio
async def test_cancel_queries_by_id_only_and_reconciles_racing_fill(venue):
    submitted = await venue.execute(entry())
    venue._signed_request.reset_mock()
    venue._signed_request.side_effect = [
        {"Success": True},
        {
            "Success": True,
            "OrderMatched": [
                {
                    "OrderID": 12,
                    "Status": "FILLED",
                    "FilledQuantity": 2,
                    "FilledAverPrice": 100,
                    "CommissionChargeValue": 0.1,
                    "CommissionCoin": "USDT",
                    "Role": "MAKER",
                }
            ],
        },
    ]
    result = await venue.cancel(submitted.order_id, submitted.symbol)
    assert result.status == OrderStatus.FILLED and result.commission == 0.1
    assert result.liquidity == "MAKER"
    assert all(
        call.args[2] == {"order_id": "12"}
        for call in venue._signed_request.call_args_list
    )


@pytest.mark.asyncio
async def test_invalid_filled_receipt_retains_acknowledged_unknown_order(venue):
    venue._signed_request.return_value = {
        "Success": True,
        "OrderDetail": {
            "OrderID": 12,
            "Status": "FILLED",
            "FilledQuantity": 2,
            "FilledAverPrice": 100,
        },
    }
    result = await venue.execute(entry())
    assert result.exchange_acknowledged and result.status == OrderStatus.UNKNOWN


@pytest.mark.asyncio
async def test_venue_completion_time_is_not_replaced_by_poll_time(venue):
    venue._signed_request.return_value = {
        "Success": True,
        "OrderDetail": {
            "OrderID": 12,
            "Status": "FILLED",
            "FilledQuantity": 2,
            "FilledAverPrice": 100,
            "CommissionChargeValue": 0.1,
            "CommissionCoin": "USDT",
            "FinishTimestamp": 1570224271590,
            "Role": "MAKER",
        },
    }
    result = await venue.execute(entry())
    assert result.fill_time_source == "venue_finish"
    assert result.filled_at == datetime.utcfromtimestamp(1570224271.590)


@pytest.mark.asyncio
async def test_nonfinite_receipt_never_contaminates_cash(venue):
    executor, tracker = AsyncMock(), PortfolioTracker(1000)
    manager = OrderManager(executor, tracker)
    original = entry()
    executor.execute.return_value = original.model_copy(
        update={"status": OrderStatus.SUBMITTED}
    )
    await manager.submit(original)
    executor.get_status.return_value = receipt(original, float("nan"))
    await manager.check_pending()
    assert tracker.snapshot().cash == 1000
    assert tracker.get_position(original.symbol).quantity == 0
    assert manager.has_pending
