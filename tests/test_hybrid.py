"""Synthetic checks, not evidence of predictive or trading performance."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from core.models import OHLCV, Order, OrderStatus, OrderType, Side
from data.buffer import LiveBuffer
from execution.order_manager import OrderManager
from execution.sim_executor import SimExecutor
from plugins.model_inference.forecasts import CausalScores, ForecastBook, ForecastPacket
from risk.tracker import PortfolioTracker
from strategy.fusion import SignalFusion
from strategy.hybrid import Holding, HybridCoordinator, HourlyHistory


def packet(t, value, symbol="BTC/USDT", **updates):
    data = dict(
        symbol=symbol,
        as_of=t,
        data_cutoff=t,
        available_at=t,
        valid_until=t + 120,
        eligible_after=0,
        model_version="fixture",
        preprocessing_version="fixture",
        heads=[
            dict(
                name="short",
                window_start_seconds=450,
                window_end_seconds=900,
                prediction=value,
            ),
            dict(
                name="long",
                window_start_seconds=7200,
                window_end_seconds=14400,
                prediction=value,
            ),
        ],
    )
    data.update(updates)
    return ForecastPacket(**data)


def book_config():
    return dict(
        model_version="fixture",
        preprocessing_version="fixture",
        max_data_age_seconds=120,
        long_heads=["long"],
        normalization=dict(
            window_seconds=600, min_samples=2, min_span_seconds=60, max_gap_seconds=300
        ),
        heads=[
            dict(
                name=h.name,
                window_start_seconds=h.window_start_seconds,
                window_end_seconds=h.window_end_seconds,
                target=h.target,
                weight=1,
            )
            for h in packet(0, 0).heads
        ],
    )


def test_normalization_excludes_current_and_is_asset_specific():
    scores = CausalScores(
        window_seconds=600, min_samples=2, min_span_seconds=60, max_gap_seconds=300
    )
    assert scores.update(packet(0, 0)) is None
    assert scores.update(packet(60, 2)) is None
    assert scores.update(packet(120, 5))["short"] == 4
    assert scores.update(packet(120, 5, "XRP/USDT")) is None
    with pytest.raises(ValueError, match="advance strictly"):
        scores.update(packet(120, 5))
    assert scores.update(packet(500, 5)) is None


@pytest.mark.parametrize(
    "update",
    [
        dict(data_cutoff=101),
        dict(available_at=99),
        dict(eligible_after=101),
        dict(valid_until=100),
        dict(schema_version=2),
    ],
)
def test_noncausal_packet_rejected(update):
    with pytest.raises(ValueError):
        packet(100, 0, **update)


def test_forecast_admission_versions_windows_staleness_and_cold_start():
    book = ForecastBook(book_config())
    book.ingest(packet(0, 0), 0)
    assert book.signal("BTC/USDT", 0) is None
    book.ingest(packet(60, 2), 60)
    book.ingest(packet(120, 5), 120)
    assert book.signal("BTC/USDT", 120)["composite"] == 4
    assert book.signal("BTC/USDT", 119) is None
    assert book.signal("BTC/USDT", 240) is None
    for bad in (
        packet(180, 5, model_version="wrong"),
        packet(180, 5, available_at=181),
        packet(1, 5),
    ):
        with pytest.raises(ValueError):
            book.ingest(bad, 180)
    bad = packet(180, 5)
    bad.heads[0].window_end_seconds += 1
    with pytest.raises(ValueError, match="definition"):
        book.ingest(bad, 180)
    bad = packet(180, 5)
    bad.heads[0].name = "missing"
    with pytest.raises(ValueError, match="heads"):
        book.ingest(bad, 180)


def test_fusion_is_not_merely_model_filter():
    combined = SignalFusion({})
    model = dict(composite=1.5, long_support=1)
    deep = combined.evaluate(80, 100, 10, 1, model)
    shallow = combined.evaluate(89, 100, 10, 1, model)
    assert deep.eligible and deep.joint_strength == 1.25
    assert not shallow.eligible
    assert (
        not SignalFusion({"mode": "model_filter"})
        .evaluate(80, 100, 10, 1, model)
        .eligible
    )
    assert combined.evaluate(89, 100, 10, 1, dict(composite=3, long_support=1)).eligible
    assert not combined.evaluate(
        101, 100, 10, 1, dict(composite=30, long_support=10)
    ).eligible


def test_model_never_replaces_rules_or_missing_forecasts():
    fusion = SignalFusion({})
    assert not fusion.evaluate(80, 100, 10, 1, None).eligible
    assert not fusion.evaluate(
        80, 100, 10, 1, dict(composite=5, long_support=-1)
    ).eligible
    assert SignalFusion({"mode": "price_only"}).evaluate(80, 100, 10, 1, None).eligible


def test_hourly_history_requires_continuous_completed_observations():
    history = HourlyHistory(1, 8)
    now = 24 * 3600
    for k in range(1, 25):
        history.add("BTC/USDT", k * 3600, 100 + k, now)
    assert history.bands("BTC/USDT", now) is not None
    del history.values["BTC/USDT"][3600]
    assert history.bands("BTC/USDT", now) is None
    with pytest.raises(ValueError):
        history.add("BTC/USDT", now + 3600, 100, now)


async def setup_hybrid(**overrides):
    clock = [24 * 3600]
    buffer = LiveBuffer()
    tracker = PortfolioTracker(100_000)
    executor = SimExecutor(
        {"slippage_bps": 0}, buffer, now=lambda: datetime.utcfromtimestamp(clock[0])
    )
    manager = OrderManager(
        executor,
        tracker,
        timeout_seconds=120,
        now=lambda: datetime.utcfromtimestamp(clock[0]),
    )
    config = dict(
        decision_interval_seconds=60,
        lookback_days=1,
        sample_hours=8,
        rules={"BTC/USDT": dict(entry_sigma=1, exit_z=0)},
        fusion={"mode": "price_only"},
    )
    config.update(overrides)
    book = ForecastBook(book_config())
    coordinator = HybridCoordinator(config, book, manager, tracker, lambda: clock[0])
    for k in range(1, 25):
        coordinator.history.add("BTC/USDT", k * 3600, 100 + (k % 3), clock[0])

    async def step(t, price, **kwargs):
        clock[0] = t
        await buffer.push_candle(
            OHLCV(
                symbol="BTC/USDT",
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1,
                timestamp=datetime.utcfromtimestamp(t),
            )
        )
        await coordinator.step({"BTC/USDT": (t, price)}, **kwargs)

    return clock, tracker, manager, coordinator, step


@pytest.mark.asyncio
async def test_actual_fill_starts_hold_and_four_hour_exit_survives_missing_model():
    clock, tracker, manager, hybrid, step = await setup_hybrid()
    start = clock[0]
    await step(start, 99)
    assert manager.has_pending and not hybrid.holdings
    await step(start + 30, 98.9)
    assert hybrid.holdings["BTC/USDT"].opened_at == start + 30
    # No new hourly history or forecasts; deadline still closes real inventory.
    await step(start + 30 + 14400, 99)
    assert tracker.get_position("BTC/USDT").quantity == 0
    close = [e for e in hybrid.events if e["event"] == "position_closed"][-1]
    assert close["holding_seconds"] == 14400
    assert close["reason"] == "holding_deadline"


@pytest.mark.asyncio
async def test_price_recovery_has_no_three_hour_minimum_and_missing_bands_do_not_block_exit():
    clock, tracker, manager, hybrid, step = await setup_hybrid()
    start = clock[0]
    await step(start, 99)
    await step(start + 30, 98.9)
    hybrid.history.values.clear()
    await step(start + 60, 102)
    assert manager.for_symbol("BTC/USDT")[0].side == Side.SELL
    assert tracker.get_position("BTC/USDT").quantity > 0
    await step(start + 90, 102.1)
    assert tracker.get_position("BTC/USDT").quantity == 0


@pytest.mark.asyncio
async def test_ordinary_exit_timeout_switches_to_taker_after_reconciliation():
    clock, tracker, manager, hybrid, step = await setup_hybrid()
    start = clock[0]
    await step(start, 99)
    await step(start + 30, 98.9)
    await step(start + 60, 102)
    await step(start + 180, 102)
    assert tracker.get_position("BTC/USDT").quantity == 0
    final = [e["order"] for e in hybrid.events if e["event"] == "order_event"][-1]
    assert final["liquidity"] == "TAKER" and final["reason"] == "exit_timeout"


@pytest.mark.asyncio
async def test_stop_closes_without_forecast_or_entry_history():
    clock, tracker, manager, hybrid, step = await setup_hybrid()
    start = clock[0]
    await step(start, 99)
    await step(start + 30, 98.9)
    hybrid.history.values.clear()
    await step(start + 60, 95)
    assert tracker.get_position("BTC/USDT").quantity == 0
    assert any(e.get("reason") == "price_stop" for e in hybrid.events)


@pytest.mark.asyncio
async def test_required_cadence_and_no_fallback_when_model_missing():
    with pytest.raises(ValueError, match="undecided"):
        await setup_hybrid(decision_interval_seconds=None)
    clock, tracker, manager, hybrid, step = await setup_hybrid(
        fusion={"mode": "combined"}
    )
    await step(clock[0], 99)
    assert not manager.has_pending and tracker.snapshot().cash == 100_000


@pytest.mark.asyncio
async def test_first_partial_receipt_time_is_used_not_submission_or_poll_time():
    clock, tracker, manager, hybrid, step = await setup_hybrid()
    hybrid.pending_targets["BTC/USDT"] = 101
    order = Order(
        symbol="BTC/USDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=2,
        filled_quantity=1,
        filled_price=99,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_at=datetime.utcfromtimestamp(clock[0] - 20),
    )
    tracker.on_fill(order)
    hybrid._on_order(order)
    assert hybrid.holdings["BTC/USDT"].opened_at == clock[0] - 20


@pytest.mark.asyncio
async def test_joint_signal_ranks_candidates_before_shared_cash_allocation():
    clock, tracker, manager, hybrid, step = await setup_hybrid(
        fusion={"mode": "combined"},
        rules={
            "BTC/USDT": dict(entry_sigma=1, exit_z=0),
            "XRP/USDT": dict(entry_sigma=1, exit_z=0),
        },
    )
    hybrid.history.values["XRP/USDT"] = dict(hybrid.history.values["BTC/USDT"])
    hybrid.book.signal = lambda symbol, now: dict(composite=4, long_support=1)
    hybrid._enter = AsyncMock()
    await hybrid.step({"BTC/USDT": (clock[0], 99), "XRP/USDT": (clock[0], 98)})
    assert [call.args[0] for call in hybrid._enter.call_args_list] == [
        "XRP/USDT",
        "BTC/USDT",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        {"fusion": {"mode": "price_only"}},
        {"fusion": {"mode": "combined"}, "model_exit_enabled": False},
    ],
)
async def test_price_only_and_entry_ablation_never_consult_model_for_exit(settings):
    clock, tracker, manager, hybrid, step = await setup_hybrid(**settings)
    hybrid.holdings["BTC/USDT"] = Holding(clock[0] - 12000, 101, 99, 99)
    hybrid._exit = AsyncMock()
    await hybrid._holding_decision("BTC/USDT", 99, {"long_support": -3}, clock[0])
    await hybrid._holding_decision("BTC/USDT", 99, {"long_support": -3}, clock[0] + 600)
    hybrid._exit.assert_not_awaited()
