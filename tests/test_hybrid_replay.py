"""Offline end-to-end replay fixtures, not empirical strategy results."""

import pytest

from scripts.replay_hybrid import ReplayEvent, run_replay
from tests.test_hybrid import book_config, packet


def fixture(mode="combined"):
    start = 24 * 3600
    config = dict(
        strategy=dict(
            decision_interval_seconds=60,
            lookback_days=1,
            sample_hours=8,
            rules={"BTC/USDT": dict(entry_sigma=1, exit_z=0)},
            fusion={"mode": mode},
        ),
        forecast=book_config(),
        execution={},
        replay=dict(start_at=start, end_at=start + 15000, timer_seconds=5),
    )
    events = [
        ReplayEvent(
            kind="hourly",
            symbol="BTC/USDT",
            timestamp=k * 3600,
            available_at=k * 3600,
            close=100 + (k % 3),
        )
        for k in range(1, 25)
    ]
    for t, value in ((start - 120, 0), (start - 60, 2), (start, 5)):
        p = packet(t, value)
        events.append(
            ReplayEvent(
                kind="forecast",
                timestamp=t,
                available_at=t,
                symbol=p.symbol,
                forecast=p,
            )
        )
    for t, price in (
        (start, 99),
        (start + 30, 98.9),
        (start + 60, 102),
        (start + 90, 102.1),
    ):
        events.append(
            ReplayEvent(
                kind="price",
                timestamp=t,
                available_at=t,
                symbol="BTC/USDT",
                close=price,
            )
        )
    return config, sorted(events, key=lambda e: e.available_at)


@pytest.mark.asyncio
async def test_combined_replay_has_a_real_decision_fill_exit_chain():
    config, events = fixture()
    result = await run_replay(config, events)
    kinds = [e["event"] for e in result["events"]]
    assert (
        "entry_intent" in kinds
        and "position_opened" in kinds
        and "position_closed" in kinds
    )
    assert result["summary"]["completed_holds"] == 1
    assert result["summary"]["active_fill_days_utc8"] == 1
    assert not result["summary"]["positions"]
    assert result["summary"]["requests"] > 0
    candidate = next(e for e in result["events"] if e["event"] == "price_candidate")
    assert candidate["fusion"]["model_strength"] is not None
    assert candidate["fusion"]["rule_strength"] > 0


@pytest.mark.asyncio
async def test_missing_forecasts_blocks_combined_but_not_explicit_price_only_control():
    config, events = fixture()
    events = [e for e in events if e.kind != "forecast"]
    result = await run_replay(config, events)
    assert not any(e["event"] == "entry_intent" for e in result["events"])
    config["strategy"]["fusion"]["mode"] = "price_only"
    result = await run_replay(config, events)
    assert any(e["event"] == "entry_intent" for e in result["events"])


@pytest.mark.asyncio
async def test_cash_start_and_no_fabricated_terminal_liquidation():
    config, events = fixture()
    start = config["replay"]["start_at"]
    config["replay"]["end_at"] = start + 85
    result = await run_replay(config, events)
    assert result["summary"]["positions"]
    assert result["summary"]["completed_holds"] == 0
    assert result["summary"]["open_hold_age_seconds"]["BTC/USDT"] == 55
    assert all(
        e["timestamp"] >= start
        for e in result["events"]
        if e["event"] == "entry_intent"
    )


@pytest.mark.asyncio
async def test_budget_delayed_fill_receipt_is_not_claimed_to_be_flat():
    config, events = fixture()
    config["replay"]["end_at"] = config["replay"]["start_at"] + 55
    result = await run_replay(config, events)
    assert result["summary"]["accounting_provisional"]
    assert result["summary"]["pending_orders"]
    assert result["summary"]["venue_fills_not_yet_reconciled"]


def test_invalid_data_availability_rejected():
    with pytest.raises(ValueError, match="before"):
        ReplayEvent(
            kind="price", symbol="BTC/USDT", close=100, timestamp=2, available_at=1
        )


@pytest.mark.asyncio
async def test_replay_cannot_fall_through_into_live_bot():
    from main import main

    config, _ = fixture()
    config["strategy"]["engine"] = "hybrid"
    with pytest.raises(ValueError, match="research-only"):
        await main(config)
