"""Run causal, offline hybrid replay. Never connects to an exchange or a machine.

Usage: python -m scripts.replay_hybrid --config config/hybrid.yaml --output logs/replay.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from statistics import median
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.models import OHLCV
from data.buffer import LiveBuffer
from execution.order_manager import OrderManager
from execution.replay_executor import ReplayExecutor
from plugins.model_inference.forecasts import ForecastBook, ForecastPacket
from risk.tracker import PortfolioTracker
from strategy.hybrid import HybridCoordinator


class ReplayEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["price", "hourly", "forecast"]
    available_at: int = Field(ge=0)
    timestamp: int = Field(ge=0)
    symbol: str
    close: float | None = Field(default=None, gt=0)
    forecast: ForecastPacket | None = None

    @model_validator(mode="after")
    def check_event(self):
        if self.timestamp > self.available_at:
            raise ValueError("Event is available before its data timestamp")
        if self.kind == "forecast":
            p = self.forecast
            if (
                p is None
                or self.close is not None
                or (p.available_at, p.as_of, p.symbol)
                != (self.available_at, self.timestamp, self.symbol)
            ):
                raise ValueError("Forecast envelope mismatch")
        elif self.close is None or self.forecast is not None:
            raise ValueError(
                "Price events require a close and cannot contain forecasts"
            )
        if self.kind == "hourly" and self.timestamp % 3600:
            raise ValueError("Hourly timestamp must be the completed hour boundary")
        return self


def read_events(path):
    previous = -1
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = ReplayEvent.model_validate_json(line)
            if event.available_at < previous:
                raise ValueError(f"Events out of availability order at line {number}")
            previous = event.available_at
            yield event


async def run_replay(config, events):
    spec = config["replay"]
    start, end = spec.get("start_at"), spec.get("end_at")
    timer = spec.get("timer_seconds", 5)
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        raise ValueError("Explicit replay start_at and end_at are required")
    if not isinstance(timer, int) or not 1 <= timer <= 60:
        raise ValueError("Risk timer must be between one and sixty seconds")
    entry_cutoff = spec.get('entry_cutoff_seconds', 0)
    closeout = spec.get('closeout_seconds', 0)
    if (not isinstance(entry_cutoff, int) or not isinstance(closeout, int)
            or not 0 <= closeout <= entry_cutoff < end - start):
        raise ValueError('Invalid independent-window closeout schedule')
    capital = float(spec.get("initial_capital", 100000))
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("Invalid initial capital")
    clock = [start]
    buffer = LiveBuffer(max_candles=2)
    tracker = PortfolioTracker(capital)
    executor = ReplayExecutor(config["execution"], buffer, lambda: clock[0])
    manager = OrderManager(
        executor,
        tracker,
        timeout_seconds=config["execution"].get("order_timeout_seconds", 120),
        now=lambda: datetime.utcfromtimestamp(clock[0]),
    )
    book = ForecastBook(config["forecast"])
    hybrid = HybridCoordinator(
        config["strategy"], book, manager, tracker, lambda: clock[0]
    )
    curve, audit = [], []
    last_prices = {}
    last_available = -1
    next_timer = start

    async def tick(t, prices=None):
        clock[0] = t
        if closeout and t >= end - closeout:
            # Close inside the window through the real budgeted order lifecycle.
            # Never grant a free terminal fill or reuse inventory in the next run.
            for symbol in set(hybrid.holdings) | {
                o.symbol for o in manager.active_orders.values()
            }:
                hybrid.urgent[symbol] = 'window_closeout'
        await hybrid.step(prices, allow_entries=t < end - entry_cutoff)
        snap = tracker.snapshot()
        point = dict(timestamp=t, nav=snap.nav, exposure=tracker.get_total_exposure())
        if curve and curve[-1]["timestamp"] == t:
            curve[-1] = point
        else:
            curve.append(point)
        audit.extend(hybrid.events)
        hybrid.events.clear()

    for available_at, group in groupby(events, key=lambda e: e.available_at):
        if available_at < last_available:
            raise ValueError("Events must be sorted by availability, not data time")
        last_available = available_at
        if available_at >= end:
            break
        while next_timer < available_at:
            await tick(next_timer)
            next_timer += timer
        clock[0] = available_at
        updates = {}
        for event in group:
            if event.symbol not in hybrid.rules:
                raise ValueError("Replay symbol outside declared trading universe")
            if event.kind == "forecast":
                hybrid.ingest_forecast(event.forecast)
            elif event.kind == "hourly":
                hybrid.history.add(
                    event.symbol, event.timestamp, event.close, available_at
                )
            else:
                if event.timestamp <= last_prices.get(event.symbol, -1):
                    raise ValueError("Price events must strictly advance per asset")
                last_prices[event.symbol] = event.timestamp
                updates[event.symbol] = (event.timestamp, event.close)
                await buffer.push_candle(
                    OHLCV(
                        symbol=event.symbol,
                        open=event.close,
                        high=event.close,
                        low=event.close,
                        close=event.close,
                        volume=0,
                        timestamp=datetime.utcfromtimestamp(event.timestamp),
                    )
                )
        if available_at < start:
            # Warmup populates indicators and normalization only; never positions.
            hybrid.prices.update(updates)
            if hybrid.ema:
                for symbol, (timestamp, price) in updates.items():
                    hybrid.ema.observe(symbol, timestamp, price, available_at)
            audit.extend(hybrid.events)
            hybrid.events.clear()
            continue
        await executor.observe()
        await tick(available_at, updates)
        if next_timer == available_at:
            next_timer += timer
    while next_timer < end:
        await tick(next_timer)
        next_timer += timer
    clock[0] = end
    snap = tracker.snapshot()
    peak, max_drawdown = capital, 0.0
    exposure_time = 0.0
    for i, point in enumerate(curve):
        peak = max(peak, point["nav"])
        max_drawdown = max(max_drawdown, 1 - point["nav"] / peak)
        until = curve[i + 1]["timestamp"] if i + 1 < len(curve) else end
        exposure_time += point["exposure"] * (until - point["timestamp"])
    cumulative, fees, fill_days, turnover = {}, {}, set(), 0.0
    for event in audit:
        if event["event"] != "order_event":
            continue
        receipt = event["order"]
        old_qty, old_fee = cumulative.get(receipt["order_id"], (0, 0))
        qty, fee = receipt["filled_quantity"], receipt["commission"] or 0
        if qty > old_qty:
            when = (
                datetime.fromisoformat(receipt["filled_at"])
                if receipt["filled_at"]
                else datetime.utcfromtimestamp(event["timestamp"])
            )
            epoch = int(
                when.replace(tzinfo=timezone.utc).timestamp()
                if when.tzinfo is None
                else when.timestamp()
            )
            fill_days.add((epoch + 8 * 3600) // 86400)
            turnover += (qty - old_qty) * receipt["filled_price"]
        if fee > old_fee:
            key = f"{receipt['liquidity'] or 'UNKNOWN'}:{receipt['commission_asset']}"
            fees[key] = fees.get(key, 0) + fee - old_fee
        cumulative[receipt["order_id"]] = (qty, fee)
    holds = [e["holding_seconds"] for e in audit if e["event"] == "position_closed"]
    unreconciled = [
        receipt.model_dump(mode="json")
        for order_id, receipt in executor.receipts.items()
        if receipt.filled_quantity > cumulative.get(order_id, (0, 0))[0]
    ]
    summary = dict(
        evidence="engineering_replay_not_venue_calibration",
        start_at=start,
        end_at=end,
        window_days=(end - start) / 86400,
        initial_capital=capital,
        initial_positions=[],
        entry_cutoff_seconds=entry_cutoff,
        closeout_seconds=closeout,
        final_nav=snap.nav,
        net_return=snap.nav / capital - 1,
        max_drawdown=max_drawdown,
        time_weighted_exposure=exposure_time / (end - start),
        turnover_notional=turnover,
        fees_by_role_and_asset=fees,
        active_fill_days_utc8=len(fill_days),
        meets_eight_active_days=len(fill_days) >= 8,
        completed_holds=len(holds),
        median_hold_seconds=median(holds) if holds else None,
        fraction_completed_holds_3_to_4h=sum(10800 <= h <= 14400 for h in holds)
        / len(holds)
        if holds
        else None,
        fraction_completed_holds_1_to_2h=sum(3600 <= h <= 7200 for h in holds) / len(holds) if holds else None,
        open_hold_age_seconds={
            s: end - h.opened_at for s, h in hybrid.holdings.items()
        },
        positions=[p.model_dump(mode="json") for p in snap.positions if p.quantity > 0],
        pending_orders=[
            o.model_dump(mode="json") for o in manager.active_orders.values()
        ],
        venue_fills_not_yet_reconciled=unreconciled,
        accounting_provisional=bool(unreconciled),
        requests=executor.budget.total,
        daily_ic="not_computed_without_matured_labels_use_research_daily_ic_evaluator",
    )
    return dict(summary=summary, config=config, equity=curve, events=audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    source = config["replay"].get("events_path")
    if not source:
        parser.error("replay.events_path is required; no synthetic-data fallback")
    result = asyncio.run(run_replay(config, read_events(source)))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Avoid overwriting existing experiments.
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
