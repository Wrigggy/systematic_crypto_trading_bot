"""Long-only price mean reversion qualified by causal multi-horizon forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev

from core.models import Order, OrderStatus, OrderType, Side
from execution.order_manager import OrderManager
from plugins.model_inference.forecasts import ForecastBook, ForecastPacket
from risk.tracker import PortfolioTracker
from strategy.fusion import SignalFusion
from strategy.ema_pullback import EmaPullbackHistory


@dataclass
class Holding:
    opened_at: int
    target: float
    entry_price: float
    peak: float
    weak_since: int | None = None
    exit_since: int | None = None
    exit_reason: str = ""


class HourlyHistory:
    """Completed hourly closes indexed by their closing boundary, never opening time."""

    def __init__(self, lookback_days=20, sample_hours=8):
        if lookback_days < 1 or sample_hours < 1 or (lookback_days * 24) % sample_hours:
            raise ValueError("Invalid Bollinger sampling specification")
        self.hours = lookback_days * 24
        self.sample_hours = sample_hours
        self.values = {}
        self._bands_cache = {}

    def add(self, symbol: str, closed_at: int, close: float, now: int):
        if (
            closed_at % 3600
            or closed_at > now
            or not math.isfinite(close)
            or close <= 0
        ):
            raise ValueError("Invalid completed hourly close")
        history = self.values.setdefault(symbol, {})
        if closed_at in history and history[closed_at] != close:
            raise ValueError("Conflicting completed-hour history")
        if history and closed_at < max(history):
            raise ValueError("Hourly observations must not arrive out of order")
        history[closed_at] = close
        self._bands_cache.pop(symbol, None)
        for old in [t for t in history if t < closed_at - (self.hours + 1) * 3600]:
            del history[old]

    def bands(self, symbol: str, now: int):
        end = now // 3600 * 3600
        history = self.values.get(symbol, {})
        cached = self._bands_cache.get(symbol)
        cache_key = (end, len(history))
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        if any(end - k * 3600 not in history for k in range(self.hours)):
            self._bands_cache[symbol] = (cache_key, None)
            return None
        values = [
            history[end - k * 3600] for k in range(0, self.hours, self.sample_hours)
        ]
        std = pstdev(values)
        result = (mean(values), std) if std > 1e-12 else None
        self._bands_cache[symbol] = (cache_key, result)
        return result


class HybridCoordinator:
    """Shared online/replay decision implementation; execution owns actual fills."""

    def __init__(
        self,
        config: dict,
        forecasts: ForecastBook,
        orders: OrderManager,
        tracker: PortfolioTracker,
        now,
    ):
        self.config = config
        self.cadence = config.get("decision_interval_seconds")
        if (
            not isinstance(self.cadence, int)
            or isinstance(self.cadence, bool)
            or self.cadence <= 0
        ):
            raise ValueError(
                "decision_interval_seconds is undecided; provide it explicitly"
            )
        self.rules = config["rules"]
        if not self.rules:
            raise ValueError("No mean-reversion universe configured")
        self.book, self.orders, self.tracker, self.now = forecasts, orders, tracker, now
        if any(p.quantity > 0 for p in tracker.snapshot().positions):
            raise ValueError(
                "Hybrid requires cash start; existing inventory needs explicit state recovery"
            )
        self.fusion = SignalFusion(config.get("fusion", {}))
        for rule in self.rules.values():
            if (
                not math.isfinite(rule["entry_sigma"])
                or rule["entry_sigma"] <= 0
                or not math.isfinite(rule["exit_z"])
            ):
                raise ValueError("Invalid mean-reversion rule")
        self.history = HourlyHistory(
            config.get("lookback_days", 20), config.get("sample_hours", 8)
        )
        self.ema = EmaPullbackHistory(config['ema_pullback']) if config.get('ema_pullback', {}).get('enabled') else None
        self.holdings = {}
        self.prices = {}
        self.last_decision = {}
        self.blocked_episode = set()
        self.pending_targets = {}
        self.urgent = {}
        self.events = []
        self._last_event_time = -1
        self._day = None
        self._day_peak = tracker.snapshot().nav
        self.halted = False
        self.review = int(config.get("review_after_seconds", 10800))
        self.maximum = int(config.get("max_holding_seconds", 14400))
        self.confirm = int(config.get("weak_support_seconds", 300))
        self.exit_wait = int(config.get("exit_timeout_seconds", 120))
        self.stop_pct = float(config.get("stop_loss_pct", 0.03))
        self.trail_pct = float(config.get("trailing_stop_pct", 0.03))
        self.max_price_age = int(config.get("max_price_age_seconds", 120))
        if (
            not 0 < self.review <= self.maximum
            or self.confirm <= 0
            or self.exit_wait <= 0
        ):
            raise ValueError("Invalid holding or exit timing")
        if not 0 < self.stop_pct < 1 or not 0 < self.trail_pct < 1:
            raise ValueError("Invalid price risk limits")
        for key, default in (
            ("position_weight", 0.1),
            ("max_single_exposure", 0.15),
            ("max_portfolio_exposure", 0.5),
            ("daily_drawdown_limit", 0.05),
        ):
            if not 0 < float(config.get(key, default)) <= 1:
                raise ValueError(f"Invalid risk parameter: {key}")
        if self.max_price_age <= 0 or int(config.get("max_positions", 3)) <= 0:
            raise ValueError("Invalid capacity or price freshness")
        self.orders.register_fill_callback(self._on_order)

    def _record(self, event, symbol="", **fields):
        self.events.append(
            {"timestamp": int(self.now()), "event": event, "symbol": symbol, **fields}
        )

    def ingest_forecast(self, packet: ForecastPacket):
        try:
            self.book.ingest(packet, int(self.now()))
        except ValueError as exc:
            self.book.scores.pop(packet.symbol, None)
            self._record("forecast_rejected", packet.symbol, reason=str(exc))
            return False
        return True

    def _on_order(self, order):
        now = int(self.now())
        symbol = order.symbol
        quantity = self.tracker.get_position(symbol).quantity
        if quantity > 1e-10 and symbol not in self.holdings:
            if order.side != Side.BUY or not order.filled_price:
                raise RuntimeError("Inventory exists without an entry receipt")
            # A receipt may arrive after the first execution. Never time the
            # position from submission or from a later polling acknowledgement.
            filled_at = order.filled_at
            if filled_at is None:
                opened_at = now  # Receipt time fallback, recorded explicitly.
            else:
                opened_at = int(
                    filled_at.replace(tzinfo=timezone.utc).timestamp()
                    if filled_at.tzinfo is None
                    else filled_at.timestamp()
                )
                if opened_at > now:
                    self._record("execution_clock_skew", symbol, receipt_time=opened_at)
                    opened_at = now
            self.holdings[symbol] = Holding(
                opened_at,
                self.pending_targets[symbol],
                self.tracker.get_position(symbol).entry_price,
                order.filled_price,
            )
            self._record(
                "position_opened",
                symbol,
                entry_price=order.filled_price,
                opened_at=opened_at,
                timing_source=(order.fill_time_source or "receipt")
                if filled_at
                else "observation",
            )
        if quantity > 1e-10 and symbol in self.holdings and order.side == Side.BUY:
            self.holdings[symbol].entry_price = self.tracker.get_position(
                symbol
            ).entry_price
        if quantity <= 1e-10 and symbol in self.holdings:
            holding = self.holdings.pop(symbol)
            closed_at = now
            if order.filled_at is not None:
                filled = order.filled_at
                closed_at = int(filled.replace(tzinfo=timezone.utc).timestamp()
                                if filled.tzinfo is None else filled.timestamp())
                closed_at = max(holding.opened_at, min(now, closed_at))
            self._record(
                "position_closed",
                symbol,
                holding_seconds=closed_at - holding.opened_at,
                closed_at=closed_at,
                receipt_delay_seconds=now - closed_at,
                reason=holding.exit_reason or order.reason,
            )
            self.blocked_episode.add(symbol)
            self.urgent.pop(symbol, None)
        if (
            order.status in {OrderStatus.CANCELLED, OrderStatus.REJECTED}
            and order.side == Side.BUY
            and order.reason != "request_budget_deferred"
        ):
            self.blocked_episode.add(symbol)
        self._record("order_event", symbol, order=order.model_dump(mode="json"))

    async def step(
        self, prices: dict[str, tuple[int, float]] | None = None, allow_entries=True
    ):
        """Observe causal completed prices, service risk, then make scheduled decisions."""
        now = int(self.now())
        if now < self._last_event_time:
            raise ValueError("Hybrid time cannot move backwards")
        self._last_event_time = now
        for symbol, (timestamp, price) in (prices or {}).items():
            if timestamp > now or not math.isfinite(price) or price <= 0:
                raise ValueError("Future or invalid observed price")
            if symbol in self.prices and timestamp < self.prices[symbol][0]:
                raise ValueError("Out-of-order price")
            self.prices[symbol] = (timestamp, price)
            self.tracker.update_prices(symbol, price)
            if self.ema:
                self.ema.observe(symbol, timestamp, price, now)
        # First service already-triggered exits. Routine polling cannot consume all
        # risk capacity before a stop or deadline is evaluated.
        await self._risk(now)
        await self._service_urgent()
        await self.orders.check_pending()
        # A first partial fill discovered above starts risk timing immediately.
        await self._risk(now)
        await self._service_urgent()
        candidates = []
        for symbol in self.rules:
            signal = self.book.signal(symbol, now)
            pending = self.orders.for_symbol(symbol)
            observed = self.prices.get(symbol)
            fresh = observed is not None and now - observed[0] <= self.max_price_age
            bands = self.ema.bands(symbol, now) if self.ema else self.history.bands(symbol, now)
            rule = self.rules[symbol]
            fused = (
                self.fusion.evaluate(observed[1], *bands, rule["entry_sigma"], signal)
                if fresh and bands
                else None
            )
            ema_reason = (self.ema.entry_check(symbol, observed[1], now,
                float(self.config['ema_pullback'].get('min_recovery_bps', 20)))
                if self.ema and fresh else 'qualified')
            if any(o.side == Side.BUY for o in pending) and (
                fused is None or not fused.eligible or ema_reason != 'qualified'
            ):
                await self.orders.cancel_symbol(symbol)
            if now - self.last_decision.get(symbol, -(10**12)) < self.cadence:
                continue
            self.last_decision[symbol] = now
            if not fresh:
                continue
            price = observed[1]
            # Frozen-target exits do not require a still-complete entry history.
            if symbol in self.holdings:
                await self._holding_decision(symbol, price, signal, now)
                continue
            if bands is None:
                continue
            middle, std = bands
            lower = middle - float(rule["entry_sigma"]) * std
            if (
                price >= lower
                and symbol not in self.holdings
                and not self.orders.for_symbol(symbol)
            ):
                self.blocked_episode.discard(symbol)
            if allow_entries and price < lower and symbol not in self.blocked_episode:
                self._record(
                    "price_candidate",
                    symbol,
                    price=price,
                    model_score=None if signal is None else signal["composite"],
                    fusion=fused.as_dict(),
                )
                if fused.eligible:
                    if self.ema:
                        if ema_reason != 'qualified':
                            self._record('ema_entry_rejected', symbol, reason=ema_reason)
                            continue
                    candidates.append(
                        (
                            fused.joint_strength,
                            symbol,
                            price,
                            middle + float(rule["exit_z"]) * std,
                        )
                    )
        # Competing assets share cash. Rank jointly instead of giving dictionary
        # insertion order priority over the old and learned signal strengths.
        for _, symbol, price, target in sorted(candidates, key=lambda x: (-x[0], x[1])):
            await self._enter(symbol, price, target)

    async def _enter(self, symbol, price, target):
        if self.halted or self.urgent or self.orders.has_pending or target <= price:
            return
        if len(self.holdings) >= int(self.config.get("max_positions", 3)):
            return
        snap = self.tracker.snapshot()
        size_multiplier = self.ema.size_multiplier(symbol, int(self.now())) if self.ema else 1.
        allocation = min(
            snap.nav * float(self.config.get("position_weight", 0.1)) * size_multiplier,
            snap.nav * float(self.config.get("max_single_exposure", 0.15)),
            snap.nav
            * max(
                0.0,
                float(self.config.get("max_portfolio_exposure", 0.5))
                - self.tracker.get_total_exposure(),
            ),
            snap.cash / 1.002,
        )
        if allocation <= 0:
            return
        maker = self.config.get("maker_preferred", True)
        self.pending_targets[symbol] = target
        self._record("entry_intent", symbol, target=target, size_multiplier=size_multiplier)
        await self.orders.submit(
            Order(
                symbol=symbol,
                side=Side.BUY,
                order_type=OrderType.LIMIT if maker else OrderType.MARKET,
                quantity=allocation / price,
                price=price,
                maker_preferred=maker,
                created_at=datetime.utcfromtimestamp(self.now()),
                reason="hybrid_entry",
            )
        )

    async def _holding_decision(self, symbol, price, signal, now):
        holding = self.holdings[symbol]
        if price >= holding.target:
            await self._exit(symbol, "price_recovery", now)
        elif (
            self.fusion.mode != "price_only"
            and self.config.get("model_exit_enabled", True)
            and now - holding.opened_at >= self.review
        ):
            weak = signal is not None and signal["long_support"] <= self.config.get(
                "review_support_threshold", 0.0
            )
            if weak:
                holding.weak_since = (
                    holding.weak_since if holding.weak_since is not None else now
                )
                if now - holding.weak_since >= self.confirm:
                    await self._exit(symbol, "weak_long_horizon", now)
            else:
                holding.weak_since = None

    async def _exit(self, symbol, reason, now):
        holding = self.holdings[symbol]
        if holding.exit_since is None:
            holding.exit_since, holding.exit_reason = now, reason
        pending = self.orders.for_symbol(symbol)
        if pending:
            if any(o.side == Side.BUY for o in pending):
                await self.orders.cancel_symbol(symbol)
            return
        maker = self.config.get("maker_preferred", True)
        price = (
            max(self.prices[symbol][1], holding.target)
            if reason == "price_recovery"
            else self.prices[symbol][1]
        )
        await self.orders.submit(
            Order(
                symbol=symbol,
                side=Side.SELL,
                order_type=OrderType.LIMIT if maker else OrderType.MARKET,
                quantity=self.tracker.get_position(symbol).quantity,
                price=price,
                maker_preferred=maker,
                reason=reason,
                created_at=datetime.utcfromtimestamp(now),
            )
        )

    async def _risk(self, now):
        day = (now + 8 * 3600) // 86400
        nav = self.tracker.snapshot().nav
        if day != self._day:
            self._day, self._day_peak, self.halted = day, nav, False
        self._day_peak = max(self._day_peak, nav)
        if self._day_peak > 0 and 1 - nav / self._day_peak >= self.config.get(
            "daily_drawdown_limit", 0.05
        ):
            self.halted = True
        if self.halted:
            for order in self.orders.active_orders.values():
                self.urgent[order.symbol] = "circuit_breaker"
        for symbol, holding in list(self.holdings.items()):
            observed = self.prices.get(symbol)
            if self.halted:
                self.urgent[symbol] = "circuit_breaker"
            elif now - holding.opened_at >= self.maximum:
                self.urgent[symbol] = "holding_deadline"
            elif (
                holding.exit_since is not None
                and now - holding.exit_since >= self.exit_wait
            ):
                self.urgent[symbol] = "exit_timeout"
            elif observed and now - observed[0] <= self.max_price_age:
                price = observed[1]
                holding.peak = max(holding.peak, price)
                if price <= holding.entry_price * (
                    1 - self.stop_pct
                ) or price <= holding.peak * (1 - self.trail_pct):
                    self.urgent[symbol] = "price_stop"

    async def _service_urgent(self):
        for symbol, reason in list(self.urgent.items()):
            pending = self.orders.for_symbol(symbol)
            if any(o.urgent for o in pending):
                continue
            if pending and not await self.orders.cancel_symbol(symbol):
                continue
            quantity = self.tracker.get_position(symbol).quantity
            if quantity <= 1e-10:
                self.urgent.pop(symbol, None)
                continue
            if symbol in self.holdings:
                self.holdings[symbol].exit_reason = reason
            await self.orders.submit(
                Order(
                    symbol=symbol,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    urgent=True,
                    reason=reason,
                    created_at=datetime.utcfromtimestamp(self.now()),
                )
            )
