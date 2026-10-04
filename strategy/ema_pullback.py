"""Causal completed-five-minute EMA context for directional pullback research."""
from collections import deque
from dataclasses import dataclass, field
import math


@dataclass
class EmaState:
    closed_at: int
    fast: float
    slow: float
    residual_mean: float = 0.
    residual_variance: float = 0.
    count: int = 1
    closes: deque = field(default_factory=lambda: deque(maxlen=2))
    fast_history: deque = field(default_factory=lambda: deque(maxlen=13))


class EmaPullbackHistory:
    """Consume the last observed second before each completed 5-minute boundary.

    A missing boundary restarts warmup rather than silently joining separated data.
    The replay's second-open proxy is not an exact exchange candle close.
    """
    def __init__(self, config):
        self.fast_hours = float(config.get('fast_half_life_hours', 6))
        self.slow_hours = float(config.get('slow_half_life_hours', 24))
        self.vol_hours = float(config.get('vol_half_life_hours', 24))
        self.warmup = int(config.get('warmup_bars', 2016))
        self.trend_mode = config.get('trend_mode', 'strict')
        self.downtrend_size = float(config.get('downtrend_size_multiplier', .5))
        if self.trend_mode not in {'strict', 'soft'} or not 0 < self.downtrend_size <= 1:
            raise ValueError('Invalid EMA trend mode or downside size')
        if not (0 < self.fast_hours < self.slow_hours and self.vol_hours > 0 and self.warmup >= 13):
            raise ValueError('Invalid EMA horizons or warmup')
        self.states = {}
        self.resets = {}

    def observe(self, symbol, timestamp, price, now):
        if timestamp > now or not math.isfinite(price) or price <= 0:
            raise ValueError('Invalid causal EMA price')
        boundary = timestamp + 1
        if boundary % 300 or boundary > now:
            return
        state = self.states.get(symbol)
        if state and boundary <= state.closed_at:
            if boundary == state.closed_at and price == state.closes[-1]:
                return
            raise ValueError('Conflicting or out-of-order EMA observation')
        if state is None or boundary - state.closed_at != 300:
            if state:
                self.resets[symbol] = self.resets.get(symbol, 0) + 1
            state = EmaState(boundary, price, price)
            state.closes.append(price)
            state.fast_history.append(price)
            self.states[symbol] = state
            return
        # Residual is measured against the previous middle, before updating it.
        residual = price - state.fast
        a = 1 - 2 ** (-300 / (self.vol_hours * 3600))
        delta = residual - state.residual_mean
        state.residual_mean += a * delta
        state.residual_variance = (1 - a) * (state.residual_variance + a * delta * delta)
        state.fast += (1 - 2 ** (-300 / (self.fast_hours * 3600))) * (price - state.fast)
        state.slow += (1 - 2 ** (-300 / (self.slow_hours * 3600))) * (price - state.slow)
        state.closed_at, state.count = boundary, state.count + 1
        state.closes.append(price)
        state.fast_history.append(state.fast)

    def bands(self, symbol, now):
        state = self.states.get(symbol)
        if state is None or state.count < self.warmup or not 0 <= now - state.closed_at < 300:
            return None
        std = math.sqrt(max(0., state.residual_variance))
        return (state.fast, std) if std > 1e-12 else None

    def entry_check(self, symbol, price, now, min_recovery_bps):
        if self.bands(symbol, now) is None:
            return 'ema_unready'
        state = self.states[symbol]
        if self.trend_mode == 'strict' and not (
            state.fast > state.slow and state.fast > state.fast_history[0]
        ):
            return 'trend_not_up'
        if state.closes[-1] <= state.closes[-2]:
            return 'no_completed_bar_recovery'
        if (state.fast / price - 1) * 10000 < min_recovery_bps:
            return 'insufficient_recovery_distance'
        return 'qualified'

    def size_multiplier(self, symbol, now):
        """Soft trend changes risk, not forecast eligibility or the MR trigger."""
        if self.bands(symbol, now) is None:
            return 0.
        state = self.states[symbol]
        return (self.downtrend_size if self.trend_mode == 'soft'
                and state.fast <= state.slow else 1.)
