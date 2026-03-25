from __future__ import annotations

import math
from collections import defaultdict, deque


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0
    return cov / denom


class BayesianFactorICIRManager:
    """Bayesian-shrunk ICIR-style online factor reweighting for selected factors."""

    def __init__(self, config: dict):
        strategy_cfg = config.get("strategy", {})
        self._enabled = bool(strategy_cfg.get("factor_icir_enabled", False))
        self._targets = tuple(strategy_cfg.get("factor_icir_targets", []))
        self._window = max(20, int(strategy_cfg.get("factor_icir_window", 120)))
        self._min_samples = max(
            10, int(strategy_cfg.get("factor_icir_min_samples", 30))
        )
        self._min_lambda = float(strategy_cfg.get("factor_icir_min_lambda", 0.35))
        self._tau = float(strategy_cfg.get("factor_icir_tau", 60.0))
        self._floor = float(strategy_cfg.get("factor_icir_weight_floor", 0.70))
        self._ceiling = float(strategy_cfg.get("factor_icir_weight_ceiling", 1.35))
        base_weights = strategy_cfg.get("factor_weights", {})
        target_weights = {
            name: float(base_weights.get(name, 0.0)) for name in self._targets
        }
        total = sum(abs(value) for value in target_weights.values())
        if total > 1e-12:
            self._prior_shares = {
                name: abs(value) / total for name, value in target_weights.items()
            }
        else:
            uniform = 1.0 / max(len(self._targets), 1)
            self._prior_shares = {name: uniform for name in self._targets}
        self._factor_history = defaultdict(
            lambda: {name: deque(maxlen=self._window) for name in self._targets}
        )
        self._return_history = defaultdict(lambda: deque(maxlen=self._window))
        self._sample_count = defaultdict(int)

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._targets)

    @property
    def targets(self) -> tuple[str, ...]:
        return self._targets

    def record(
        self, symbol: str, factor_scores: dict[str, float], forward_return: float
    ) -> None:
        if not self.enabled:
            return
        for name in self._targets:
            self._factor_history[symbol][name].append(float(factor_scores.get(name, 0.0)))
        self._return_history[symbol].append(float(forward_return))
        self._sample_count[symbol] += 1

    def multiplier(self, symbol: str, factor_name: str) -> float:
        if not self.enabled or factor_name not in self._targets:
            return 1.0
        if self._sample_count[symbol] < self._min_samples:
            return 1.0
        returns = list(self._return_history[symbol])
        if len(returns) < self._min_samples:
            return 1.0

        online_scores = {}
        for name in self._targets:
            online_scores[name] = abs(_pearson(list(self._factor_history[symbol][name]), returns))
        total = sum(online_scores.values())
        if total < 1e-12:
            return 1.0

        online_share = online_scores[factor_name] / total
        prior_share = self._prior_shares.get(
            factor_name, 1.0 / max(len(self._targets), 1)
        )
        lam = self._min_lambda + (1.0 - self._min_lambda) * math.exp(
            -self._sample_count[symbol] / max(self._tau, 1e-6)
        )
        shrunk_share = lam * prior_share + (1.0 - lam) * online_share
        raw_multiplier = shrunk_share / max(prior_share, 1e-6)
        return max(self._floor, min(self._ceiling, raw_multiplier))
