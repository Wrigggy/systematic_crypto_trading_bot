from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class _EdgePosterior:
    returns: deque[float] = field(default_factory=deque)
    posterior_mean: float = 0.0
    posterior_variance: float = 1.0
    posterior_score: float = 0.0


class BayesianSymbolPerformanceManager:
    """Rolling Bayesian shrinkage for per-symbol trade performance."""

    def __init__(self, config: dict):
        strategy_cfg = config.get("strategy", {})
        self._enabled: bool = bool(
            strategy_cfg.get("bayesian_symbol_performance_enabled", True)
        )
        self._window: int = max(
            3, int(strategy_cfg.get("symbol_performance_window", 8))
        )
        self._prior_mean: float = float(
            strategy_cfg.get("bayesian_symbol_prior_mean", 0.0)
        )
        self._prior_strength: float = max(
            1e-6, float(strategy_cfg.get("bayesian_symbol_prior_strength", 6.0))
        )
        self._observation_strength: float = max(
            1e-6,
            float(strategy_cfg.get("bayesian_symbol_observation_strength", 1.5)),
        )
        self._uncertainty_penalty: float = max(
            0.0,
            float(strategy_cfg.get("bayesian_symbol_uncertainty_penalty", 1.25)),
        )
        self._min_trades: int = max(
            1, int(strategy_cfg.get("symbol_performance_min_trades", 3))
        )
        self._states: dict[str, _EdgePosterior] = defaultdict(self._new_state)

    def _new_state(self) -> _EdgePosterior:
        return _EdgePosterior(
            returns=deque(maxlen=self._window),
            posterior_mean=self._prior_mean,
            posterior_variance=1.0 / self._prior_strength,
            posterior_score=self._prior_mean,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_trade(self, symbol: str, trade_return: float) -> None:
        if not self._enabled:
            return

        state = self._states[symbol]
        state.returns.append(float(trade_return))

        n_obs = len(state.returns)
        sample_mean = sum(state.returns) / max(n_obs, 1)
        posterior_precision = self._prior_strength + (
            n_obs * self._observation_strength
        )
        posterior_mean = (
            self._prior_strength * self._prior_mean
            + (n_obs * self._observation_strength * sample_mean)
        ) / posterior_precision
        posterior_variance = 1.0 / posterior_precision
        posterior_score = posterior_mean - (
            self._uncertainty_penalty * (posterior_variance ** 0.5)
        )

        state.posterior_mean = posterior_mean
        state.posterior_variance = posterior_variance
        state.posterior_score = posterior_score

    def score(self, symbol: str) -> float:
        if not self._enabled:
            return 0.0
        state = self._states[symbol]
        if len(state.returns) < self._min_trades:
            return 0.0
        return float(state.posterior_score)

    def profile(self, symbol: str) -> dict[str, float]:
        state = self._states[symbol]
        return {
            "posterior_mean": float(state.posterior_mean),
            "posterior_variance": float(state.posterior_variance),
            "posterior_score": float(
                self.score(symbol) if self._enabled else self._prior_mean
            ),
            "sample_count": float(len(state.returns)),
        }

