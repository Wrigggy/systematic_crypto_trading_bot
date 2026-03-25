from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class _PosteriorState:
    observations: deque[float] = field(default_factory=deque)
    posterior_log_mean: float = 0.0
    posterior_log_var: float = 1.0
    latest_observed_vol: float = 0.0


class BayesianVolatilityManager:
    """Track per-symbol volatility posteriors and derive dynamic risk controls."""

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        trend_cfg = config.get("trend", {})

        self._enabled: bool = bool(
            risk_cfg.get("bayesian_volatility_enabled", False)
        )
        prior_vol = max(
            float(
                risk_cfg.get(
                    "bayesian_volatility_prior_mean",
                    trend_cfg.get("vol_target_floor", 0.01),
                )
            ),
            1e-6,
        )
        self._prior_log_mean: float = math.log(prior_vol)
        self._prior_strength: float = max(
            1e-6, float(risk_cfg.get("bayesian_volatility_prior_strength", 20.0))
        )
        self._observation_strength: float = max(
            1e-6,
            float(risk_cfg.get("bayesian_volatility_observation_strength", 4.0)),
        )
        self._window: int = max(
            5, int(risk_cfg.get("bayesian_volatility_window", 96))
        )
        self._uncertainty_penalty: float = max(
            0.0,
            float(risk_cfg.get("bayesian_volatility_uncertainty_penalty", 1.5)),
        )
        self._target_volatility: float = max(
            1e-6,
            float(
                risk_cfg.get(
                    "bayesian_volatility_target",
                    trend_cfg.get("vol_target_floor", 0.01),
                )
            ),
        )

        self._min_size_multiplier: float = max(
            0.01,
            float(risk_cfg.get("bayesian_volatility_min_size_multiplier", 0.35)),
        )
        self._max_size_multiplier: float = max(
            self._min_size_multiplier,
            float(risk_cfg.get("bayesian_volatility_max_size_multiplier", 1.20)),
        )
        self._min_single_exposure_multiplier: float = max(
            0.01,
            float(
                risk_cfg.get(
                    "bayesian_volatility_min_single_exposure_multiplier", 0.45
                )
            ),
        )
        self._max_single_exposure_multiplier: float = max(
            self._min_single_exposure_multiplier,
            float(
                risk_cfg.get(
                    "bayesian_volatility_max_single_exposure_multiplier", 1.10
                )
            ),
        )
        self._min_portfolio_exposure_multiplier: float = max(
            0.01,
            float(
                risk_cfg.get(
                    "bayesian_volatility_min_portfolio_exposure_multiplier", 0.55
                )
            ),
        )
        self._max_portfolio_exposure_multiplier: float = max(
            self._min_portfolio_exposure_multiplier,
            float(
                risk_cfg.get(
                    "bayesian_volatility_max_portfolio_exposure_multiplier", 1.10
                )
            ),
        )
        self._min_stop_multiplier: float = max(
            0.10,
            float(risk_cfg.get("bayesian_volatility_min_stop_multiplier", 0.90)),
        )
        self._max_stop_multiplier: float = max(
            self._min_stop_multiplier,
            float(risk_cfg.get("bayesian_volatility_max_stop_multiplier", 1.35)),
        )

        self._states: dict[str, _PosteriorState] = defaultdict(self._new_state)

    def _new_state(self) -> _PosteriorState:
        return _PosteriorState(
            observations=deque(maxlen=self._window),
            posterior_log_mean=self._prior_log_mean,
            posterior_log_var=1.0 / self._prior_strength,
            latest_observed_vol=math.exp(self._prior_log_mean),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, symbol: str, realized_volatility: float) -> None:
        if not self._enabled or realized_volatility <= 0:
            return

        state = self._states[symbol]
        state.latest_observed_vol = realized_volatility
        state.observations.append(math.log(max(realized_volatility, 1e-8)))

        n_obs = len(state.observations)
        obs_sum = sum(state.observations)
        posterior_precision = self._prior_strength + (
            n_obs * self._observation_strength
        )
        posterior_mean = (
            (self._prior_strength * self._prior_log_mean)
            + (self._observation_strength * obs_sum)
        ) / posterior_precision

        state.posterior_log_mean = posterior_mean
        state.posterior_log_var = 1.0 / posterior_precision

    def profile(
        self,
        symbol: str,
        realized_volatility: float | None = None,
    ) -> dict[str, float]:
        if not self._enabled:
            fallback_vol = max(realized_volatility or self._target_volatility, 1e-8)
            return {
                "posterior_volatility": fallback_vol,
                "posterior_uncertainty": 0.0,
                "risk_adjusted_volatility": fallback_vol,
                "size_multiplier": 1.0,
                "single_exposure_multiplier": 1.0,
                "portfolio_exposure_multiplier": 1.0,
                "stop_multiplier": 1.0,
            }

        state = self._states[symbol]
        posterior_log_mean = state.posterior_log_mean
        posterior_log_var = state.posterior_log_var
        posterior_volatility = math.exp(
            posterior_log_mean + 0.5 * posterior_log_var
        )
        posterior_uncertainty = math.sqrt(max(posterior_log_var, 0.0))
        risk_adjusted_volatility = posterior_volatility * (
            1.0 + self._uncertainty_penalty * posterior_uncertainty
        )

        raw_ratio = self._target_volatility / max(risk_adjusted_volatility, 1e-8)
        size_multiplier = _clamp(
            raw_ratio,
            self._min_size_multiplier,
            self._max_size_multiplier,
        )
        single_exposure_multiplier = _clamp(
            raw_ratio ** 0.85,
            self._min_single_exposure_multiplier,
            self._max_single_exposure_multiplier,
        )
        portfolio_exposure_multiplier = _clamp(
            raw_ratio ** 0.65,
            self._min_portfolio_exposure_multiplier,
            self._max_portfolio_exposure_multiplier,
        )
        stop_multiplier = _clamp(
            math.sqrt(max(risk_adjusted_volatility, 1e-8) / self._target_volatility),
            self._min_stop_multiplier,
            self._max_stop_multiplier,
        )

        if realized_volatility is not None and realized_volatility > 0:
            posterior_volatility = 0.7 * posterior_volatility + 0.3 * realized_volatility
            risk_adjusted_volatility = posterior_volatility * (
                1.0 + self._uncertainty_penalty * posterior_uncertainty
            )

        return {
            "posterior_volatility": posterior_volatility,
            "posterior_uncertainty": posterior_uncertainty,
            "risk_adjusted_volatility": risk_adjusted_volatility,
            "size_multiplier": size_multiplier,
            "single_exposure_multiplier": single_exposure_multiplier,
            "portfolio_exposure_multiplier": portfolio_exposure_multiplier,
            "stop_multiplier": stop_multiplier,
        }
