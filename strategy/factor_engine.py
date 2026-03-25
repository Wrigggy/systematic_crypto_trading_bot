from __future__ import annotations
from typing import Dict, List, Optional

from core.models import (
    FactorBias,
    FactorObservation,
    FactorSnapshot,
    FeatureVector,
    OHLCV,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _signed_clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class FactorEngine:
    """Translate raw features and market context into explicit strategy factors."""

    def __init__(self, config: dict):
        strategy_cfg = config.get("strategy", {})
        regime_cfg = config.get("regime", {})
        trend_cfg = config.get("trend", {})
        self._symbol_overrides: Dict[str, dict] = strategy_cfg.get(
            "symbol_overrides", {}
        )
        self._performance_manager = None
        self._factor_icir_manager = None
        self._dynamic_entry_factor_weighting_enabled: bool = bool(
            strategy_cfg.get("dynamic_entry_factor_weighting_enabled", False)
        )
        self._dynamic_entry_factor_weight_targets: set[str] = set(
            strategy_cfg.get(
                "dynamic_entry_factor_weight_targets",
                ["turtle_breakout_entry", "grid_reversion_entry"],
            )
        )
        self._dynamic_entry_factor_weight_floor: float = float(
            strategy_cfg.get("dynamic_entry_factor_weight_floor", 0.55)
        )
        self._dynamic_entry_factor_weight_ceiling: float = float(
            strategy_cfg.get("dynamic_entry_factor_weight_ceiling", 1.10)
        )
        self._dynamic_entry_factor_negative_scale: float = float(
            strategy_cfg.get("dynamic_entry_factor_negative_scale", 10.0)
        )
        self._dynamic_entry_factor_positive_scale: float = float(
            strategy_cfg.get("dynamic_entry_factor_positive_scale", 4.0)
        )
        self._factor_weights: Dict[str, float] = {
            "market_regime": 0.20,
            "trend_alignment": 0.28,
            "momentum_impulse": 0.22,
            "breakout_confirmation": 0.12,
            "turtle_breakout_entry": 0.0,
            "pullback_reentry": 0.0,
            "volatility_compression_entry": 0.0,
            "grid_reversion_entry": 0.0,
            "overextension_exit": 0.0,
            "onchain_mvrv_state": 0.0,
            "onchain_activity_state": 0.0,
            "onchain_nvt_proxy": 0.0,
            "onchain_ecosystem_state": 0.0,
            "volume_confirmation": 0.15,
            "liquidity_balance": 0.10,
            "perp_crowding": 0.15,
            "volatility_regime": 0.10,
            **strategy_cfg.get("factor_weights", {}),
        }
        self._min_volume_ratio: float = strategy_cfg.get("min_volume_ratio", 1.10)
        self._min_order_book_imbalance: float = strategy_cfg.get(
            "min_order_book_imbalance", 1.02
        )
        self._max_funding_rate: float = strategy_cfg.get("max_funding_rate", 0.0005)
        self._max_taker_ratio: float = strategy_cfg.get("max_taker_ratio", 1.20)
        self._max_open_interest_change: float = strategy_cfg.get(
            "max_open_interest_change", 0.03
        )
        self._open_interest_lookback_samples: int = max(
            2, int(strategy_cfg.get("open_interest_lookback_samples", 60))
        )
        self._max_volatility: float = strategy_cfg.get("max_volatility", 0.025)
        self._regime_enabled: bool = regime_cfg.get("enabled", False)
        self._regime_mode: str = str(regime_cfg.get("mode", "global")).lower()
        self._risk_on_threshold: float = regime_cfg.get("risk_on_threshold", 0.25)
        self._neutral_threshold: float = regime_cfg.get("neutral_threshold", 0.05)
        self._breakout_min_distance: float = trend_cfg.get(
            "breakout_min_distance", 0.10
        )
        self._breakout_overheat_distance: float = trend_cfg.get(
            "breakout_overheat_distance", 1.75
        )
        self._min_trend_slope: float = trend_cfg.get("min_trend_slope", 0.0005)
        self._min_volume_zscore: float = trend_cfg.get("min_volume_zscore", -0.25)
        self._enable_pullback_reentry: bool = strategy_cfg.get(
            "enable_pullback_reentry", False
        )
        self._enable_turtle_breakout_entry: bool = strategy_cfg.get(
            "enable_turtle_breakout_entry", False
        )
        self._turtle_min_breakout_distance: float = strategy_cfg.get(
            "turtle_min_breakout_distance", max(self._breakout_min_distance, 0.10)
        )
        self._turtle_max_breakout_distance: float = strategy_cfg.get(
            "turtle_max_breakout_distance", self._breakout_overheat_distance
        )
        self._turtle_min_volume_ratio: float = strategy_cfg.get(
            "turtle_min_volume_ratio", self._min_volume_ratio
        )
        self._turtle_trend_slack: float = strategy_cfg.get(
            "turtle_trend_slack", 0.90
        )
        self._turtle_min_rsi: float = strategy_cfg.get("turtle_min_rsi", 52.0)
        self._turtle_max_rsi: float = strategy_cfg.get("turtle_max_rsi", 74.0)
        self._pullback_min_breakout_distance: float = strategy_cfg.get(
            "pullback_min_breakout_distance", -0.65
        )
        self._pullback_max_breakout_distance: float = strategy_cfg.get(
            "pullback_max_breakout_distance", 0.20
        )
        self._pullback_target_breakout_distance: float = strategy_cfg.get(
            "pullback_target_breakout_distance", -0.10
        )
        self._pullback_min_rsi: float = strategy_cfg.get("pullback_min_rsi", 40.0)
        self._pullback_max_rsi: float = strategy_cfg.get("pullback_max_rsi", 60.0)
        self._pullback_target_rsi: float = strategy_cfg.get(
            "pullback_target_rsi", 52.0
        )
        self._pullback_min_momentum: float = strategy_cfg.get(
            "pullback_min_momentum", -0.02
        )
        self._pullback_max_momentum: float = strategy_cfg.get(
            "pullback_max_momentum", 0.01
        )
        self._pullback_min_volume_ratio: float = strategy_cfg.get(
            "pullback_min_volume_ratio", 0.75
        )
        self._pullback_trend_slack: float = strategy_cfg.get(
            "pullback_trend_slack", 0.70
        )
        self._enable_volatility_compression_entry: bool = strategy_cfg.get(
            "enable_volatility_compression_entry", False
        )
        self._vol_compression_min_breakout_distance: float = strategy_cfg.get(
            "vol_compression_min_breakout_distance", -0.05
        )
        self._vol_compression_max_breakout_distance: float = strategy_cfg.get(
            "vol_compression_max_breakout_distance", 0.25
        )
        self._vol_compression_min_rsi: float = strategy_cfg.get(
            "vol_compression_min_rsi", 48.0
        )
        self._vol_compression_max_rsi: float = strategy_cfg.get(
            "vol_compression_max_rsi", 64.0
        )
        self._vol_compression_short_vol_ratio_max: float = strategy_cfg.get(
            "vol_compression_short_vol_ratio_max", 0.90
        )
        self._vol_compression_min_volume_ratio: float = strategy_cfg.get(
            "vol_compression_min_volume_ratio", 0.75
        )
        self._vol_compression_max_volume_ratio: float = strategy_cfg.get(
            "vol_compression_max_volume_ratio", 1.20
        )
        self._vol_compression_trend_slack: float = strategy_cfg.get(
            "vol_compression_trend_slack", 0.85
        )
        self._enable_grid_reversion_entry: bool = strategy_cfg.get(
            "enable_grid_reversion_entry", False
        )
        self._grid_reversion_min_breakout_distance: float = strategy_cfg.get(
            "grid_reversion_min_breakout_distance", -0.25
        )
        self._grid_reversion_max_breakout_distance: float = strategy_cfg.get(
            "grid_reversion_max_breakout_distance", 0.04
        )
        self._grid_reversion_target_breakout_distance: float = strategy_cfg.get(
            "grid_reversion_target_breakout_distance", -0.08
        )
        self._grid_reversion_min_rsi: float = strategy_cfg.get(
            "grid_reversion_min_rsi", 45.0
        )
        self._grid_reversion_max_rsi: float = strategy_cfg.get(
            "grid_reversion_max_rsi", 56.0
        )
        self._grid_reversion_target_rsi: float = strategy_cfg.get(
            "grid_reversion_target_rsi", 50.0
        )
        self._grid_reversion_min_momentum: float = strategy_cfg.get(
            "grid_reversion_min_momentum", -0.006
        )
        self._grid_reversion_max_momentum: float = strategy_cfg.get(
            "grid_reversion_max_momentum", 0.004
        )
        self._grid_reversion_min_volume_ratio: float = strategy_cfg.get(
            "grid_reversion_min_volume_ratio", 0.80
        )
        self._grid_reversion_max_volume_ratio: float = strategy_cfg.get(
            "grid_reversion_max_volume_ratio", 1.35
        )
        self._grid_reversion_short_vol_ratio_max: float = strategy_cfg.get(
            "grid_reversion_short_vol_ratio_max", 1.10
        )
        self._grid_reversion_trend_slack: float = strategy_cfg.get(
            "grid_reversion_trend_slack", 0.80
        )
        self._enable_overextension_exit: bool = strategy_cfg.get(
            "enable_overextension_exit", False
        )
        self._overextension_min_breakout_distance: float = strategy_cfg.get(
            "overextension_min_breakout_distance", 0.50
        )
        self._overextension_min_rsi: float = strategy_cfg.get(
            "overextension_min_rsi", 68.0
        )
        self._overextension_min_volume_zscore: float = strategy_cfg.get(
            "overextension_min_volume_zscore", 0.0
        )
        self._overextension_min_taker_ratio: float = strategy_cfg.get(
            "overextension_min_taker_ratio", 1.02
        )
        self._onchain_stale_days_max: float = float(
            strategy_cfg.get("onchain_stale_days_max", 10.0)
        )
        self._onchain_mvrv_bullish_max: float = float(
            strategy_cfg.get("onchain_mvrv_bullish_max", 1.05)
        )
        self._onchain_mvrv_bearish_min: float = float(
            strategy_cfg.get("onchain_mvrv_bearish_min", 1.80)
        )
        self._onchain_activity_lookback: int = max(
            5, int(strategy_cfg.get("onchain_activity_lookback", 14))
        )
        self._onchain_activity_bullish_growth: float = float(
            strategy_cfg.get("onchain_activity_bullish_growth", 0.08)
        )
        self._onchain_activity_bearish_growth: float = float(
            strategy_cfg.get("onchain_activity_bearish_growth", -0.10)
        )
        self._onchain_nvt_proxy_lookback: int = max(
            10, int(strategy_cfg.get("onchain_nvt_proxy_lookback", 30))
        )
        self._onchain_nvt_proxy_bullish_zscore: float = float(
            strategy_cfg.get("onchain_nvt_proxy_bullish_zscore", -0.50)
        )
        self._onchain_nvt_proxy_bearish_zscore: float = float(
            strategy_cfg.get("onchain_nvt_proxy_bearish_zscore", 1.00)
        )
        self._onchain_ecosystem_lookback: int = max(
            7, int(strategy_cfg.get("onchain_ecosystem_lookback", 14))
        )
        self._onchain_ecosystem_bullish_growth: float = float(
            strategy_cfg.get("onchain_ecosystem_bullish_growth", 0.12)
        )
        self._onchain_ecosystem_bearish_growth: float = float(
            strategy_cfg.get("onchain_ecosystem_bearish_growth", -0.12)
        )

    def set_performance_manager(self, manager) -> None:
        self._performance_manager = manager

    def set_factor_icir_manager(self, manager) -> None:
        self._factor_icir_manager = manager

    def _symbol_param(self, symbol: str, key: str, default):
        overrides = self._symbol_overrides.get(symbol, {})
        return overrides.get(key, default)

    def _factor_weight(self, symbol: str, factor_name: str) -> float:
        weight = self._factor_weights.get(factor_name, 0.0)
        if weight <= 0:
            return weight

        if self._factor_icir_manager is not None:
            weight *= self._factor_icir_manager.multiplier(symbol, factor_name)

        if (
            not self._dynamic_entry_factor_weighting_enabled
            or factor_name not in self._dynamic_entry_factor_weight_targets
            or self._performance_manager is None
        ):
            return weight

        profile = self._performance_manager.profile(symbol)
        score = float(profile.get("posterior_score", 0.0))
        if score < 0:
            multiplier = 1.0 + score * self._dynamic_entry_factor_negative_scale
        else:
            multiplier = 1.0 + score * self._dynamic_entry_factor_positive_scale
        multiplier = _clamp(
            multiplier,
            self._dynamic_entry_factor_weight_floor,
            self._dynamic_entry_factor_weight_ceiling,
        )
        return weight * multiplier

    def evaluate(
        self,
        features: FeatureVector,
        supplementary: Optional[dict] = None,
        supplementary_history: Optional[dict] = None,
        candles_15m: Optional[List[OHLCV]] = None,
        candles_1h: Optional[List[OHLCV]] = None,
        market_context: Optional[dict] = None,
    ) -> FactorSnapshot:
        supp = supplementary or {}
        hist = supplementary_history or {}
        observations = [
            self._market_regime(features, market_context),
            self._trend_alignment(features, candles_15m, candles_1h),
            self._momentum_impulse(features),
            self._breakout_confirmation(features),
            self._turtle_breakout_entry(features),
            self._pullback_reentry(features),
            self._volatility_compression_entry(features),
            self._grid_reversion_entry(features),
            self._overextension_exit(features, supp),
            self._volume_confirmation(features),
            self._liquidity_balance(features, supp),
            self._perp_crowding(features, supp, hist),
            self._onchain_mvrv_state(features, supp),
            self._onchain_activity_state(features, supp, hist),
            self._onchain_nvt_proxy(features, supp, hist),
            self._onchain_ecosystem_state(features, supp, hist),
            self._volatility_regime(features),
        ]

        support_weight = 0.0
        support_score = 0.0
        blocker_weight = 0.0
        blocker_score = 0.0
        exit_weight = 0.0
        exit_score = 0.0

        supporting_factors = []
        blocking_factors = []
        for obs in observations:
            weight = self._factor_weight(features.symbol, obs.name)
            if obs.bias == FactorBias.BULLISH:
                support_weight += weight
                support_score += weight * obs.strength
                supporting_factors.append(obs.name)
            elif obs.bias == FactorBias.BEARISH:
                blocker_weight += weight
                blocker_score += weight * obs.strength
                exit_weight += weight
                exit_score += weight * obs.strength
                blocking_factors.append(obs.name)

        entry_score = support_score / support_weight if support_weight > 0 else 0.0
        blocker = blocker_score / blocker_weight if blocker_weight > 0 else 0.0
        exit_score = exit_score / exit_weight if exit_weight > 0 else 0.0
        confidence = _clamp(entry_score * (1.0 - 0.5 * blocker))

        if market_context is not None:
            regime = market_context.get("regime", "neutral")
        elif entry_score >= 0.65 and blocker < 0.35:
            regime = "risk_on"
        elif blocker >= 0.60:
            regime = "risk_off"
        else:
            regime = "neutral"

        summary_parts = [obs.thesis for obs in observations if obs.bias != FactorBias.NEUTRAL]
        summary = "; ".join(summary_parts[:3]) if summary_parts else "No dominant factor signal"

        return FactorSnapshot(
            symbol=features.symbol,
            timestamp=features.timestamp,
            regime=regime,
            entry_score=_clamp(entry_score),
            exit_score=_clamp(exit_score),
            blocker_score=_clamp(blocker),
            confidence=confidence,
            observations=observations,
            supporting_factors=supporting_factors,
            blocking_factors=blocking_factors,
            summary=summary,
        )

    @property
    def supplementary_history_window(self) -> int:
        """Minimum raw supplementary history required by the factor engine."""
        return max(
            self._open_interest_lookback_samples,
            self._onchain_activity_lookback,
            self._onchain_nvt_proxy_lookback,
            self._onchain_ecosystem_lookback,
        )

    def _market_regime(
        self, features: FeatureVector, market_context: Optional[dict]
    ) -> FactorObservation:
        if not self._regime_enabled:
            return FactorObservation(
                symbol=features.symbol,
                name="market_regime",
                category="market",
                timestamp=features.timestamp,
                bias=FactorBias.NEUTRAL,
                strength=0.0,
                value=0.0,
                threshold=self._neutral_threshold,
                horizon_minutes=240,
                expected_move_bps=0.0,
                thesis="Market regime filter disabled",
                invalidate_condition="Regime filter becomes active",
                metadata={},
            )

        if self._regime_mode == "per_symbol":
            risk_on_threshold = self._symbol_param(
                features.symbol, "risk_on_threshold", self._risk_on_threshold
            )
            neutral_threshold = self._symbol_param(
                features.symbol, "neutral_threshold", self._neutral_threshold
            )
            max_volatility = self._symbol_param(
                features.symbol, "max_volatility", self._max_volatility
            )
            ema_spread = (
                (features.ema_fast - features.ema_slow) / features.ema_slow
                if features.ema_slow > 0
                else 0.0
            )
            trend_slope = float(features.raw.get("trend_slope", 0.0))
            breakout_distance = float(features.raw.get("breakout_distance", 0.0))
            momentum_component = _signed_clamp(features.momentum * 20.0)
            ema_component = _signed_clamp(ema_spread * 180.0)
            slope_component = _signed_clamp(
                trend_slope / max(self._min_trend_slope * 2.5, 1e-6)
            )
            breakout_component = _signed_clamp(
                breakout_distance / max(self._breakout_overheat_distance, 1e-6)
            )
            volume_component = _signed_clamp((features.volume_ratio - 1.0) / 1.5)
            vol_penalty = _clamp(
                max(features.volatility - max_volatility, 0.0)
                / max(max_volatility, 1e-6)
            )
            score = (
                0.32 * ema_component
                + 0.24 * momentum_component
                + 0.18 * slope_component
                + 0.14 * breakout_component
                + 0.12 * volume_component
                - 0.20 * vol_penalty
            )
            score = _signed_clamp(score)

            if score >= risk_on_threshold:
                regime = "risk_on"
                bias = FactorBias.BULLISH
                strength = _clamp(score / max(risk_on_threshold, 1e-6))
            elif score <= -neutral_threshold:
                regime = "risk_off"
                bias = FactorBias.BEARISH
                strength = _clamp(
                    abs(score) / max(neutral_threshold, 1e-6)
                )
            else:
                regime = "neutral"
                bias = FactorBias.NEUTRAL
                strength = 0.0

            return FactorObservation(
                symbol=features.symbol,
                name="market_regime",
                category="market",
                timestamp=features.timestamp,
                bias=bias,
                strength=strength,
                value=score,
                threshold=neutral_threshold,
                horizon_minutes=180,
                expected_move_bps=40.0 + 110.0 * strength,
                thesis=(
                    f"Local regime {regime} from EMA spread {ema_spread:.4f}, "
                    f"momentum {features.momentum:.4f}, breakout {breakout_distance:.2f} ATR"
                ),
                invalidate_condition="Local trend alignment, momentum, and volatility revert toward neutral",
                metadata={
                    "mode": "per_symbol",
                    "ema_spread": ema_spread,
                    "trend_slope": trend_slope,
                    "breakout_distance": breakout_distance,
                    "volume_ratio": features.volume_ratio,
                    "volatility": features.volatility,
                    "score": score,
                    "regime": regime,
                },
            )

        if market_context is None:
            return FactorObservation(
                symbol=features.symbol,
                name="market_regime",
                category="market",
                timestamp=features.timestamp,
                bias=FactorBias.NEUTRAL,
                strength=0.0,
                value=0.0,
                threshold=self._neutral_threshold,
                horizon_minutes=240,
                expected_move_bps=0.0,
                thesis="Market regime context unavailable",
                invalidate_condition="Market regime context is restored",
                metadata={},
            )

        regime = market_context.get("regime", "neutral")
        score = float(market_context.get("score", 0.0))
        breadth = float(market_context.get("breadth", 0.5))
        bias = FactorBias.NEUTRAL
        strength = 0.0
        if regime == "risk_on":
            denom = max(self._risk_on_threshold - self._neutral_threshold, 1e-6)
            strength = _clamp((score - self._neutral_threshold) / denom)
            bias = FactorBias.BULLISH
        elif regime == "risk_off":
            denom = max(self._neutral_threshold + 1.0, 1e-6)
            strength = _clamp((self._neutral_threshold - score) / denom)
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="market_regime",
            category="market",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength,
            value=score,
            threshold=self._neutral_threshold,
            horizon_minutes=240,
            expected_move_bps=60.0 + 120.0 * strength,
            thesis=(
                f"Market regime {regime} with score {score:.3f} and breadth {breadth:.2f}"
            ),
            invalidate_condition="Benchmark trend and market breadth revert toward neutral",
            metadata=dict(market_context),
        )

    def _turtle_breakout_entry(self, features: FeatureVector) -> FactorObservation:
        breakout_distance = float(features.raw.get("breakout_distance", 0.0))
        trend_slope = float(features.raw.get("trend_slope", 0.0))
        ema_spread = (
            (features.ema_fast - features.ema_slow) / features.ema_slow
            if features.ema_slow > 0
            else 0.0
        )
        min_breakout = self._symbol_param(
            features.symbol,
            "turtle_min_breakout_distance",
            self._turtle_min_breakout_distance,
        )
        max_breakout = self._symbol_param(
            features.symbol,
            "turtle_max_breakout_distance",
            self._turtle_max_breakout_distance,
        )
        min_volume_ratio = self._symbol_param(
            features.symbol,
            "turtle_min_volume_ratio",
            self._turtle_min_volume_ratio,
        )
        trend_slack = self._symbol_param(
            features.symbol, "turtle_trend_slack", self._turtle_trend_slack
        )
        min_rsi = self._symbol_param(
            features.symbol, "turtle_min_rsi", self._turtle_min_rsi
        )
        max_rsi = self._symbol_param(
            features.symbol, "turtle_max_rsi", self._turtle_max_rsi
        )

        if not self._enable_turtle_breakout_entry:
            return FactorObservation(
                symbol=features.symbol,
                name="turtle_breakout_entry",
                category="entry_timing",
                timestamp=features.timestamp,
                bias=FactorBias.NEUTRAL,
                strength=0.0,
                value=breakout_distance,
                threshold=min_breakout,
                horizon_minutes=180,
                expected_move_bps=0.0,
                thesis="Turtle breakout entry disabled",
                invalidate_condition="Turtle breakout entry becomes enabled",
                metadata={},
            )

        bias = FactorBias.NEUTRAL
        strength = 0.0
        trend_floor = self._min_trend_slope * trend_slack
        if (
            breakout_distance >= min_breakout
            and breakout_distance <= max_breakout
            and trend_slope >= trend_floor
            and ema_spread > 0
            and features.volume_ratio >= min_volume_ratio
            and min_rsi <= features.rsi <= max_rsi
            and features.momentum > 0
        ):
            distance_width = max(
                max_breakout - min_breakout,
                1e-6,
            )
            breakout_score = _clamp(
                (breakout_distance - min_breakout) / distance_width
            )
            trend_score = _clamp(
                (trend_slope - trend_floor) / max(abs(trend_floor) * 3.0, 1e-6)
            )
            ema_score = _clamp(ema_spread / 0.01)
            volume_score = _clamp(
                (features.volume_ratio - min_volume_ratio + 0.2) / 1.2
            )
            momentum_score = _clamp(features.momentum / 0.02)
            strength = _clamp(
                0.30 * breakout_score
                + 0.22 * trend_score
                + 0.18 * ema_score
                + 0.15 * volume_score
                + 0.15 * momentum_score
            )
            bias = FactorBias.BULLISH

        return FactorObservation(
            symbol=features.symbol,
            name="turtle_breakout_entry",
            category="entry_timing",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias == FactorBias.BULLISH else 0.0,
            value=breakout_distance,
            threshold=min_breakout,
            horizon_minutes=180,
            expected_move_bps=55.0 + 140.0 * strength,
            thesis=(
                f"Turtle breakout {breakout_distance:.2f} ATR with slope {trend_slope:.4f}, "
                f"EMA spread {ema_spread:.4f}, volume ratio {features.volume_ratio:.2f}"
            ),
            invalidate_condition="Breakout fails back into the channel or trend slope collapses",
            metadata={
                "breakout_distance": breakout_distance,
                "trend_slope": trend_slope,
                "ema_spread": ema_spread,
                "volume_ratio": features.volume_ratio,
                "rsi": features.rsi,
                "momentum": features.momentum,
            },
        )

    def _trend_alignment(
        self,
        features: FeatureVector,
        candles_15m: Optional[List[OHLCV]],
        candles_1h: Optional[List[OHLCV]],
    ) -> FactorObservation:
        primary_diff = 0.0
        if features.ema_slow > 0:
            primary_diff = (features.ema_fast - features.ema_slow) / features.ema_slow

        tf15 = 0.0
        if candles_15m and len(candles_15m) >= 2:
            prev = candles_15m[-2].close
            tf15 = (candles_15m[-1].close - prev) / prev if prev > 0 else 0.0

        tf1h = 0.0
        if candles_1h and len(candles_1h) >= 4:
            prev = candles_1h[-4].close
            tf1h = (candles_1h[-1].close - prev) / prev if prev > 0 else 0.0

        trend_strength = _clamp(max(primary_diff * 180.0, 0.0) + max(tf15 * 40.0, 0.0) + max(tf1h * 20.0, 0.0))
        bias = FactorBias.NEUTRAL
        if primary_diff > 0 and (tf15 >= -0.001) and (tf1h >= -0.002):
            bias = FactorBias.BULLISH
        elif primary_diff < 0 and tf15 < 0 and tf1h < 0:
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="trend_alignment",
            category="price_structure",
            timestamp=features.timestamp,
            bias=bias,
            strength=trend_strength if bias != FactorBias.NEUTRAL else 0.0,
            value=primary_diff,
            threshold=0.0,
            horizon_minutes=240,
            expected_move_bps=80.0 + 180.0 * trend_strength,
            thesis=(
                f"Trend aligned with EMA spread {primary_diff:.4f}, 15m drift {tf15:.4f}, 1h drift {tf1h:.4f}"
            ),
            invalidate_condition="Primary EMA spread flips negative or higher-timeframe drift turns down",
            metadata={"ema_spread": primary_diff, "tf15": tf15, "tf1h": tf1h},
        )

    def _momentum_impulse(self, features: FeatureVector) -> FactorObservation:
        rsi_penalty = _clamp((features.rsi - 68.0) / 20.0)
        momentum_strength = _clamp(
            max(features.momentum * 25.0, 0.0) * (1.0 - 0.5 * rsi_penalty)
        )

        bias = FactorBias.NEUTRAL
        if features.momentum > 0 and features.rsi < 75:
            bias = FactorBias.BULLISH
        elif features.momentum < 0:
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="momentum_impulse",
            category="price_structure",
            timestamp=features.timestamp,
            bias=bias,
            strength=momentum_strength if bias != FactorBias.NEUTRAL else 0.0,
            value=features.momentum,
            threshold=0.0,
            horizon_minutes=180,
            expected_move_bps=60.0 + 160.0 * momentum_strength,
            thesis=f"Momentum {features.momentum:.4f} with RSI {features.rsi:.1f}",
            invalidate_condition="Momentum turns negative or RSI closes back below neutral after breakout failure",
            metadata={"momentum": features.momentum, "rsi": features.rsi},
        )

    def _volume_confirmation(self, features: FeatureVector) -> FactorObservation:
        vol_strength = _clamp((features.volume_ratio - self._min_volume_ratio) / 1.2)
        bias = FactorBias.BULLISH if features.volume_ratio >= self._min_volume_ratio else FactorBias.NEUTRAL

        return FactorObservation(
            symbol=features.symbol,
            name="volume_confirmation",
            category="flow",
            timestamp=features.timestamp,
            bias=bias,
            strength=vol_strength if bias == FactorBias.BULLISH else 0.0,
            value=features.volume_ratio,
            threshold=self._min_volume_ratio,
            horizon_minutes=120,
            expected_move_bps=40.0 + 100.0 * vol_strength,
            thesis=f"Volume ratio {features.volume_ratio:.2f} vs trigger {self._min_volume_ratio:.2f}",
            invalidate_condition="Volume ratio falls back below confirmation threshold",
            metadata={"volume_ratio": features.volume_ratio},
        )

    def _pullback_reentry(self, features: FeatureVector) -> FactorObservation:
        breakout_distance = float(features.raw.get("breakout_distance", 0.0))
        trend_slope = float(features.raw.get("trend_slope", 0.0))
        ema_spread = (
            (features.ema_fast - features.ema_slow) / features.ema_slow
            if features.ema_slow > 0
            else 0.0
        )

        if not self._enable_pullback_reentry:
            return FactorObservation(
                symbol=features.symbol,
                name="pullback_reentry",
                category="entry_timing",
                timestamp=features.timestamp,
                bias=FactorBias.NEUTRAL,
                strength=0.0,
                value=breakout_distance,
                threshold=self._pullback_max_breakout_distance,
                horizon_minutes=120,
                expected_move_bps=0.0,
                thesis="Pullback re-entry disabled",
                invalidate_condition="Pullback re-entry becomes enabled",
                metadata={},
            )

        trend_floor = self._min_trend_slope * self._pullback_trend_slack
        bias = FactorBias.NEUTRAL
        strength = 0.0

        if (
            ema_spread > 0
            and trend_slope >= trend_floor
            and self._pullback_min_breakout_distance
            <= breakout_distance
            <= self._pullback_max_breakout_distance
            and self._pullback_min_rsi <= features.rsi <= self._pullback_max_rsi
            and self._pullback_min_momentum
            <= features.momentum
            <= self._pullback_max_momentum
            and features.volume_ratio >= self._pullback_min_volume_ratio
        ):
            distance_width = max(
                self._pullback_max_breakout_distance
                - self._pullback_min_breakout_distance,
                1e-6,
            )
            rsi_width = max(self._pullback_max_rsi - self._pullback_min_rsi, 1e-6)
            distance_score = 1.0 - min(
                abs(breakout_distance - self._pullback_target_breakout_distance)
                / max(distance_width * 0.5, 1e-6),
                1.0,
            )
            rsi_score = 1.0 - min(
                abs(features.rsi - self._pullback_target_rsi)
                / max(rsi_width * 0.5, 1e-6),
                1.0,
            )
            trend_score = _clamp(
                (trend_slope - trend_floor) / max(abs(trend_floor) * 3.0, 1e-6)
            )
            ema_score = _clamp(ema_spread / 0.01)
            strength = _clamp(
                0.35 * distance_score
                + 0.20 * rsi_score
                + 0.25 * trend_score
                + 0.20 * ema_score
            )
            bias = FactorBias.BULLISH

        return FactorObservation(
            symbol=features.symbol,
            name="pullback_reentry",
            category="entry_timing",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias == FactorBias.BULLISH else 0.0,
            value=breakout_distance,
            threshold=self._pullback_max_breakout_distance,
            horizon_minutes=120,
            expected_move_bps=40.0 + 110.0 * strength,
            thesis=(
                f"Trend pullback {breakout_distance:.2f} ATR with RSI {features.rsi:.1f}, "
                f"momentum {features.momentum:.4f}, slope {trend_slope:.4f}"
            ),
            invalidate_condition="EMA spread collapses, pullback becomes a breakdown, or price re-extends without confirmation",
            metadata={
                "breakout_distance": breakout_distance,
                "rsi": features.rsi,
                "momentum": features.momentum,
                "trend_slope": trend_slope,
                "ema_spread": ema_spread,
                "volume_ratio": features.volume_ratio,
            },
        )

    def _volatility_compression_entry(self, features: FeatureVector) -> FactorObservation:
        breakout_distance = float(features.raw.get("breakout_distance", 0.0))
        trend_slope = float(features.raw.get("trend_slope", 0.0))
        realized_vol_short = float(
            features.raw.get("realized_vol_short", features.volatility)
        )
        ema_spread = (
            (features.ema_fast - features.ema_slow) / features.ema_slow
            if features.ema_slow > 0
            else 0.0
        )

        if not self._enable_volatility_compression_entry:
            return FactorObservation(
                symbol=features.symbol,
                name="volatility_compression_entry",
                category="volatility",
                timestamp=features.timestamp,
                bias=FactorBias.NEUTRAL,
                strength=0.0,
                value=realized_vol_short,
                threshold=self._vol_compression_short_vol_ratio_max,
                horizon_minutes=180,
                expected_move_bps=0.0,
                thesis="Volatility compression entry disabled",
                invalidate_condition="Volatility compression entry becomes enabled",
                metadata={},
            )

        short_vol_ratio = realized_vol_short / max(features.volatility, 1e-6)
        trend_floor = self._min_trend_slope * self._vol_compression_trend_slack
        bias = FactorBias.NEUTRAL
        strength = 0.0

        if (
            ema_spread > 0
            and trend_slope >= trend_floor
            and self._vol_compression_min_breakout_distance
            <= breakout_distance
            <= self._vol_compression_max_breakout_distance
            and self._vol_compression_min_rsi <= features.rsi <= self._vol_compression_max_rsi
            and short_vol_ratio <= self._vol_compression_short_vol_ratio_max
            and self._vol_compression_min_volume_ratio
            <= features.volume_ratio
            <= self._vol_compression_max_volume_ratio
        ):
            compression_score = _clamp(
                (
                    self._vol_compression_short_vol_ratio_max - short_vol_ratio
                )
                / max(self._vol_compression_short_vol_ratio_max, 1e-6)
            )
            distance_score = 1.0 - min(
                abs(
                    breakout_distance
                    - (
                        self._vol_compression_min_breakout_distance
                        + self._vol_compression_max_breakout_distance
                    )
                    / 2.0
                )
                / max(
                    (
                        self._vol_compression_max_breakout_distance
                        - self._vol_compression_min_breakout_distance
                    )
                    * 0.5,
                    1e-6,
                ),
                1.0,
            )
            trend_score = _clamp(
                (trend_slope - trend_floor) / max(abs(trend_floor) * 3.0, 1e-6)
            )
            volume_mid = (
                self._vol_compression_min_volume_ratio
                + self._vol_compression_max_volume_ratio
            ) / 2.0
            volume_score = 1.0 - min(
                abs(features.volume_ratio - volume_mid)
                / max(
                    (
                        self._vol_compression_max_volume_ratio
                        - self._vol_compression_min_volume_ratio
                    )
                    * 0.5,
                    1e-6,
                ),
                1.0,
            )
            strength = _clamp(
                0.35 * compression_score
                + 0.25 * distance_score
                + 0.25 * trend_score
                + 0.15 * volume_score
            )
            bias = FactorBias.BULLISH

        return FactorObservation(
            symbol=features.symbol,
            name="volatility_compression_entry",
            category="volatility",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias == FactorBias.BULLISH else 0.0,
            value=short_vol_ratio,
            threshold=self._vol_compression_short_vol_ratio_max,
            horizon_minutes=180,
            expected_move_bps=35.0 + 95.0 * strength,
            thesis=(
                f"Short vol ratio {short_vol_ratio:.2f} with breakout {breakout_distance:.2f} ATR, "
                f"RSI {features.rsi:.1f}, slope {trend_slope:.4f}"
            ),
            invalidate_condition="Trend weakens or short-term volatility expands back above the compression threshold",
            metadata={
                "breakout_distance": breakout_distance,
                "trend_slope": trend_slope,
                "realized_vol_short": realized_vol_short,
                "volatility": features.volatility,
                "short_vol_ratio": short_vol_ratio,
                "volume_ratio": features.volume_ratio,
                "rsi": features.rsi,
            },
        )

    def _grid_reversion_entry(self, features: FeatureVector) -> FactorObservation:
        breakout_distance = float(features.raw.get("breakout_distance", 0.0))
        trend_slope = float(features.raw.get("trend_slope", 0.0))
        realized_vol_short = float(
            features.raw.get("realized_vol_short", features.volatility)
        )
        ema_spread = (
            (features.ema_fast - features.ema_slow) / features.ema_slow
            if features.ema_slow > 0
            else 0.0
        )
        short_vol_ratio = realized_vol_short / max(features.volatility, 1e-6)
        min_breakout = self._symbol_param(
            features.symbol,
            "grid_reversion_min_breakout_distance",
            self._grid_reversion_min_breakout_distance,
        )
        max_breakout = self._symbol_param(
            features.symbol,
            "grid_reversion_max_breakout_distance",
            self._grid_reversion_max_breakout_distance,
        )
        target_breakout = self._symbol_param(
            features.symbol,
            "grid_reversion_target_breakout_distance",
            self._grid_reversion_target_breakout_distance,
        )
        min_rsi = self._symbol_param(
            features.symbol, "grid_reversion_min_rsi", self._grid_reversion_min_rsi
        )
        max_rsi = self._symbol_param(
            features.symbol, "grid_reversion_max_rsi", self._grid_reversion_max_rsi
        )
        target_rsi = self._symbol_param(
            features.symbol,
            "grid_reversion_target_rsi",
            self._grid_reversion_target_rsi,
        )
        min_momentum = self._symbol_param(
            features.symbol,
            "grid_reversion_min_momentum",
            self._grid_reversion_min_momentum,
        )
        max_momentum = self._symbol_param(
            features.symbol,
            "grid_reversion_max_momentum",
            self._grid_reversion_max_momentum,
        )
        min_volume_ratio = self._symbol_param(
            features.symbol,
            "grid_reversion_min_volume_ratio",
            self._grid_reversion_min_volume_ratio,
        )
        max_volume_ratio = self._symbol_param(
            features.symbol,
            "grid_reversion_max_volume_ratio",
            self._grid_reversion_max_volume_ratio,
        )
        short_vol_ratio_max = self._symbol_param(
            features.symbol,
            "grid_reversion_short_vol_ratio_max",
            self._grid_reversion_short_vol_ratio_max,
        )
        trend_slack = self._symbol_param(
            features.symbol,
            "grid_reversion_trend_slack",
            self._grid_reversion_trend_slack,
        )

        if not self._enable_grid_reversion_entry:
            return FactorObservation(
                symbol=features.symbol,
                name="grid_reversion_entry",
                category="entry_timing",
                timestamp=features.timestamp,
                bias=FactorBias.NEUTRAL,
                strength=0.0,
                value=breakout_distance,
                threshold=max_breakout,
                horizon_minutes=90,
                expected_move_bps=0.0,
                thesis="Grid reversion entry disabled",
                invalidate_condition="Grid reversion entry becomes enabled",
                metadata={},
            )

        trend_floor = self._min_trend_slope * trend_slack
        bias = FactorBias.NEUTRAL
        strength = 0.0

        if (
            ema_spread > 0
            and trend_slope >= trend_floor
            and min_breakout <= breakout_distance <= max_breakout
            and min_rsi <= features.rsi <= max_rsi
            and min_momentum <= features.momentum <= max_momentum
            and min_volume_ratio <= features.volume_ratio <= max_volume_ratio
            and short_vol_ratio <= short_vol_ratio_max
        ):
            distance_width = max(
                max_breakout - min_breakout,
                1e-6,
            )
            rsi_width = max(max_rsi - min_rsi, 1e-6)
            momentum_width = max(
                max_momentum - min_momentum,
                1e-6,
            )
            distance_score = 1.0 - min(
                abs(breakout_distance - target_breakout)
                / max(distance_width * 0.5, 1e-6),
                1.0,
            )
            rsi_score = 1.0 - min(
                abs(features.rsi - target_rsi)
                / max(rsi_width * 0.5, 1e-6),
                1.0,
            )
            momentum_mid = (min_momentum + max_momentum) / 2.0
            momentum_score = 1.0 - min(
                abs(features.momentum - momentum_mid)
                / max(momentum_width * 0.5, 1e-6),
                1.0,
            )
            vol_score = _clamp(
                (short_vol_ratio_max - short_vol_ratio)
                / max(short_vol_ratio_max, 1e-6)
            )
            trend_score = _clamp(
                (trend_slope - trend_floor) / max(abs(trend_floor) * 3.0, 1e-6)
            )
            strength = _clamp(
                0.28 * distance_score
                + 0.22 * rsi_score
                + 0.16 * momentum_score
                + 0.16 * vol_score
                + 0.18 * trend_score
            )
            bias = FactorBias.BULLISH

        return FactorObservation(
            symbol=features.symbol,
            name="grid_reversion_entry",
            category="entry_timing",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias == FactorBias.BULLISH else 0.0,
            value=breakout_distance,
            threshold=max_breakout,
            horizon_minutes=90,
            expected_move_bps=25.0 + 85.0 * strength,
            thesis=(
                f"Grid-style reversion at {breakout_distance:.2f} ATR with RSI {features.rsi:.1f}, "
                f"momentum {features.momentum:.4f}, short vol ratio {short_vol_ratio:.2f}"
            ),
            invalidate_condition="Pullback turns into breakdown or short-term volatility expands sharply",
            metadata={
                "breakout_distance": breakout_distance,
                "trend_slope": trend_slope,
                "rsi": features.rsi,
                "momentum": features.momentum,
                "volume_ratio": features.volume_ratio,
                "realized_vol_short": realized_vol_short,
                "volatility": features.volatility,
                "short_vol_ratio": short_vol_ratio,
            },
        )

    def _breakout_confirmation(self, features: FeatureVector) -> FactorObservation:
        breakout_distance = float(features.raw.get("breakout_distance", 0.0))
        trend_slope = float(features.raw.get("trend_slope", 0.0))
        volume_zscore = float(features.raw.get("volume_zscore", 0.0))

        bias = FactorBias.NEUTRAL
        strength = 0.0
        if (
            breakout_distance >= self._breakout_min_distance
            and breakout_distance <= self._breakout_overheat_distance
            and trend_slope >= self._min_trend_slope
            and volume_zscore >= self._min_volume_zscore
        ):
            distance_component = breakout_distance / max(
                self._breakout_overheat_distance, 1e-6
            )
            slope_component = trend_slope / max(self._min_trend_slope * 4.0, 1e-6)
            volume_component = (volume_zscore + 1.0) / 3.0
            strength = _clamp(
                0.5 * distance_component
                + 0.3 * slope_component
                + 0.2 * volume_component
            )
            bias = FactorBias.BULLISH
        elif breakout_distance > self._breakout_overheat_distance:
            strength = _clamp(
                (breakout_distance - self._breakout_overheat_distance)
                / max(self._breakout_overheat_distance, 1e-6)
            )
            bias = FactorBias.BEARISH
        elif breakout_distance < -0.50 and trend_slope < 0:
            strength = _clamp(
                min(abs(breakout_distance), 2.0) / 2.0
                + min(abs(trend_slope) / max(self._min_trend_slope * 4.0, 1e-6), 1.0)
                * 0.25
            )
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="breakout_confirmation",
            category="price_structure",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias != FactorBias.NEUTRAL else 0.0,
            value=breakout_distance,
            threshold=self._breakout_min_distance,
            horizon_minutes=180,
            expected_move_bps=30.0 + 130.0 * strength,
            thesis=(
                f"Breakout distance {breakout_distance:.2f} ATR, trend slope {trend_slope:.4f}, "
                f"volume z-score {volume_zscore:.2f}"
            ),
            invalidate_condition="Breakout distance compresses back into the prior range or price becomes overextended",
            metadata={
                "breakout_distance": breakout_distance,
                "trend_slope": trend_slope,
                "volume_zscore": volume_zscore,
            },
        )

    def _overextension_exit(
        self,
        features: FeatureVector,
        supplementary: dict,
    ) -> FactorObservation:
        breakout_distance = float(features.raw.get("breakout_distance", 0.0))
        volume_zscore = float(features.raw.get("volume_zscore", 0.0))
        taker_ratio = supplementary.get("taker_ratio", features.taker_ratio)
        funding_rate = supplementary.get("funding_rate", features.funding_rate)

        if not self._enable_overextension_exit:
            return FactorObservation(
                symbol=features.symbol,
                name="overextension_exit",
                category="risk",
                timestamp=features.timestamp,
                bias=FactorBias.NEUTRAL,
                strength=0.0,
                value=breakout_distance,
                threshold=self._overextension_min_breakout_distance,
                horizon_minutes=60,
                expected_move_bps=0.0,
                thesis="Overextension exit disabled",
                invalidate_condition="Overextension exit becomes enabled",
                metadata={},
            )

        bias = FactorBias.NEUTRAL
        strength = 0.0
        if (
            breakout_distance >= self._overextension_min_breakout_distance
            and features.rsi >= self._overextension_min_rsi
            and (
                volume_zscore >= self._overextension_min_volume_zscore
                or taker_ratio >= self._overextension_min_taker_ratio
            )
        ):
            distance_score = _clamp(
                (breakout_distance - self._overextension_min_breakout_distance)
                / max(self._overextension_min_breakout_distance + 0.75, 1e-6)
            )
            rsi_score = _clamp(
                (features.rsi - self._overextension_min_rsi)
                / max(85.0 - self._overextension_min_rsi, 1e-6)
            )
            volume_score = _clamp(
                (volume_zscore - self._overextension_min_volume_zscore + 1.0) / 3.0
            )
            taker_score = _clamp(
                (taker_ratio - self._overextension_min_taker_ratio) / 0.20
            )
            funding_score = _clamp(
                max(funding_rate, 0.0) / max(self._max_funding_rate * 2.0, 1e-6)
            )
            strength = _clamp(
                0.35 * distance_score
                + 0.25 * rsi_score
                + 0.15 * volume_score
                + 0.15 * taker_score
                + 0.10 * funding_score
            )
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="overextension_exit",
            category="risk",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias == FactorBias.BEARISH else 0.0,
            value=breakout_distance,
            threshold=self._overextension_min_breakout_distance,
            horizon_minutes=60,
            expected_move_bps=20.0 + 90.0 * strength,
            thesis=(
                f"Overextension {breakout_distance:.2f} ATR with RSI {features.rsi:.1f}, "
                f"volume z-score {volume_zscore:.2f}, taker ratio {taker_ratio:.3f}"
            ),
            invalidate_condition="Price cools back toward the breakout base and momentum pressure normalizes",
            metadata={
                "breakout_distance": breakout_distance,
                "rsi": features.rsi,
                "volume_zscore": volume_zscore,
                "taker_ratio": taker_ratio,
                "funding_rate": funding_rate,
            },
        )

    def _liquidity_balance(
        self,
        features: FeatureVector,
        supplementary: dict,
    ) -> FactorObservation:
        imbalance = supplementary.get(
            "order_book_imbalance", features.order_book_imbalance
        )
        strength = _clamp(abs(imbalance - 1.0) / 0.25)
        inverse_threshold = 1.0 / max(self._min_order_book_imbalance, 1e-6)

        bias = FactorBias.NEUTRAL
        if imbalance >= self._min_order_book_imbalance:
            bias = FactorBias.BULLISH
        elif 0.0 < imbalance <= inverse_threshold:
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="liquidity_balance",
            category="microstructure",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias != FactorBias.NEUTRAL else 0.0,
            value=imbalance,
            threshold=self._min_order_book_imbalance,
            horizon_minutes=30,
            expected_move_bps=20.0 + 70.0 * strength,
            thesis=f"Order-book imbalance {imbalance:.3f}",
            invalidate_condition="Top-of-book imbalance mean reverts back to neutral",
            metadata={"order_book_imbalance": imbalance},
        )

    def _perp_crowding(
        self,
        features: FeatureVector,
        supplementary: dict,
        supplementary_history: dict,
    ) -> FactorObservation:
        has_funding = supplementary.get("has_funding_rate", False)
        has_taker_ratio = supplementary.get("has_taker_ratio", False)
        has_open_interest = supplementary.get("has_open_interest", False)

        funding = supplementary.get("funding_rate", features.funding_rate)
        taker_ratio = supplementary.get("taker_ratio", features.taker_ratio)
        open_interest = supplementary.get("open_interest", 0.0)

        oi_hist = supplementary_history.get("open_interest", [])
        oi_window = oi_hist[-self._open_interest_lookback_samples :]
        oi_change = 0.0
        if has_open_interest and len(oi_window) >= 2 and oi_window[0] > 0:
            oi_change = (oi_window[-1] - oi_window[0]) / oi_window[0]

        crowded_components = [0.0]
        if has_funding:
            crowded_components.append(
                max(funding - self._max_funding_rate, 0.0)
                / max(self._max_funding_rate, 1e-6)
            )
        if has_taker_ratio:
            crowded_components.append(
                max(taker_ratio - self._max_taker_ratio, 0.0)
                / max(self._max_taker_ratio, 1e-6)
            )
        if has_open_interest:
            crowded_components.append(
                max(oi_change - self._max_open_interest_change, 0.0)
                / max(self._max_open_interest_change, 1e-6)
            )
        crowded = max(crowded_components)
        strength = _clamp(crowded)

        bias = FactorBias.NEUTRAL
        if strength > 0:
            bias = FactorBias.BEARISH
        elif (
            has_funding
            and has_taker_ratio
            and 0.0 < funding < self._max_funding_rate * 0.5
            and taker_ratio <= 1.05
        ):
            bias = FactorBias.BULLISH
            strength = 0.15

        return FactorObservation(
            symbol=features.symbol,
            name="perp_crowding",
            category="derivatives",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength,
            value=funding,
            threshold=self._max_funding_rate,
            horizon_minutes=180,
            expected_move_bps=40.0 + 120.0 * strength,
            thesis=(
                f"Funding {funding:.6f}, taker ratio {taker_ratio:.3f}, open-interest change {oi_change:.3f}"
            ),
            invalidate_condition="Funding and taker pressure normalize back toward neutral",
            metadata={
                "funding_rate": funding,
                "taker_ratio": taker_ratio,
                "open_interest": open_interest,
                "open_interest_change": oi_change,
                "has_funding_rate": has_funding,
                "has_taker_ratio": has_taker_ratio,
                "has_open_interest": has_open_interest,
            },
        )

    def _onchain_stale_days(self, supplementary: dict, supplementary_history: dict) -> float:
        history = supplementary_history.get("onchain_mvrv_ratio", [])
        if history and supplementary.get("has_onchain_mvrv_ratio", False):
            return 0.0
        return self._onchain_stale_days_max + 1.0

    def _onchain_mvrv_state(
        self,
        features: FeatureVector,
        supplementary: dict,
    ) -> FactorObservation:
        has_mvrv = supplementary.get("has_onchain_mvrv_ratio", False)
        mvrv = float(supplementary.get("onchain_mvrv_ratio", 0.0))
        stale = not has_mvrv
        bias = FactorBias.NEUTRAL
        strength = 0.0

        if not stale:
            if mvrv <= self._onchain_mvrv_bullish_max:
                strength = _clamp(
                    (self._onchain_mvrv_bullish_max - mvrv)
                    / max(self._onchain_mvrv_bullish_max, 1e-6)
                )
                bias = FactorBias.BULLISH
            elif mvrv >= self._onchain_mvrv_bearish_min:
                strength = _clamp(
                    (mvrv - self._onchain_mvrv_bearish_min)
                    / max(self._onchain_mvrv_bearish_min, 1e-6)
                )
                bias = FactorBias.BEARISH

        thesis = (
            f"On-chain MVRV {mvrv:.2f}"
            if has_mvrv
            else "On-chain MVRV unavailable for this symbol/source"
        )
        return FactorObservation(
            symbol=features.symbol,
            name="onchain_mvrv_state",
            category="onchain",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias != FactorBias.NEUTRAL else 0.0,
            value=mvrv,
            threshold=self._onchain_mvrv_bullish_max,
            horizon_minutes=1440,
            expected_move_bps=25.0 + 70.0 * strength,
            thesis=thesis,
            invalidate_condition="MVRV reverts back toward its neutral valuation band",
            metadata={"onchain_mvrv_ratio": mvrv, "has_onchain_mvrv_ratio": has_mvrv},
        )

    def _onchain_activity_state(
        self,
        features: FeatureVector,
        supplementary: dict,
        supplementary_history: dict,
    ) -> FactorObservation:
        has_addr = supplementary.get("has_onchain_active_addresses", False)
        has_tx = supplementary.get("has_onchain_tx_count", False)
        addr = float(supplementary.get("onchain_active_addresses", 0.0))
        tx_count = float(supplementary.get("onchain_tx_count", 0.0))
        addr_hist = supplementary_history.get("onchain_active_addresses", [])
        tx_hist = supplementary_history.get("onchain_tx_count", [])
        lookback = max(self._onchain_activity_lookback, 2)
        addr_ref = addr_hist[-lookback:-1] if len(addr_hist) >= 2 else []
        tx_ref = tx_hist[-lookback:-1] if len(tx_hist) >= 2 else []

        bias = FactorBias.NEUTRAL
        strength = 0.0
        activity_growth = 0.0
        if has_addr and has_tx and addr > 0 and tx_count > 0 and addr_ref and tx_ref:
            addr_base = sum(addr_ref) / max(len(addr_ref), 1)
            tx_base = sum(tx_ref) / max(len(tx_ref), 1)
            addr_growth = (addr - addr_base) / max(addr_base, 1e-6)
            tx_growth = (tx_count - tx_base) / max(tx_base, 1e-6)
            activity_growth = 0.5 * addr_growth + 0.5 * tx_growth
            if activity_growth >= self._onchain_activity_bullish_growth:
                bias = FactorBias.BULLISH
                strength = _clamp(activity_growth / 0.50)
            elif activity_growth <= self._onchain_activity_bearish_growth:
                bias = FactorBias.BEARISH
                strength = _clamp(abs(activity_growth) / 0.50)

        return FactorObservation(
            symbol=features.symbol,
            name="onchain_activity_state",
            category="onchain",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias != FactorBias.NEUTRAL else 0.0,
            value=activity_growth,
            threshold=self._onchain_activity_bullish_growth,
            horizon_minutes=1440,
            expected_move_bps=20.0 + 60.0 * strength,
            thesis=(
                f"On-chain activity growth {activity_growth:.2%} from active addresses {addr:.0f} "
                f"and tx count {tx_count:.0f}"
            ),
            invalidate_condition="Active addresses and transfer count roll back to baseline",
            metadata={
                "onchain_active_addresses": addr,
                "onchain_tx_count": tx_count,
                "activity_growth": activity_growth,
                "has_onchain_active_addresses": has_addr,
                "has_onchain_tx_count": has_tx,
            },
        )

    def _onchain_nvt_proxy(
        self,
        features: FeatureVector,
        supplementary: dict,
        supplementary_history: dict,
    ) -> FactorObservation:
        has_nvt_proxy = supplementary.get("has_onchain_nvt_proxy", False)
        current = float(supplementary.get("onchain_nvt_proxy", 0.0))
        history = supplementary_history.get("onchain_nvt_proxy", [])
        lookback = max(self._onchain_nvt_proxy_lookback, 5)
        ref = history[-lookback:-1] if len(history) >= 2 else []

        bias = FactorBias.NEUTRAL
        strength = 0.0
        zscore = 0.0
        if has_nvt_proxy and current > 0 and len(ref) >= 5:
            mean = sum(ref) / len(ref)
            variance = sum((value - mean) ** 2 for value in ref) / len(ref)
            std = variance ** 0.5
            if std > 1e-10:
                zscore = (current - mean) / std
            if zscore <= self._onchain_nvt_proxy_bullish_zscore:
                bias = FactorBias.BULLISH
                strength = _clamp(abs(zscore) / 3.0)
            elif zscore >= self._onchain_nvt_proxy_bearish_zscore:
                bias = FactorBias.BEARISH
                strength = _clamp(abs(zscore) / 3.0)

        return FactorObservation(
            symbol=features.symbol,
            name="onchain_nvt_proxy",
            category="onchain",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias != FactorBias.NEUTRAL else 0.0,
            value=zscore,
            threshold=self._onchain_nvt_proxy_bearish_zscore,
            horizon_minutes=1440,
            expected_move_bps=20.0 + 55.0 * strength,
            thesis=(
                f"On-chain NVT proxy z-score {zscore:.2f} from market cap "
                f"{supplementary.get('onchain_market_cap_usd', 0.0):.2f} and tx count "
                f"{supplementary.get('onchain_tx_count', 0.0):.0f}"
            ),
            invalidate_condition="Network-value-to-activity proxy reverts toward its rolling mean",
            metadata={
                "onchain_nvt_proxy": current,
                "onchain_nvt_proxy_zscore": zscore,
                "has_onchain_nvt_proxy": has_nvt_proxy,
            },
        )

    def _onchain_ecosystem_state(
        self,
        features: FeatureVector,
        supplementary: dict,
        supplementary_history: dict,
    ) -> FactorObservation:
        lookback = max(self._onchain_ecosystem_lookback, 3)
        metric_specs = (
            (
                "onchain_chain_tvl_usd",
                "has_onchain_chain_tvl_usd",
                0.20,
                "TVL",
            ),
            (
                "onchain_chain_stablecoin_supply_usd",
                "has_onchain_chain_stablecoin_supply_usd",
                0.30,
                "stablecoin supply",
            ),
            (
                "onchain_chain_dex_volume_usd",
                "has_onchain_chain_dex_volume_usd",
                0.30,
                "DEX volume",
            ),
            (
                "onchain_chain_fees_usd",
                "has_onchain_chain_fees_usd",
                0.20,
                "fees",
            ),
        )

        weighted_growth = 0.0
        total_weight = 0.0
        component_growth: dict[str, float] = {}

        for value_key, has_key, weight, label in metric_specs:
            if not supplementary.get(has_key, False):
                continue
            current = float(supplementary.get(value_key, 0.0))
            history = supplementary_history.get(value_key, [])
            ref = history[-lookback:-1] if len(history) >= 2 else []
            if current <= 0.0 or not ref:
                continue
            baseline = sum(ref) / len(ref)
            if baseline <= 0.0:
                continue
            growth = (current - baseline) / baseline
            growth = _signed_clamp(growth, low=-1.0, high=2.0)
            weighted_growth += weight * growth
            total_weight += weight
            component_growth[label] = growth

        bias = FactorBias.NEUTRAL
        strength = 0.0
        ecosystem_growth = 0.0
        if total_weight >= 0.50:
            ecosystem_growth = weighted_growth / total_weight
            if ecosystem_growth >= self._onchain_ecosystem_bullish_growth:
                bias = FactorBias.BULLISH
                strength = _clamp(ecosystem_growth / 0.50)
            elif ecosystem_growth <= self._onchain_ecosystem_bearish_growth:
                bias = FactorBias.BEARISH
                strength = _clamp(abs(ecosystem_growth) / 0.50)

        if component_growth:
            component_text = ", ".join(
                f"{label} {growth:.1%}" for label, growth in component_growth.items()
            )
            thesis = f"Chain ecosystem growth {ecosystem_growth:.1%} from {component_text}"
        else:
            thesis = "Chain ecosystem metrics unavailable for this symbol/source"

        return FactorObservation(
            symbol=features.symbol,
            name="onchain_ecosystem_state",
            category="onchain",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias != FactorBias.NEUTRAL else 0.0,
            value=ecosystem_growth,
            threshold=self._onchain_ecosystem_bullish_growth,
            horizon_minutes=1440,
            expected_move_bps=20.0 + 65.0 * strength,
            thesis=thesis,
            invalidate_condition="Chain TVL, stablecoin liquidity, DEX volume, and fees revert back to baseline",
            metadata={
                "ecosystem_growth": ecosystem_growth,
                "components": component_growth,
                "has_onchain_chain_tvl_usd": supplementary.get(
                    "has_onchain_chain_tvl_usd", False
                ),
                "has_onchain_chain_stablecoin_supply_usd": supplementary.get(
                    "has_onchain_chain_stablecoin_supply_usd", False
                ),
                "has_onchain_chain_dex_volume_usd": supplementary.get(
                    "has_onchain_chain_dex_volume_usd", False
                ),
                "has_onchain_chain_fees_usd": supplementary.get(
                    "has_onchain_chain_fees_usd", False
                ),
            },
        )

    def _volatility_regime(self, features: FeatureVector) -> FactorObservation:
        excess_vol = max(features.volatility - self._max_volatility, 0.0)
        strength = _clamp(excess_vol / max(self._max_volatility, 1e-6))
        bias = FactorBias.BEARISH if features.volatility > self._max_volatility else FactorBias.NEUTRAL

        return FactorObservation(
            symbol=features.symbol,
            name="volatility_regime",
            category="risk",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias == FactorBias.BEARISH else 0.0,
            value=features.volatility,
            threshold=self._max_volatility,
            horizon_minutes=60,
            expected_move_bps=0.0,
            thesis=f"Realized volatility {features.volatility:.4f} vs cap {self._max_volatility:.4f}",
            invalidate_condition="Volatility cools back into the strategy operating range",
            metadata={"volatility": features.volatility},
        )
