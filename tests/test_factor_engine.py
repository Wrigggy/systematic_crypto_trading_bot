"""Tests for strategy/factor_engine.py — explicit factor observations."""

from __future__ import annotations

from datetime import datetime, timedelta

from core.models import FactorBias, FeatureVector, OHLCV
from strategy.factor_engine import FactorEngine


def _feature_vector(**overrides) -> FeatureVector:
    base = FeatureVector(
        symbol="BTC/USDT",
        timestamp=datetime(2025, 1, 1),
        rsi=58.0,
        ema_fast=101.0,
        ema_slow=100.0,
        atr=1.5,
        momentum=0.02,
        volatility=0.01,
        order_book_imbalance=1.05,
        volume_ratio=1.30,
        funding_rate=0.0001,
        taker_ratio=1.02,
        raw={
            "breakout_distance": 0.45,
            "trend_slope": 0.0015,
            "volume_zscore": 1.10,
            "realized_vol_short": 0.009,
        },
    )
    return base.model_copy(update=overrides)


def _candles(start: float, step: float, n: int) -> list[OHLCV]:
    base = datetime(2025, 1, 1)
    candles = []
    price = start
    for i in range(n):
        candles.append(
            OHLCV(
                symbol="BTC/USDT",
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=100.0,
                timestamp=base + timedelta(minutes=i),
                is_closed=True,
            )
        )
        price += step
    return candles


class TestFactorEngine:
    def test_bullish_snapshot_from_trend_and_volume(self):
        engine = FactorEngine({"strategy": {}, "regime": {"enabled": True}})
        features = _feature_vector()
        snapshot = engine.evaluate(
            features,
            supplementary={
                "order_book_imbalance": 1.08,
                "funding_rate": 0.0001,
                "taker_ratio": 1.01,
                "open_interest": 1000.0,
                "has_funding_rate": True,
                "has_taker_ratio": True,
                "has_open_interest": True,
            },
            supplementary_history={"open_interest": [1000.0, 1005.0, 1010.0]},
            candles_15m=_candles(100.0, 0.5, 4),
            candles_1h=_candles(100.0, 1.0, 4),
            market_context={"regime": "risk_on", "score": 0.45, "breadth": 0.7},
        )
        assert snapshot.entry_score > 0.5
        assert snapshot.blocker_score < 0.4
        assert snapshot.regime == "risk_on"
        assert "market_regime" in snapshot.supporting_factors
        assert "trend_alignment" in snapshot.supporting_factors
        assert "breakout_confirmation" in snapshot.supporting_factors

    def test_perp_crowding_becomes_blocker(self):
        engine = FactorEngine({"strategy": {}})
        features = _feature_vector()
        snapshot = engine.evaluate(
            features,
            supplementary={
                "order_book_imbalance": 1.01,
                "funding_rate": 0.0010,
                "taker_ratio": 1.40,
                "open_interest": 1200.0,
                "has_funding_rate": True,
                "has_taker_ratio": True,
                "has_open_interest": True,
            },
            supplementary_history={"open_interest": [1000.0, 1100.0, 1200.0]},
        )
        perp = next(obs for obs in snapshot.observations if obs.name == "perp_crowding")
        assert perp.bias == FactorBias.BEARISH
        assert snapshot.blocker_score > 0.0

    def test_perp_crowding_stays_neutral_when_derivatives_data_is_missing(self):
        engine = FactorEngine({"strategy": {}})
        snapshot = engine.evaluate(
            _feature_vector(),
            supplementary={},
            supplementary_history={},
        )
        perp = next(obs for obs in snapshot.observations if obs.name == "perp_crowding")
        assert perp.bias == FactorBias.NEUTRAL
        assert perp.strength == 0.0
        assert perp.metadata["has_funding_rate"] is False
        assert perp.metadata["has_taker_ratio"] is False
        assert perp.metadata["has_open_interest"] is False

    def test_market_risk_off_adds_regime_blocker(self):
        engine = FactorEngine({"strategy": {}, "regime": {"enabled": True}})
        snapshot = engine.evaluate(
            _feature_vector(),
            market_context={"regime": "risk_off", "score": -0.20, "breadth": 0.2},
        )
        regime_obs = next(
            obs for obs in snapshot.observations if obs.name == "market_regime"
        )
        assert regime_obs.bias == FactorBias.BEARISH
        assert snapshot.regime == "risk_off"
        assert snapshot.blocker_score > 0.0

    def test_perp_crowding_uses_fixed_open_interest_lookback(self):
        engine = FactorEngine(
            {"strategy": {"open_interest_lookback_samples": 3}}
        )
        features = _feature_vector(funding_rate=0.0, taker_ratio=1.0)
        supplementary = {
            "order_book_imbalance": 1.0,
            "funding_rate": 0.0,
            "taker_ratio": 1.0,
            "open_interest": 130.0,
            "has_funding_rate": True,
            "has_taker_ratio": True,
            "has_open_interest": True,
        }

        short_snapshot = engine.evaluate(
            features,
            supplementary=supplementary,
            supplementary_history={"open_interest": [100.0, 100.0, 130.0]},
        )
        long_snapshot = engine.evaluate(
            features,
            supplementary=supplementary,
            supplementary_history={"open_interest": [60.0, 80.0, 100.0, 100.0, 130.0]},
        )

        short_perp = next(
            obs for obs in short_snapshot.observations if obs.name == "perp_crowding"
        )
        long_perp = next(
            obs for obs in long_snapshot.observations if obs.name == "perp_crowding"
        )

        assert short_perp.metadata["open_interest_change"] == long_perp.metadata[
            "open_interest_change"
        ]
        assert short_snapshot.blocker_score == long_snapshot.blocker_score

    def test_pullback_reentry_turns_bullish_for_trend_dip(self):
        engine = FactorEngine(
            {
                "strategy": {
                    "enable_pullback_reentry": True,
                    "pullback_min_breakout_distance": -0.60,
                    "pullback_max_breakout_distance": 0.20,
                    "pullback_min_rsi": 42.0,
                    "pullback_max_rsi": 60.0,
                    "pullback_min_momentum": -0.02,
                    "pullback_max_momentum": 0.01,
                    "pullback_min_volume_ratio": 0.80,
                    "pullback_trend_slack": 0.70,
                    "factor_weights": {
                        "pullback_reentry": 0.25,
                        "trend_alignment": 0.25,
                        "market_regime": 0.20,
                    },
                },
                "regime": {"enabled": True},
                "trend": {"min_trend_slope": 0.0006},
            }
        )
        snapshot = engine.evaluate(
            _feature_vector(
                rsi=51.0,
                momentum=-0.004,
                volume_ratio=0.92,
                raw={
                    "breakout_distance": -0.18,
                    "trend_slope": 0.0011,
                    "volume_zscore": 0.25,
                    "realized_vol_short": 0.008,
                },
            ),
            market_context={"regime": "risk_on", "score": 0.40, "breadth": 0.65},
            candles_15m=_candles(100.0, 0.3, 4),
            candles_1h=_candles(100.0, 0.6, 4),
        )
        pullback = next(
            obs for obs in snapshot.observations if obs.name == "pullback_reentry"
        )
        assert pullback.bias == FactorBias.BULLISH
        assert pullback.strength > 0.0

    def test_overextension_exit_turns_bearish_for_stretched_move(self):
        engine = FactorEngine(
            {
                "strategy": {
                    "enable_overextension_exit": True,
                    "overextension_min_breakout_distance": 0.45,
                    "overextension_min_rsi": 67.0,
                    "overextension_min_volume_zscore": -0.10,
                    "overextension_min_taker_ratio": 1.01,
                    "factor_weights": {"overextension_exit": 0.25},
                }
            }
        )
        snapshot = engine.evaluate(
            _feature_vector(
                rsi=73.0,
                taker_ratio=1.08,
                raw={
                    "breakout_distance": 0.82,
                    "trend_slope": 0.0018,
                    "volume_zscore": 0.85,
                    "realized_vol_short": 0.010,
                },
            ),
            supplementary={
                "taker_ratio": 1.08,
                "funding_rate": 0.0002,
            },
        )
        overextension = next(
            obs for obs in snapshot.observations if obs.name == "overextension_exit"
        )
        assert overextension.bias == FactorBias.BEARISH
        assert overextension.strength > 0.0

    def test_volatility_compression_entry_turns_bullish_for_quiet_trend(self):
        engine = FactorEngine(
            {
                "strategy": {
                    "enable_volatility_compression_entry": True,
                    "vol_compression_min_breakout_distance": -0.05,
                    "vol_compression_max_breakout_distance": 0.20,
                    "vol_compression_min_rsi": 50.0,
                    "vol_compression_max_rsi": 64.0,
                    "vol_compression_short_vol_ratio_max": 0.85,
                    "vol_compression_min_volume_ratio": 0.80,
                    "vol_compression_max_volume_ratio": 1.15,
                    "factor_weights": {"volatility_compression_entry": 0.18},
                },
                "trend": {"min_trend_slope": 0.0006},
            }
        )
        snapshot = engine.evaluate(
            _feature_vector(
                rsi=57.0,
                volatility=0.010,
                volume_ratio=0.96,
                raw={
                    "breakout_distance": 0.08,
                    "trend_slope": 0.0010,
                    "volume_zscore": -0.10,
                    "realized_vol_short": 0.0065,
                },
            )
        )
        obs = next(
            item
            for item in snapshot.observations
            if item.name == "volatility_compression_entry"
        )
        assert obs.bias == FactorBias.BULLISH
        assert obs.strength > 0.0

    def test_grid_reversion_entry_turns_bullish_for_shallow_dip(self):
        engine = FactorEngine(
            {
                "strategy": {
                    "enable_grid_reversion_entry": True,
                    "grid_reversion_min_breakout_distance": -0.25,
                    "grid_reversion_max_breakout_distance": 0.04,
                    "grid_reversion_target_breakout_distance": -0.08,
                    "grid_reversion_min_rsi": 45.0,
                    "grid_reversion_max_rsi": 56.0,
                    "grid_reversion_target_rsi": 50.0,
                    "grid_reversion_min_momentum": -0.006,
                    "grid_reversion_max_momentum": 0.004,
                    "grid_reversion_min_volume_ratio": 0.80,
                    "grid_reversion_max_volume_ratio": 1.20,
                    "grid_reversion_short_vol_ratio_max": 1.05,
                    "factor_weights": {"grid_reversion_entry": 0.14},
                },
                "trend": {"min_trend_slope": 0.0006},
            }
        )
        snapshot = engine.evaluate(
            _feature_vector(
                rsi=50.0,
                momentum=-0.0015,
                volume_ratio=0.92,
                raw={
                    "breakout_distance": -0.09,
                    "trend_slope": 0.0010,
                    "volume_zscore": 0.05,
                    "realized_vol_short": 0.0090,
                },
            )
        )
        obs = next(
            item for item in snapshot.observations if item.name == "grid_reversion_entry"
        )
        assert obs.bias == FactorBias.BULLISH
        assert obs.strength > 0.0
