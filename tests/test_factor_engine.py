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
        engine = FactorEngine({"strategy": {}})
        features = _feature_vector()
        snapshot = engine.evaluate(
            features,
            supplementary={
                "order_book_imbalance": 1.08,
                "funding_rate": 0.0001,
                "taker_ratio": 1.01,
                "open_interest": 1000.0,
            },
            supplementary_history={"open_interest": [1000.0, 1005.0, 1010.0]},
            candles_15m=_candles(100.0, 0.5, 4),
            candles_1h=_candles(100.0, 1.0, 4),
        )
        assert snapshot.entry_score > 0.5
        assert snapshot.blocker_score < 0.4
        assert "trend_alignment" in snapshot.supporting_factors

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
            },
            supplementary_history={"open_interest": [1000.0, 1100.0, 1200.0]},
        )
        perp = next(obs for obs in snapshot.observations if obs.name == "perp_crowding")
        assert perp.bias == FactorBias.BEARISH
        assert snapshot.blocker_score > 0.0
