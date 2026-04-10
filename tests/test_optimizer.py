import pytest
from datetime import datetime
from core.models import FactorSnapshot
from strategy.optimizer import PortfolioOptimizer


def _snap(symbol: str, entry_score: float) -> FactorSnapshot:
    return FactorSnapshot(
        symbol=symbol,
        timestamp=datetime(2026, 1, 1),
        entry_score=entry_score,
        exit_score=entry_score,
        confidence=abs(entry_score),
    )


class TestEqualWeight:
    def test_two_symbols(self):
        opt = PortfolioOptimizer(mode="equal_weight", max_positions=2)
        snapshots = [_snap("BTC/USDT", 0.8), _snap("ETH/USDT", 0.6)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] == pytest.approx(0.5)
        assert weights["ETH/USDT"] == pytest.approx(0.5)

    def test_respects_max_positions(self):
        opt = PortfolioOptimizer(mode="equal_weight", max_positions=1)
        snapshots = [_snap("BTC/USDT", 0.8), _snap("ETH/USDT", 0.6)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] == pytest.approx(1.0)
        assert "ETH/USDT" not in weights or weights["ETH/USDT"] == 0.0

    def test_filters_negative_scores(self):
        opt = PortfolioOptimizer(mode="equal_weight", max_positions=2)
        snapshots = [_snap("BTC/USDT", 0.8), _snap("ETH/USDT", -0.3)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] == pytest.approx(1.0)


class TestScoreTilted:
    def test_higher_score_gets_more_weight(self):
        opt = PortfolioOptimizer(mode="score_tilted", max_positions=2, temperature=0.8)
        snapshots = [_snap("BTC/USDT", 0.9), _snap("ETH/USDT", 0.3)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] > weights["ETH/USDT"]

    def test_max_single_weight_cap(self):
        opt = PortfolioOptimizer(
            mode="score_tilted", max_positions=2,
            temperature=0.8, max_single_weight=0.6,
        )
        snapshots = [_snap("BTC/USDT", 0.99), _snap("ETH/USDT", 0.01)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] <= 0.6 + 1e-9

    def test_weights_sum_to_one(self):
        opt = PortfolioOptimizer(mode="score_tilted", max_positions=3, temperature=0.8)
        snapshots = [_snap("BTC", 0.8), _snap("ETH", 0.6), _snap("SOL", 0.4)]
        weights = opt.allocate(snapshots)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestEmptyInput:
    def test_no_snapshots(self):
        opt = PortfolioOptimizer(mode="equal_weight", max_positions=2)
        weights = opt.allocate([])
        assert weights == {}
