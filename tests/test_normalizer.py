import numpy as np
import pytest
from alpha.normalizer import RollingZScoreNormalizer, CrossSectionalNormalizer


class TestRollingZScoreNormalizer:
    def test_basic_normalization(self):
        norm = RollingZScoreNormalizer(lookback=10)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        result = norm.update("BTC/USDT", values[-1], values)
        assert np.isfinite(result)

    def test_warmup_returns_zero(self):
        norm = RollingZScoreNormalizer(lookback=20)
        values = np.array([1.0, 2.0, 3.0])
        result = norm.update("BTC/USDT", values[-1], values)
        assert result == 0.0

    def test_constant_values_return_zero(self):
        norm = RollingZScoreNormalizer(lookback=5)
        values = np.full(10, 5.0)
        result = norm.update("BTC/USDT", values[-1], values)
        assert result == 0.0

    def test_separate_symbols(self):
        norm = RollingZScoreNormalizer(lookback=5)
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        v2 = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
        z1 = norm.update("BTC/USDT", v1[-1], v1)
        z2 = norm.update("ETH/USDT", v2[-1], v2)
        assert z1 > 0
        assert z2 < 0


class TestCrossSectionalNormalizer:
    def test_basic_normalization(self):
        norm = CrossSectionalNormalizer()
        scores = {"BTC/USDT": 10.0, "ETH/USDT": 5.0, "SOL/USDT": 0.0}
        result = norm.normalize(scores)
        assert result["BTC/USDT"] > 0
        assert result["SOL/USDT"] < 0

    def test_single_symbol_returns_zero(self):
        norm = CrossSectionalNormalizer()
        scores = {"BTC/USDT": 42.0}
        result = norm.normalize(scores)
        assert result["BTC/USDT"] == 0.0

    def test_equal_values_return_zero(self):
        norm = CrossSectionalNormalizer()
        scores = {"BTC/USDT": 5.0, "ETH/USDT": 5.0}
        result = norm.normalize(scores)
        assert result["BTC/USDT"] == 0.0
        assert result["ETH/USDT"] == 0.0
