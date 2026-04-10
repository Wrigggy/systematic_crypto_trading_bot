import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime
from alpha.registry import AlphaRegistry
from core.models import OHLCV, FactorObservation, FactorSnapshot


def _make_candles(n: int = 100, symbol: str = "BTC/USDT") -> list:
    np.random.seed(42)
    base = 65000.0 + np.cumsum(np.random.randn(n) * 100)
    candles = []
    for i in range(n):
        candles.append(OHLCV(
            symbol=symbol,
            open=base[i] + np.random.randn() * 10,
            high=base[i] + abs(np.random.randn()) * 50,
            low=base[i] - abs(np.random.randn()) * 50,
            close=float(base[i]),
            volume=float(np.random.exponential(100)),
            timestamp=datetime(2026, 1, 1, i // 60, i % 60),
        ))
    return candles


def _write_alpha(tmp_path: Path, alpha_id: str, expression: str, weight: float = 1.0) -> Path:
    data = {
        "alpha_id": alpha_id,
        "version": "1.0.0",
        "type": "expression_tree",
        "expression": expression,
        "normalization": {"method": "rolling_zscore", "lookback": 20},
        "weight_hint": weight,
        "horizon": "intraday",
        "meta": {"author": "test"},
    }
    f = tmp_path / f"{alpha_id}.json"
    f.write_text(json.dumps(data))
    return f


class TestAlphaRegistry:
    def test_load_single_alpha(self, tmp_path):
        _write_alpha(tmp_path, "test_delta", "Delta($close, 5)")
        registry = AlphaRegistry(alpha_dir=tmp_path)
        assert len(registry.alphas) == 1
        assert "test_delta" in registry.alphas

    def test_load_multiple_alphas(self, tmp_path):
        _write_alpha(tmp_path, "alpha1", "Delta($close, 5)", weight=0.6)
        _write_alpha(tmp_path, "alpha2", "Mean($volume, 10)", weight=0.4)
        registry = AlphaRegistry(alpha_dir=tmp_path)
        assert len(registry.alphas) == 2

    def test_evaluate_produces_factor_snapshot(self, tmp_path):
        _write_alpha(tmp_path, "test_delta", "Delta($close, 5)")
        registry = AlphaRegistry(alpha_dir=tmp_path)
        candles = _make_candles(100)

        snapshot = registry.evaluate("BTC/USDT", candles)
        assert isinstance(snapshot, FactorSnapshot)
        assert snapshot.symbol == "BTC/USDT"
        assert len(snapshot.observations) == 1
        assert isinstance(snapshot.observations[0], FactorObservation)
        assert snapshot.observations[0].name == "test_delta"

    def test_entry_score_is_weighted(self, tmp_path):
        _write_alpha(tmp_path, "a1", "Delta($close, 5)", weight=0.7)
        _write_alpha(tmp_path, "a2", "Mean($close, 10)", weight=0.3)
        registry = AlphaRegistry(alpha_dir=tmp_path)
        candles = _make_candles(100)

        snapshot = registry.evaluate("BTC/USDT", candles)
        assert np.isfinite(snapshot.entry_score)

    def test_invalid_feature_raises_at_load(self, tmp_path):
        data = {
            "alpha_id": "bad",
            "version": "1.0.0",
            "type": "expression_tree",
            "expression": "$nonexistent",
            "normalization": {"method": "rolling_zscore", "lookback": 20},
        }
        (tmp_path / "bad.json").write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Unknown feature"):
            AlphaRegistry(alpha_dir=tmp_path)

    def test_empty_directory(self, tmp_path):
        registry = AlphaRegistry(alpha_dir=tmp_path)
        assert len(registry.alphas) == 0
