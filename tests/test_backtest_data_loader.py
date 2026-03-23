from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from backtest.data_loader import (
    iter_candle_batches,
    load_symbol_dataset,
    parquet_name_for_symbol,
)


def _write_symbol_parquet(tmp_path, symbol: str, closes: list[float]) -> None:
    base = datetime(2026, 1, 1, 0, 0)
    frame = pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(len(closes))],
            "open": closes,
            "high": [value + 0.5 for value in closes],
            "low": [value - 0.5 for value in closes],
            "close": closes,
            "volume": [10.0 + i for i in range(len(closes))],
        }
    )
    frame.to_parquet(tmp_path / parquet_name_for_symbol(symbol), index=False)


def _make_test_dir() -> Path:
    root = Path.cwd() / "backtest_test_tmp" / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_load_symbol_dataset_filters_dates():
    tmp_path = _make_test_dir()
    try:
        _write_symbol_parquet(tmp_path, "BTC/USDT", [100.0, 101.0, 102.0, 103.0])

        dataset = load_symbol_dataset(
            tmp_path,
            "BTC/USDT",
            start=datetime(2026, 1, 1, 0, 1),
            end=datetime(2026, 1, 1, 0, 2),
        )

        assert len(dataset.frame) == 2
        assert float(dataset.frame.iloc[0]["close"]) == 101.0
        assert float(dataset.frame.iloc[-1]["close"]) == 102.0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_iter_candle_batches_merges_symbols_in_timestamp_order():
    tmp_path = _make_test_dir()
    try:
        _write_symbol_parquet(tmp_path, "BTC/USDT", [100.0, 101.0, 102.0])
        _write_symbol_parquet(tmp_path, "ETH/USDT", [200.0, 201.0, 202.0])

        datasets = {
            "BTC/USDT": load_symbol_dataset(tmp_path, "BTC/USDT"),
            "ETH/USDT": load_symbol_dataset(tmp_path, "ETH/USDT"),
        }

        batches = list(iter_candle_batches(datasets))
        assert len(batches) == 3
        assert batches[0][0] == datetime(2026, 1, 1, 0, 0)
        assert {candle.symbol for candle in batches[0][1]} == {"BTC/USDT", "ETH/USDT"}
        assert all(len(candles) == 2 for _, candles in batches)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
