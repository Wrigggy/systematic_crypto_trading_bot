from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import pandas as pd

from core.models import OHLCV

from backtest.config import PROJECT_ROOT


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data_pipeline" / "output" / "data"


@dataclass(frozen=True)
class SymbolDataset:
    symbol: str
    path: Path
    frame: pd.DataFrame


def parquet_name_for_symbol(symbol: str) -> str:
    return symbol.replace("/", "_") + ".parquet"


def list_available_symbols(data_root: Path) -> list[str]:
    if not data_root.exists():
        return []
    symbols = []
    for path in sorted(data_root.glob("*.parquet")):
        stem = path.stem
        if "_" not in stem:
            continue
        base, quote = stem.rsplit("_", 1)
        symbols.append(f"{base}/{quote}")
    return symbols


def load_symbol_dataset(
    data_root: Path,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> SymbolDataset:
    path = data_root / parquet_name_for_symbol(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Historical parquet not found for {symbol}: {path}")

    frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = frame.copy()
    frame["timestamp"] = (
        pd.to_datetime(frame["timestamp"], utc=True)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    frame = frame.sort_values("timestamp", kind="stable")
    if start is not None:
        frame = frame.loc[frame["timestamp"] >= start]
    if end is not None:
        frame = frame.loc[frame["timestamp"] <= end]
    if frame.empty:
        raise ValueError(f"No rows left for {symbol} after applying the date range")

    frame = frame.reset_index(drop=True)
    return SymbolDataset(symbol=symbol, path=path, frame=frame)


def load_datasets(
    data_root: Path,
    symbols: Sequence[str],
    start: datetime | None = None,
    end: datetime | None = None,
) -> Dict[str, SymbolDataset]:
    return {
        symbol: load_symbol_dataset(data_root, symbol, start=start, end=end)
        for symbol in symbols
    }


def iter_candle_batches(
    datasets: Dict[str, SymbolDataset],
) -> Iterator[tuple[datetime, List[OHLCV]]]:
    """Yield all same-timestamp candles across symbols in chronological order."""
    heap: list[tuple[datetime, str, int]] = []
    for symbol, dataset in datasets.items():
        if dataset.frame.empty:
            continue
        heap.append((dataset.frame.iloc[0]["timestamp"].to_pydatetime(), symbol, 0))
    heapq.heapify(heap)

    while heap:
        ts, symbol, row_idx = heapq.heappop(heap)
        batch_meta = [(ts, symbol, row_idx)]
        while heap and heap[0][0] == ts:
            batch_meta.append(heapq.heappop(heap))

        candles: list[OHLCV] = []
        for _, batch_symbol, batch_row_idx in batch_meta:
            dataset = datasets[batch_symbol]
            row = dataset.frame.iloc[batch_row_idx]
            candles.append(
                OHLCV(
                    symbol=batch_symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    timestamp=ts,
                    is_closed=True,
                )
            )

            next_idx = batch_row_idx + 1
            if next_idx < len(dataset.frame):
                next_ts = dataset.frame.iloc[next_idx]["timestamp"].to_pydatetime()
                heapq.heappush(heap, (next_ts, batch_symbol, next_idx))

        yield ts, candles
