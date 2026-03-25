from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from onchain.schema import ONCHAIN_METRIC_KEYS


class OnchainProvider(ABC):
    name: str

    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def fetch_dataset(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def cache_root(self, base_root: Path) -> Path:
        return base_root / self.name

    def cache_path(self, base_root: Path, symbol: str) -> Path:
        return self.cache_root(base_root) / self.dataset_name(symbol)

    def dataset_name(self, symbol: str) -> str:
        return symbol.replace("/", "_") + "_onchain.parquet"

    def load_cached_dataset(
        self,
        base_root: Path,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        path = self.cache_path(base_root, symbol)
        if not path.exists():
            return empty_onchain_frame(symbol)
        frame = pd.read_parquet(path)
        return normalize_onchain_frame(frame, symbol=symbol, start=start, end=end)


def empty_onchain_frame(symbol: str | None = None) -> pd.DataFrame:
    columns = ["timestamp", "symbol", *ONCHAIN_METRIC_KEYS]
    frame = pd.DataFrame(columns=columns)
    if symbol is not None and "symbol" in frame.columns:
        frame["symbol"] = symbol
    return frame


def normalize_onchain_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return empty_onchain_frame(symbol)

    normalized = frame.copy()
    if "timestamp" not in normalized.columns:
        raise ValueError("On-chain frame missing required timestamp column")
    normalized["timestamp"] = (
        pd.to_datetime(normalized["timestamp"], utc=True)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    normalized["symbol"] = symbol

    for key in ONCHAIN_METRIC_KEYS:
        if key not in normalized.columns:
            normalized[key] = pd.NA
        normalized[key] = pd.to_numeric(normalized[key], errors="coerce")

    normalized = normalized[["timestamp", "symbol", *ONCHAIN_METRIC_KEYS]].copy()
    normalized = normalized.sort_values("timestamp", kind="stable")
    if start is not None:
        normalized = normalized.loc[normalized["timestamp"] >= start]
    if end is not None:
        normalized = normalized.loc[normalized["timestamp"] <= end]
    return normalized.reset_index(drop=True)


def merge_onchain_frames(
    symbol: str,
    frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    normalized_frames = [
        frame
        for frame in (normalize_onchain_frame(item, symbol=symbol) for item in frames)
        if not frame.empty
    ]
    if not normalized_frames:
        return empty_onchain_frame(symbol)

    merged_index = normalized_frames[0][["timestamp"]].copy()
    for frame in normalized_frames[1:]:
        merged_index = merged_index.merge(frame[["timestamp"]], on="timestamp", how="outer")
    merged_index = merged_index.sort_values("timestamp", kind="stable").reset_index(drop=True)
    merged = merged_index.copy()
    merged["symbol"] = symbol

    for key in ONCHAIN_METRIC_KEYS:
        merged[key] = pd.NA

    for frame in normalized_frames:
        indexed = frame.set_index("timestamp")
        for key in ONCHAIN_METRIC_KEYS:
            series = indexed[key]
            if series.notna().any():
                merged[key] = merged[key].combine_first(merged["timestamp"].map(series))

    if merged["onchain_nvt_proxy"].isna().all():
        market_cap = pd.to_numeric(merged["onchain_market_cap_usd"], errors="coerce")
        tx_count = pd.to_numeric(merged["onchain_tx_count"], errors="coerce")
        denom = tx_count.clip(lower=1.0)
        derived = market_cap / denom
        merged["onchain_nvt_proxy"] = merged["onchain_nvt_proxy"].combine_first(derived)

    return normalize_onchain_frame(merged, symbol=symbol)
