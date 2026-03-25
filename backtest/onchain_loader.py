from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

DEFAULT_ONCHAIN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline"
    / "output"
    / "onchain"
)

from onchain.providers import default_providers
from onchain.providers.base import (
    OnchainProvider,
    merge_onchain_frames,
    normalize_onchain_frame,
)


@dataclass(frozen=True)
class OnchainDataset:
    symbol: str
    path: Path
    frame: pd.DataFrame


def _resolve_providers(
    providers: Iterable[str | OnchainProvider] | None,
) -> tuple[OnchainProvider, ...]:
    registry = {provider.name: provider for provider in default_providers()}
    if providers is None:
        return tuple(registry.values())

    resolved: list[OnchainProvider] = []
    for provider in providers:
        if isinstance(provider, str):
            if provider not in registry:
                raise ValueError(f"Unknown on-chain provider: {provider}")
            resolved.append(registry[provider])
            continue
        resolved.append(provider)
    return tuple(resolved)


def ensure_onchain_cache(
    symbols: list[str],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    cache_root: Path = DEFAULT_ONCHAIN_ROOT,
    refresh: bool = False,
    providers: Iterable[str | OnchainProvider] | None = None,
) -> None:
    resolved_providers = _resolve_providers(providers)
    cache_root.mkdir(parents=True, exist_ok=True)
    for provider in resolved_providers:
        provider.cache_root(cache_root).mkdir(parents=True, exist_ok=True)
        for symbol in symbols:
            if not provider.supports_symbol(symbol):
                continue
            path = provider.cache_path(cache_root, symbol)
            if path.exists() and not refresh:
                continue
            frame = provider.fetch_dataset(symbol, start=start, end=end)
            if frame.empty:
                continue
            normalize_onchain_frame(
                frame,
                symbol=symbol,
                start=start,
                end=end,
            ).to_parquet(path, index=False)


def load_onchain_dataset(
    symbol: str,
    *,
    cache_root: Path = DEFAULT_ONCHAIN_ROOT,
    start: datetime | None = None,
    end: datetime | None = None,
    providers: Iterable[str | OnchainProvider] | None = None,
) -> Optional[OnchainDataset]:
    lookback_start = start - timedelta(days=40) if start is not None else None
    frames = []
    paths = []
    for provider in _resolve_providers(providers):
        if not provider.supports_symbol(symbol):
            continue
        path = provider.cache_path(cache_root, symbol)
        if not path.exists():
            continue
        frame = provider.load_cached_dataset(
            cache_root,
            symbol,
            start=lookback_start,
            end=end,
        )
        if frame.empty:
            continue
        frames.append(frame)
        paths.append(path)

    if not frames:
        return None

    merged = merge_onchain_frames(symbol, frames)
    if lookback_start is not None:
        merged = merged.loc[merged["timestamp"] >= lookback_start]
    if end is not None:
        merged = merged.loc[merged["timestamp"] <= end]
    if merged.empty:
        return None

    primary_path = paths[0] if paths else cache_root
    return OnchainDataset(symbol=symbol, path=primary_path, frame=merged.reset_index(drop=True))


def load_onchain_datasets(
    symbols: list[str],
    *,
    cache_root: Path = DEFAULT_ONCHAIN_ROOT,
    start: datetime | None = None,
    end: datetime | None = None,
    providers: Iterable[str | OnchainProvider] | None = None,
) -> Dict[str, OnchainDataset]:
    datasets: Dict[str, OnchainDataset] = {}
    for symbol in symbols:
        dataset = load_onchain_dataset(
            symbol,
            cache_root=cache_root,
            start=start,
            end=end,
            providers=providers,
        )
        if dataset is not None:
            datasets[symbol] = dataset
    return datasets
