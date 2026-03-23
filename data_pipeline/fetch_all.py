"""Fetch, clean, and export 1m OHLCV data for all competition symbols.

Usage:
    # Quick test with 2 symbols
    uv run python -m data_pipeline.fetch_all --days 90 --symbols BTC/USDT ETH/USDT

    # Full run (all 66 symbols)
    uv run python -m data_pipeline.fetch_all --days 90

Pipeline:
    1. Async fetch via ccxt (Binance spot, 1m candles)
    2. Clean: dedup, sort, validate, gap-fill
    3. Export: one parquet per symbol in output/data/
    4. Generate HF dataset card in output/README.md
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = OUTPUT_DIR / "data"


# ── Fetch ────────────────────────────────────────────────────────────────────

async def fetch_symbol(
    exchange,
    symbol: str,
    since_ms: int,
    until_ms: int,
    semaphore: asyncio.Semaphore,
) -> list[list]:
    """Fetch all 1m candles for a single symbol between since_ms and until_ms."""
    all_candles: list[list] = []
    cursor = since_ms

    while cursor < until_ms:
        async with semaphore:
            try:
                batch = await exchange.fetch_ohlcv(
                    symbol, "1m", since=cursor, limit=1000
                )
            except Exception as e:
                logger.warning("%s: fetch error at %d — %s", symbol, cursor, e)
                break

        if not batch:
            break

        all_candles.extend(batch)
        cursor = batch[-1][0] + 60_000  # next minute

    return all_candles


async def fetch_all_symbols(
    symbols: list[str],
    days: int,
    concurrency: int = 10,
    proxy: str | None = None,
) -> dict[str, list[list]]:
    """Fetch OHLCV for all symbols concurrently with rate limiting."""
    import ccxt.async_support as ccxt_async
    import aiohttp

    # Windows aiodns bug workaround: use ThreadedResolver instead of c-ares
    connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
    session = aiohttp.ClientSession(connector=connector)

    exchange_config = {
        "enableRateLimit": True,
        "session": session,
    }
    if proxy:
        exchange_config["aiohttp_proxy"] = proxy
        exchange_config["proxies"] = {"http": proxy, "https": proxy}
        logger.info("Using proxy: %s", proxy)

    exchange = ccxt_async.binance(exchange_config)
    semaphore = asyncio.Semaphore(concurrency)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    results: dict[str, list[list]] = {}

    async def _fetch_one(sym: str) -> None:
        # Resume support: check if parquet already exists and is complete
        parquet_path = DATA_DIR / f"{sym.replace('/', '_')}.parquet"
        effective_since = since_ms

        if parquet_path.exists():
            try:
                existing = pd.read_parquet(parquet_path)
                last_ts = int(existing["timestamp"].max().timestamp() * 1000)
                expected_end = now_ms - 120_000  # 2 min buffer
                if last_ts >= expected_end:
                    logger.info("%s: already complete (%d rows), skipping", sym, len(existing))
                    return
                effective_since = last_ts + 60_000
                logger.info("%s: resuming from %s", sym, datetime.fromtimestamp(effective_since / 1000, tz=timezone.utc))
            except Exception:
                pass  # corrupted file, re-fetch

        t0 = time.time()
        candles = await fetch_symbol(exchange, sym, effective_since, now_ms, semaphore)
        dt = time.time() - t0

        if candles:
            results[sym] = candles
            logger.info("%s: fetched %d candles in %.1fs", sym, len(candles), dt)
        else:
            logger.warning("%s: no data returned (may not exist on Binance spot)", sym)

    tasks = [_fetch_one(sym) for sym in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)

    try:
        await exchange.close()
    except Exception:
        pass
    try:
        await session.close()
    except Exception:
        pass

    return results


# ── Clean ────────────────────────────────────────────────────────────────────

def clean_ohlcv(raw: list[list], symbol: str) -> pd.DataFrame:
    """Clean raw ccxt OHLCV data into a validated DataFrame."""
    df = pd.DataFrame(raw, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])

    # Dedup and sort
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Drop dead candles (all OHLC identical AND volume == 0)
    dead = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
        & (df["volume"] == 0)
    )
    n_dead = dead.sum()
    if n_dead > 0:
        logger.info("%s: dropping %d dead candles", symbol, n_dead)
        df = df[~dead].reset_index(drop=True)

    # Validate: drop rows with NaN or non-positive prices
    price_cols = ["open", "high", "low", "close"]
    invalid = df[price_cols].isna().any(axis=1) | (df[price_cols] <= 0).any(axis=1) | (df["volume"] < 0)
    n_invalid = invalid.sum()
    if n_invalid > 0:
        logger.info("%s: dropping %d invalid rows", symbol, n_invalid)
        df = df[~invalid].reset_index(drop=True)

    # Gap detection and small gap fill (< 5 minutes)
    if len(df) > 1:
        full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1min", tz="UTC")
        missing = full_range.difference(df["timestamp"])
        if len(missing) > 0:
            # Find consecutive gap runs
            gap_groups = np.split(
                np.arange(len(missing)),
                np.where(np.diff(missing.astype(np.int64) // 10**9) > 60)[0] + 1,
            )
            small_gaps = sum(1 for g in gap_groups if len(g) <= 5)
            large_gaps = sum(1 for g in gap_groups if len(g) > 5)

            if large_gaps > 0:
                large_sizes = [len(g) for g in gap_groups if len(g) > 5]
                logger.warning(
                    "%s: %d large gaps (>5 min), sizes: %s",
                    symbol, large_gaps, large_sizes[:10],
                )

            # Reindex and forward-fill small gaps
            df = df.set_index("timestamp").reindex(full_range).rename_axis("timestamp")
            # Only ffill up to 5 consecutive NaNs
            df = df.ffill(limit=5)
            # Drop any remaining NaN rows (from large gaps)
            df = df.dropna().reset_index()

            filled = len(df) - (len(full_range) - len(missing))
            if filled > 0:
                logger.info("%s: forward-filled %d small gap candles", symbol, filled)

    # Add symbol column
    df["symbol"] = symbol

    # Ensure correct column order and types
    df = df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float64)

    return df


# ── Export ───────────────────────────────────────────────────────────────────

def export_parquet(df: pd.DataFrame, symbol: str) -> Path:
    """Write a single symbol's cleaned data to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = symbol.replace("/", "_") + ".parquet"
    path = DATA_DIR / filename
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def generate_dataset_card(symbols_exported: list[str], days: int) -> None:
    """Generate a HuggingFace dataset card README.md."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    card = f"""---
license: cc-by-4.0
task_categories:
  - time-series-forecasting
tags:
  - finance
  - crypto
  - ohlcv
  - binance
  - 1-minute
  - trading
size_categories:
  - 1M<n<10M
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/*.parquet"
---

# Crypto OHLCV 1-Minute Dataset

## Description

1-minute OHLCV (Open, High, Low, Close, Volume) candle data for {len(symbols_exported)} cryptocurrency trading pairs on Binance spot, covering {days} days of history.

Generated on {now}.

## Symbols

{', '.join(sorted(symbols_exported))}

## Schema

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime64[ns, UTC] | Candle open time |
| open | float64 | Opening price (USDT) |
| high | float64 | Highest price in interval |
| low | float64 | Lowest price in interval |
| close | float64 | Closing price (USDT) |
| volume | float64 | Base asset volume |
| symbol | string | Trading pair (e.g. BTC/USDT) |

## Usage

```python
from datasets import load_dataset

# Load all symbols
ds = load_dataset("parquet", data_dir="data")

# Load specific symbol
import pandas as pd
df = pd.read_parquet("data/BTC_USDT.parquet")
```

## Cleaning Applied

- Duplicates removed by timestamp
- Dead candles dropped (all OHLC identical with zero volume)
- Invalid rows removed (NaN, non-positive prices, negative volume)
- Small gaps (< 5 consecutive minutes) forward-filled
- Large gaps logged but not filled
"""
    (OUTPUT_DIR / "README.md").write_text(card)
    logger.info("Dataset card written to %s", OUTPUT_DIR / "README.md")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fetch & clean OHLCV data for HF upload")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--symbols", nargs="+", default=None, help="Specific symbols (e.g. BTC/USDT ETH/USDT). Default: all 66")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP/SOCKS5 proxy (e.g. http://127.0.0.1:7890)")
    args = parser.parse_args()

    if args.symbols is None:
        from data_pipeline.symbols import BINANCE_PAIRS
        symbols = BINANCE_PAIRS
    else:
        symbols = args.symbols

    # Auto-detect proxy from environment if not specified
    proxy = args.proxy
    if not proxy:
        import os
        proxy = (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
                 or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"))

    logger.info("Fetching %d symbols, %d days of 1m candles", len(symbols), args.days)

    # Fetch
    t0 = time.time()
    raw_data = asyncio.run(fetch_all_symbols(symbols, args.days, args.concurrency, proxy))
    logger.info("Fetch complete: %d/%d symbols in %.1fs", len(raw_data), len(symbols), time.time() - t0)

    # Clean & export (merge with existing parquet on resume)
    exported: list[str] = []
    for symbol, raw in raw_data.items():
        df_new = clean_ohlcv(raw, symbol)
        if len(df_new) == 0:
            logger.warning("%s: no valid data after cleaning, skipping", symbol)
            continue

        # Merge with existing parquet if resuming
        parquet_path = DATA_DIR / f"{symbol.replace('/', '_')}.parquet"
        if parquet_path.exists():
            try:
                df_existing = pd.read_parquet(parquet_path)
                df_new = pd.concat([df_existing, df_new], ignore_index=True)
                df_new = df_new.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            except Exception:
                pass  # corrupted existing file, just use new data

        path = export_parquet(df_new, symbol)
        exported.append(symbol)
        logger.info("%s: exported %d rows to %s", symbol, len(df_new), path)

    # Also account for symbols that were already complete (skipped during fetch)
    for sym in symbols:
        if sym not in raw_data:
            parquet_path = DATA_DIR / f"{sym.replace('/', '_')}.parquet"
            if parquet_path.exists():
                exported.append(sym)

    exported = sorted(set(exported))

    # Dataset card
    generate_dataset_card(exported, args.days)

    logger.info("Done! %d symbols exported to %s", len(exported), DATA_DIR)


if __name__ == "__main__":
    main()
