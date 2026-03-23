---
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

1-minute OHLCV (Open, High, Low, Close, Volume) candle data for 1 cryptocurrency trading pairs on Binance spot, covering 90 days of history.

Generated on 2026-03-17.

## Symbols

1000CHEEMS/USDT

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
