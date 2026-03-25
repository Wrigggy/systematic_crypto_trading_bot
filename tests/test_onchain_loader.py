from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest.onchain_loader import load_onchain_dataset


def _make_test_dir() -> Path:
    root = Path.cwd() / "onchain_loader_test_tmp" / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_load_onchain_dataset_merges_multiple_provider_caches():
    tmp_path = _make_test_dir()
    try:
        coinmetrics_root = tmp_path / "coinmetrics_community"
        defillama_root = tmp_path / "defillama_chains"
        coinmetrics_root.mkdir(parents=True, exist_ok=True)
        defillama_root.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
                "symbol": ["SOL/USDT", "SOL/USDT"],
                "onchain_mvrv_ratio": [0.95, 0.93],
                "onchain_active_addresses": [100.0, 110.0],
                "onchain_tx_count": [200.0, 230.0],
                "onchain_market_cap_usd": [1000.0, 1100.0],
                "onchain_nvt_proxy": [5.0, 4.78],
            }
        ).to_parquet(coinmetrics_root / "SOL_USDT_onchain.parquet", index=False)

        pd.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
                "symbol": ["SOL/USDT", "SOL/USDT"],
                "onchain_chain_tvl_usd": [500.0, 560.0],
                "onchain_chain_stablecoin_supply_usd": [300.0, 345.0],
                "onchain_chain_dex_volume_usd": [250.0, 310.0],
                "onchain_chain_fees_usd": [12.0, 15.0],
            }
        ).to_parquet(defillama_root / "SOL_USDT_onchain.parquet", index=False)

        dataset = load_onchain_dataset(
            "SOL/USDT",
            cache_root=tmp_path,
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 2),
        )

        assert dataset is not None
        assert len(dataset.frame) == 2
        assert float(dataset.frame.iloc[-1]["onchain_mvrv_ratio"]) == 0.93
        assert float(dataset.frame.iloc[-1]["onchain_chain_tvl_usd"]) == 560.0
        assert float(dataset.frame.iloc[-1]["onchain_chain_fees_usd"]) == 15.0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
