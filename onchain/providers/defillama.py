from __future__ import annotations

import json
import logging
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from onchain.providers.base import OnchainProvider, empty_onchain_frame, merge_onchain_frames

logger = logging.getLogger(__name__)


class DefiLlamaChainsProvider(OnchainProvider):
    name = "defillama_chains"
    _chain_map = {
        "BTC/USDT": {
            "tvl": "Bitcoin",
            "dex_volume": "Bitcoin",
            "fees": "Bitcoin",
            "stablecoins": None,
        },
        "ETH/USDT": {
            "tvl": "Ethereum",
            "dex_volume": "Ethereum",
            "fees": "Ethereum",
            "stablecoins": "Ethereum",
        },
        "SOL/USDT": {
            "tvl": "Solana",
            "dex_volume": "Solana",
            "fees": "Solana",
            "stablecoins": "Solana",
        },
        "BNB/USDT": {
            "tvl": "BSC",
            "dex_volume": "BSC",
            "fees": "BSC",
            "stablecoins": "BSC",
        },
        "XRP/USDT": {
            "tvl": None,
            "dex_volume": "XRPL",
            "fees": "XRPL",
            "stablecoins": "XRPL",
        },
    }

    def supports_symbol(self, symbol: str) -> bool:
        return symbol in self._chain_map

    def fetch_dataset(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        chains = self._chain_map.get(symbol)
        if chains is None:
            return empty_onchain_frame(symbol)

        frames = []
        if chains.get("tvl"):
            frames.append(
                self._fetch_tvl_series(chains["tvl"], start=start, end=end)
            )
        if chains.get("dex_volume"):
            frames.append(
                self._fetch_total_chart_series(
                    f"https://api.llama.fi/overview/dexs/{quote(chains['dex_volume'])}"
                    "?excludeTotalDataChartBreakdown=true&dataType=dailyVolume",
                    "onchain_chain_dex_volume_usd",
                    start=start,
                    end=end,
                )
            )
        if chains.get("fees"):
            frames.append(
                self._fetch_total_chart_series(
                    f"https://api.llama.fi/overview/fees/{quote(chains['fees'])}"
                    "?excludeTotalDataChartBreakdown=true&dataType=dailyFees",
                    "onchain_chain_fees_usd",
                    start=start,
                    end=end,
                )
            )
        if chains.get("stablecoins"):
            frames.append(
                self._fetch_stablecoin_series(
                    chains["stablecoins"],
                    start=start,
                    end=end,
                )
            )

        usable_frames = [frame for frame in frames if not frame.empty]
        if not usable_frames:
            return empty_onchain_frame(symbol)
        return merge_onchain_frames(symbol, usable_frames)

    def _request_json(self, url: str) -> object:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))

    def _fetch_tvl_series(
        self,
        chain: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        url = f"https://api.llama.fi/v2/historicalChainTvl/{quote(chain)}"
        try:
            payload = self._request_json(url)
        except HTTPError as exc:
            logger.info("DefiLlama TVL unavailable for chain=%s: %s", chain, exc)
            return pd.DataFrame(columns=["timestamp", "onchain_chain_tvl_usd"])
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "onchain_chain_tvl_usd"])
        frame = frame.rename(columns={"date": "timestamp", "tvl": "onchain_chain_tvl_usd"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        return self._trim(frame[["timestamp", "onchain_chain_tvl_usd"]], start, end)

    def _fetch_total_chart_series(
        self,
        url: str,
        field_name: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        try:
            payload = self._request_json(url)
        except HTTPError as exc:
            logger.info("DefiLlama chart unavailable for %s: %s", url, exc)
            return pd.DataFrame(columns=["timestamp", field_name])

        frame = pd.DataFrame(payload.get("totalDataChart", []), columns=["timestamp", field_name])
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", field_name])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        return self._trim(frame, start, end)

    def _fetch_stablecoin_series(
        self,
        chain: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        url = f"https://stablecoins.llama.fi/stablecoincharts/{quote(chain)}"
        try:
            payload = self._request_json(url)
        except HTTPError as exc:
            logger.info("DefiLlama stablecoins unavailable for chain=%s: %s", chain, exc)
            return pd.DataFrame(
                columns=["timestamp", "onchain_chain_stablecoin_supply_usd"]
            )

        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(
                columns=["timestamp", "onchain_chain_stablecoin_supply_usd"]
            )
        frame["timestamp"] = pd.to_datetime(
            pd.to_numeric(frame["date"], errors="coerce"),
            unit="s",
            utc=True,
        )
        frame["onchain_chain_stablecoin_supply_usd"] = frame["totalCirculatingUSD"].apply(
            lambda value: value.get("peggedUSD") if isinstance(value, dict) else None
        )
        return self._trim(
            frame[["timestamp", "onchain_chain_stablecoin_supply_usd"]],
            start,
            end,
        )

    def _trim(
        self,
        frame: pd.DataFrame,
        start: datetime | None,
        end: datetime | None,
    ) -> pd.DataFrame:
        trimmed = frame.copy()
        start_ts = self._utc_timestamp(start)
        end_ts = self._utc_timestamp(end)
        if start_ts is not None:
            trimmed = trimmed.loc[trimmed["timestamp"] >= start_ts]
        if end_ts is not None:
            trimmed = trimmed.loc[trimmed["timestamp"] <= end_ts]
        return trimmed.reset_index(drop=True)

    def _utc_timestamp(self, value: datetime | None) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
