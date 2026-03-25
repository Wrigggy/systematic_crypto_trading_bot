from __future__ import annotations

import json
import logging
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from onchain.providers.base import OnchainProvider, empty_onchain_frame, merge_onchain_frames

logger = logging.getLogger(__name__)


class CoinMetricsCommunityProvider(OnchainProvider):
    name = "coinmetrics_community"
    _base_url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    _symbol_to_asset = {
        "BTC/USDT": "btc",
        "ETH/USDT": "eth",
        "SOL/USDT": "sol",
        "BNB/USDT": "bnb",
        "LINK/USDT": "link",
        "XRP/USDT": "xrp",
    }
    _metric_map = {
        "CapMVRVCur": "onchain_mvrv_ratio",
        "AdrActCnt": "onchain_active_addresses",
        "TxCnt": "onchain_tx_count",
        "CapMrktCurUSD": "onchain_market_cap_usd",
    }

    def supports_symbol(self, symbol: str) -> bool:
        return symbol in self._symbol_to_asset

    def fetch_dataset(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        asset = self._symbol_to_asset.get(symbol)
        if asset is None:
            return empty_onchain_frame(symbol)

        frames = []
        for metric, field_name in self._metric_map.items():
            frame = self._fetch_metric_series(
                asset,
                metric,
                field_name=field_name,
                start=start,
                end=end,
            )
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return empty_onchain_frame(symbol)
        return merge_onchain_frames(symbol, frames)

    def _fetch_metric_series(
        self,
        asset: str,
        metric: str,
        *,
        field_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        params = {
            "assets": asset,
            "metrics": metric,
            "frequency": "1d",
            "page_size": 10000,
        }
        if start is not None:
            params["start_time"] = start.strftime("%Y-%m-%dT00:00:00Z")
        if end is not None:
            params["end_time"] = end.strftime("%Y-%m-%dT00:00:00Z")
        url = f"{self._base_url}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            logger.info(
                "Coin Metrics metric unavailable for asset=%s metric=%s: %s",
                asset,
                metric,
                exc,
            )
            return pd.DataFrame(columns=["timestamp", field_name])

        frame = pd.DataFrame(payload.get("data", []))
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", field_name])

        frame = frame.rename(columns={"time": "timestamp", metric: field_name})
        return frame[["timestamp", field_name]].copy()
