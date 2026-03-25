from __future__ import annotations

from onchain.providers.base import OnchainProvider
from onchain.providers.coinmetrics import CoinMetricsCommunityProvider
from onchain.providers.defillama import DefiLlamaChainsProvider


def default_providers() -> tuple[OnchainProvider, ...]:
    return (
        CoinMetricsCommunityProvider(),
        DefiLlamaChainsProvider(),
    )


__all__ = [
    "CoinMetricsCommunityProvider",
    "DefiLlamaChainsProvider",
    "OnchainProvider",
    "default_providers",
]
