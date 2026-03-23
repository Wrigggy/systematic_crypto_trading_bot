"""Coin list and Binance symbol mapping.

Source: CoinList.txt (66 base symbols from competition).
65 tradeable on Binance spot (OMNI has no spot pair).
Hardcoded so the pipeline has no external file dependency at runtime.
"""

BASE_SYMBOLS: list[str] = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "TAO", "PEPE", "ADA", "ZEC",
    "PAXG", "LINK", "SUI", "TRUMP", "AVAX", "FET", "DOT", "LTC", "NEAR", "TRX",
    "UNI", "BONK", "WIF", "ENA", "PUMP", "WLD", "PENGU", "EIGEN", "HBAR", "ASTER",
    "AAVE", "APT", "FIL", "VIRTUAL", "CFX", "SHIB", "XPL", "ICP", "XLM", "S",
    "CAKE", "ARB", "WLFI", "CRV", "FLOKI", "ONDO", "ZEN", "SEI", "TON", "POL",
    "PENDLE", "TUT", "BIO", "LINEA", "PLUME", "1000CHEEMS", "FORM", "AVNT",
    "LISTA", "OPEN", "SOMI", "HEMI", "EDEN", "MIRA", "BMT",
]

BINANCE_PAIRS: list[str] = [f"{s}/USDT" for s in BASE_SYMBOLS]

# Mapping from base symbol to Binance pair
SYMBOL_MAP: dict[str, str] = {s: f"{s}/USDT" for s in BASE_SYMBOLS}
