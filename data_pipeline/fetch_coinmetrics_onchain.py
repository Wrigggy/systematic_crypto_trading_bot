from __future__ import annotations

import argparse
from datetime import datetime

from backtest.onchain_loader import DEFAULT_ONCHAIN_ROOT, ensure_onchain_cache


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch free Coin Metrics community on-chain sidecar data"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols like BTC/USDT ETH/USDT SOL/USDT",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_onchain_cache(
        args.symbols,
        start=parse_date(args.start),
        end=parse_date(args.end),
        cache_root=DEFAULT_ONCHAIN_ROOT,
        refresh=args.refresh,
        providers=["coinmetrics_community"],
    )
    print(f"On-chain cache updated under {DEFAULT_ONCHAIN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
