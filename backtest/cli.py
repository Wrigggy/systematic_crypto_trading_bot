from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from backtest.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, build_runtime_config, parse_date
from backtest.data_loader import DEFAULT_DATA_ROOT, list_available_symbols, load_datasets
from backtest.runner import BacktestRunner


DEFAULT_REPORT_DIR = PROJECT_ROOT / "backtest_reports" / "runs"


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Historical backtest for the local trading strategy")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to project config")
    parser.add_argument("--profile", default=None, help="Optional strategy profile override")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory containing OHLCV parquet files",
    )
    parser.add_argument("--start", default=None, help="Inclusive start date, e.g. 2026-02-01")
    parser.add_argument("--end", default=None, help="Inclusive end date, e.g. 2026-03-17")
    parser.add_argument("--symbols", nargs="*", default=None, help="Explicit symbol list, e.g. BTC/USDT ETH/USDT")
    parser.add_argument("--initial-capital", type=float, default=None, help="Override initial cash")
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for summary.json, equity_curve.csv, and closed_trades.csv",
    )
    parser.add_argument("--name", default=None, help="Optional run name prefix")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    return parser


async def _run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    start = parse_date(args.start)
    end = parse_date(args.end)

    available = set(list_available_symbols(data_root))
    if not available:
        raise SystemExit(f"No parquet files found under {data_root}")

    config_symbols = args.symbols
    if not config_symbols:
        raw_config = build_runtime_config(config_path, profile=args.profile)
        config_symbols = [symbol for symbol in raw_config.get("symbols", []) if symbol in available]

    missing = sorted(symbol for symbol in config_symbols if symbol not in available)
    symbols = [symbol for symbol in config_symbols if symbol in available]
    if missing:
        print(f"Skipping symbols without data: {', '.join(missing)}")
    if not symbols:
        raise SystemExit("No symbols with available historical data were selected")

    config = build_runtime_config(
        config_path,
        profile=args.profile,
        symbols=symbols,
        initial_capital=args.initial_capital,
    )
    datasets = load_datasets(data_root, symbols, start=start, end=end)

    run_prefix = args.name or config.get("strategy", {}).get("profile", "backtest")
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_prefix}_{now}"
    runner = BacktestRunner(config, Path(args.report_dir).expanduser().resolve())
    result = await runner.run(datasets, run_id=run_id)

    print(f"Run ID: {result.run_id}")
    print(f"Profile: {result.strategy_profile}")
    print(f"Symbols: {len(result.symbols)}")
    print(f"Period: {result.start.isoformat()} -> {result.end.isoformat()}")
    print(f"Final NAV: {result.final_nav:.2f}")
    print(f"Return: {result.total_return_pct:.2f}%")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe: {result.sharpe_ratio:.3f}")
    print(f"Sortino: {result.sortino_ratio:.3f}")
    print(f"Calmar: {result.calmar_ratio:.3f}")
    print(f"Closed Trades: {result.trade_count}")
    print(f"Win Rate: {result.win_rate * 100:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.3f}")
    print(f"Open Positions: {result.open_positions}")
    print(f"Artifacts: {(Path(args.report_dir).expanduser().resolve() / result.run_id)}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    return asyncio.run(_run(args))
