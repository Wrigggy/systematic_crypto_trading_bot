from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

from backtest.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, build_runtime_config, parse_date
from backtest.data_loader import DEFAULT_DATA_ROOT, SymbolDataset, list_available_symbols, load_datasets
from backtest.runner import BacktestRunner
from backtest.window_sweep import resolve_common_range

logger = logging.getLogger(__name__)


DEFAULT_REPORT_DIR = PROJECT_ROOT / "backtest_reports" / "rolling_windows"


@dataclass(frozen=True)
class RollingWindowSpec:
    index: int
    start: datetime
    end: datetime


@dataclass(frozen=True)
class RollingWindowResultRow:
    run_id: str
    strategy_profile: str
    window_index: int
    window_start: datetime
    window_end: datetime
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    trade_count: int
    win_rate: float
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    annualized_volatility_pct: float


@dataclass(frozen=True)
class RollingWindowSummary:
    strategy_profile: str
    window_count: int
    positive_windows: int
    non_negative_windows: int
    positive_window_ratio: float
    mean_return_pct: float
    median_return_pct: float
    worst_window_return_pct: float
    best_window_return_pct: float
    mean_max_drawdown_pct: float
    worst_max_drawdown_pct: float
    mean_trade_count: float
    mean_win_rate: float
    median_win_rate: float
    mean_profit_factor: float
    mean_annualized_volatility_pct: float


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            for key, value in list(payload.items()):
                if isinstance(value, datetime):
                    payload[key] = value.isoformat()
            writer.writerow(payload)


def build_rolling_windows(
    range_start: datetime,
    range_end: datetime,
    *,
    window_days: int,
    step_days: int,
) -> list[RollingWindowSpec]:
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if range_end < range_start:
        raise ValueError("range_end must be on or after range_start")

    last_start = range_end - timedelta(days=window_days - 1)
    if last_start < range_start:
        raise ValueError("Requested range is shorter than the window length")

    windows: list[RollingWindowSpec] = []
    cursor = range_start
    index = 1
    while cursor <= last_start:
        windows.append(
            RollingWindowSpec(
                index=index,
                start=cursor,
                end=cursor + timedelta(days=window_days - 1),
            )
        )
        cursor += timedelta(days=step_days)
        index += 1
    return windows


def slice_datasets_inclusive(
    datasets: dict[str, SymbolDataset],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, SymbolDataset]:
    sliced: dict[str, SymbolDataset] = {}
    for symbol, dataset in datasets.items():
        frame = dataset.frame.loc[
            (dataset.frame["timestamp"] >= start)
            & (dataset.frame["timestamp"] <= end)
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No rows left for {symbol} inside window {start.isoformat()} -> {end.isoformat()}")
        sliced[symbol] = SymbolDataset(symbol=symbol, path=dataset.path, frame=frame)
    return sliced


def summarize_results(profile: str, rows: Sequence[RollingWindowResultRow]) -> RollingWindowSummary:
    if not rows:
        raise ValueError("No rolling window rows supplied")

    returns = pd.Series([row.total_return_pct for row in rows], dtype="float64")
    drawdowns = pd.Series([row.max_drawdown_pct for row in rows], dtype="float64")
    trade_counts = pd.Series([row.trade_count for row in rows], dtype="float64")
    win_rates = pd.Series([row.win_rate for row in rows], dtype="float64")
    profit_factors = pd.Series([row.profit_factor for row in rows], dtype="float64")
    annualized_vols = pd.Series([row.annualized_volatility_pct for row in rows], dtype="float64")

    positive_windows = int((returns > 0.0).sum())
    non_negative_windows = int((returns >= 0.0).sum())
    window_count = len(rows)

    return RollingWindowSummary(
        strategy_profile=profile,
        window_count=window_count,
        positive_windows=positive_windows,
        non_negative_windows=non_negative_windows,
        positive_window_ratio=float(positive_windows / window_count),
        mean_return_pct=float(returns.mean()),
        median_return_pct=float(returns.median()),
        worst_window_return_pct=float(returns.min()),
        best_window_return_pct=float(returns.max()),
        mean_max_drawdown_pct=float(drawdowns.mean()),
        worst_max_drawdown_pct=float(drawdowns.max()),
        mean_trade_count=float(trade_counts.mean()),
        mean_win_rate=float(win_rates.mean()),
        median_win_rate=float(win_rates.median()),
        mean_profit_factor=float(profit_factors.mean()),
        mean_annualized_volatility_pct=float(annualized_vols.mean()),
    )


async def run_rolling_windows(
    *,
    config_path: Path,
    data_root: Path,
    report_dir: Path,
    profile: str,
    symbols: Sequence[str],
    range_start: datetime,
    range_end: datetime,
    window_days: int,
    step_days: int,
    initial_capital: float | None,
    run_name: str | None,
) -> Path:
    datasets = load_datasets(data_root, symbols, start=range_start, end=range_end)
    common_start, common_end = resolve_common_range(datasets)
    effective_start = max(range_start, common_start)
    effective_end = min(range_end, common_end)
    windows = build_rolling_windows(
        effective_start,
        effective_end,
        window_days=window_days,
        step_days=step_days,
    )

    rolling_id = run_name or f"{profile}_rolling_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    rolling_root = report_dir / rolling_id
    runs_root = rolling_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    rows: list[RollingWindowResultRow] = []
    for window in windows:
        config = build_runtime_config(
            config_path,
            profile=profile,
            symbols=symbols,
            initial_capital=initial_capital,
        )
        window_datasets = slice_datasets_inclusive(datasets, start=window.start, end=window.end)
        run_id = f"{profile}_w{window.index:03d}"
        runner = BacktestRunner(config, runs_root)
        result = await runner.run(window_datasets, run_id=run_id)
        rows.append(
            RollingWindowResultRow(
                run_id=result.run_id,
                strategy_profile=result.strategy_profile,
                window_index=window.index,
                window_start=result.start,
                window_end=result.end,
                total_return_pct=result.total_return_pct,
                max_drawdown_pct=result.max_drawdown_pct,
                sharpe_ratio=result.sharpe_ratio,
                sortino_ratio=result.sortino_ratio,
                calmar_ratio=result.calmar_ratio,
                trade_count=result.trade_count,
                win_rate=result.win_rate,
                avg_trade_return_pct=result.avg_trade_return_pct,
                avg_win_pct=result.avg_win_pct,
                avg_loss_pct=result.avg_loss_pct,
                profit_factor=result.profit_factor,
                annualized_volatility_pct=result.annualized_volatility_pct,
            )
        )

    summary = summarize_results(profile, rows)
    _write_csv(rolling_root / "window_results.csv", rows)

    summary_payload = asdict(summary)
    (rolling_root / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    manifest = {
        "rolling_id": rolling_id,
        "config_path": str(config_path),
        "data_root": str(data_root),
        "strategy_profile": profile,
        "symbols": list(symbols),
        "requested_start": range_start.isoformat(),
        "requested_end": range_end.isoformat(),
        "effective_start": effective_start.isoformat(),
        "effective_end": effective_end.isoformat(),
        "window_days": window_days,
        "step_days": step_days,
        "window_count": len(windows),
    }
    (rolling_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return rolling_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rolling-window backtests for a single local strategy profile")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to project config")
    parser.add_argument("--profile", required=True, help="Strategy profile to evaluate")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Directory containing OHLCV parquet files")
    parser.add_argument("--start", required=True, help="Inclusive rolling analysis start date, e.g. 2026-02-01")
    parser.add_argument("--end", required=True, help="Inclusive rolling analysis end date, e.g. 2026-03-17")
    parser.add_argument("--symbols", nargs="*", default=None, help="Explicit symbol list")
    parser.add_argument("--window-days", type=int, default=10, help="Rolling window length in days")
    parser.add_argument("--step-days", type=int, default=1, help="Step size between windows in days")
    parser.add_argument("--initial-capital", type=float, default=None, help="Override initial cash")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Directory for rolling artifacts")
    parser.add_argument("--name", default=None, help="Optional rolling run directory name")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    return parser


async def _run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    start = parse_date(args.start)
    end = parse_date(args.end)
    if start is None or end is None:
        raise SystemExit("--start and --end must be valid ISO dates")

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

    rolling_root = await run_rolling_windows(
        config_path=config_path,
        data_root=data_root,
        report_dir=report_dir,
        profile=args.profile,
        symbols=symbols,
        range_start=start,
        range_end=end,
        window_days=args.window_days,
        step_days=args.step_days,
        initial_capital=args.initial_capital,
        run_name=args.name,
    )

    print(f"Rolling artifacts: {rolling_root}")
    print(f"Window results: {rolling_root / 'window_results.csv'}")
    print(f"Summary: {rolling_root / 'summary.json'}")
    print(f"Manifest: {rolling_root / 'manifest.json'}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    return asyncio.run(_run(args))
