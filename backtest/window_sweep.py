from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

from backtest.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, build_runtime_config
from backtest.data_loader import (
    DEFAULT_DATA_ROOT,
    SymbolDataset,
    list_available_symbols,
    load_datasets,
)
from backtest.runner import BacktestRunner

logger = logging.getLogger(__name__)


DEFAULT_SWEEP_DIR = PROJECT_ROOT / "backtest_reports" / "window_sweeps"

LIQUID_CORE_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "BNB/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LTC/USDT",
    "DOT/USDT",
    "UNI/USDT",
]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    profile: str
    overrides: Dict[str, Any]


@dataclass(frozen=True)
class WindowSpec:
    index: int
    start: datetime
    end_exclusive: datetime


@dataclass(frozen=True)
class WindowResultRow:
    strategy_name: str
    strategy_profile: str
    run_id: str
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
    profit_factor: float
    annualized_volatility_pct: float = 0.0


@dataclass(frozen=True)
class StrategySummaryRow:
    strategy_name: str
    strategy_profile: str
    window_count: int
    positive_windows: int
    non_negative_windows: int
    mean_return_pct: float
    median_return_pct: float
    worst_window_return_pct: float
    best_window_return_pct: float
    mean_max_drawdown_pct: float
    worst_max_drawdown_pct: float
    mean_sharpe_ratio: float
    mean_profit_factor: float
    total_trades: int
    target_hit_ge_3pct: int
    target_hit_ge_5pct: int
    target_hit_band_3_to_5_pct: int
    mean_sortino_ratio: float = 0.0
    mean_calmar_ratio: float = 0.0
    mean_annualized_volatility_pct: float = 0.0
    worst_annualized_volatility_pct: float = 0.0


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _resolve_strategy_specs(
    include_lstm: bool,
    lstm_model_path: Path | None,
) -> list[StrategySpec]:
    strategies = [
        StrategySpec(name="conservative", profile="capital_preservation_v1", overrides={}),
        StrategySpec(name="normal", profile="regime_trend_v1", overrides={}),
        StrategySpec(name="trend_pullback_t", profile="trend_pullback_t_v1", overrides={}),
    ]

    if not include_lstm:
        return strategies

    if lstm_model_path is None or not lstm_model_path.exists():
        raise SystemExit(
            "LSTM comparison requested but --lstm-model-path was not supplied or does not exist"
        )

    strategies.append(
        StrategySpec(
            name="lstm",
            profile="capital_preservation_v1",
            overrides={
                "alpha": {"engine": "lstm", "model_path": str(lstm_model_path)},
                "strategy": {"use_model_overlay": False},
            },
        )
    )
    return strategies


def _infer_candle_step(datasets: Dict[str, SymbolDataset]) -> timedelta:
    candidates: list[timedelta] = []
    for dataset in datasets.values():
        timestamps = dataset.frame["timestamp"].drop_duplicates().sort_values()
        if len(timestamps) < 2:
            continue
        deltas = timestamps.diff().dropna()
        positive = deltas[deltas > pd.Timedelta(0)]
        if positive.empty:
            continue
        candidates.append(positive.min().to_pytimedelta())

    if not candidates:
        return timedelta(minutes=1)
    return min(candidates)


def resolve_common_range(datasets: Dict[str, SymbolDataset]) -> tuple[datetime, datetime]:
    starts = [dataset.frame.iloc[0]["timestamp"].to_pydatetime() for dataset in datasets.values()]
    ends = [dataset.frame.iloc[-1]["timestamp"].to_pydatetime() for dataset in datasets.values()]
    return max(starts), min(ends)


def build_rolling_windows(
    common_start: datetime,
    common_end: datetime,
    candle_step: timedelta,
    *,
    lookback_days: int,
    window_days: int,
) -> list[WindowSpec]:
    if lookback_days <= 0 or window_days <= 0:
        raise ValueError("lookback_days and window_days must both be positive")
    if lookback_days % window_days != 0:
        raise ValueError("lookback_days must be divisible by window_days for non-overlapping windows")

    analysis_end_exclusive = common_end + candle_step
    analysis_start = analysis_end_exclusive - timedelta(days=lookback_days)
    if analysis_start < common_start:
        raise ValueError(
            f"Not enough shared history for a {lookback_days}-day sweep. "
            f"Common range starts at {common_start.isoformat()} and ends at {common_end.isoformat()}."
        )

    windows: list[WindowSpec] = []
    n_windows = lookback_days // window_days
    for idx in range(n_windows):
        window_start = analysis_start + timedelta(days=idx * window_days)
        window_end_exclusive = window_start + timedelta(days=window_days)
        windows.append(WindowSpec(index=idx + 1, start=window_start, end_exclusive=window_end_exclusive))
    return windows


def slice_datasets(
    datasets: Dict[str, SymbolDataset],
    *,
    start: datetime,
    end_exclusive: datetime,
) -> Dict[str, SymbolDataset]:
    sliced: Dict[str, SymbolDataset] = {}
    for symbol, dataset in datasets.items():
        frame = dataset.frame.loc[
            (dataset.frame["timestamp"] >= start)
            & (dataset.frame["timestamp"] < end_exclusive)
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"No rows left for {symbol} inside window {start.isoformat()} -> {end_exclusive.isoformat()}"
            )
        sliced[symbol] = SymbolDataset(symbol=symbol, path=dataset.path, frame=frame)
    return sliced


def summarize_window_results(rows: Sequence[WindowResultRow]) -> list[StrategySummaryRow]:
    grouped: dict[str, list[WindowResultRow]] = {}
    for row in rows:
        grouped.setdefault(row.strategy_name, []).append(row)

    summaries: list[StrategySummaryRow] = []
    for strategy_name, strategy_rows in grouped.items():
        returns = [row.total_return_pct for row in strategy_rows]
        drawdowns = [row.max_drawdown_pct for row in strategy_rows]
        sharpes = [row.sharpe_ratio for row in strategy_rows]
        sortinos = [row.sortino_ratio for row in strategy_rows]
        calmars = [row.calmar_ratio for row in strategy_rows]
        annualized_vols = [row.annualized_volatility_pct for row in strategy_rows]
        profit_factors = [row.profit_factor for row in strategy_rows]

        summaries.append(
            StrategySummaryRow(
                strategy_name=strategy_name,
                strategy_profile=strategy_rows[0].strategy_profile,
                window_count=len(strategy_rows),
                positive_windows=sum(1 for value in returns if value > 0.0),
                non_negative_windows=sum(1 for value in returns if value >= 0.0),
                mean_return_pct=sum(returns) / len(returns),
                median_return_pct=float(pd.Series(returns).median()),
                worst_window_return_pct=min(returns),
                best_window_return_pct=max(returns),
                mean_max_drawdown_pct=sum(drawdowns) / len(drawdowns),
                worst_max_drawdown_pct=max(drawdowns),
                mean_sharpe_ratio=sum(sharpes) / len(sharpes),
                mean_sortino_ratio=sum(sortinos) / len(sortinos),
                mean_calmar_ratio=sum(calmars) / len(calmars),
                mean_annualized_volatility_pct=sum(annualized_vols) / len(annualized_vols),
                worst_annualized_volatility_pct=max(annualized_vols),
                mean_profit_factor=sum(profit_factors) / len(profit_factors),
                total_trades=sum(row.trade_count for row in strategy_rows),
                target_hit_ge_3pct=sum(1 for value in returns if value >= 3.0),
                target_hit_ge_5pct=sum(1 for value in returns if value >= 5.0),
                target_hit_band_3_to_5_pct=sum(1 for value in returns if 3.0 <= value <= 5.0),
            )
        )

    return sorted(summaries, key=lambda row: row.mean_return_pct, reverse=True)


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            for key, value in list(payload.items()):
                if isinstance(value, datetime):
                    payload[key] = value.isoformat()
            writer.writerow(payload)


async def run_window_sweep(
    *,
    config_path: Path,
    data_root: Path,
    report_dir: Path,
    symbols: Sequence[str],
    lookback_days: int,
    window_days: int,
    initial_capital: float | None,
    include_lstm: bool,
    lstm_model_path: Path | None,
    sweep_name: str | None,
) -> Path:
    strategy_specs = _resolve_strategy_specs(include_lstm, lstm_model_path)
    datasets = load_datasets(data_root, symbols)
    candle_step = _infer_candle_step(datasets)
    common_start, common_end = resolve_common_range(datasets)
    windows = build_rolling_windows(
        common_start,
        common_end,
        candle_step,
        lookback_days=lookback_days,
        window_days=window_days,
    )

    analysis_start = windows[0].start
    analysis_end = windows[-1].end_exclusive - candle_step
    sweep_id = sweep_name or f"sweep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    sweep_root = report_dir / sweep_id
    runs_root = sweep_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    window_rows: list[WindowResultRow] = []
    for strategy in strategy_specs:
        logger.info("Running strategy %s (%s) across %d windows", strategy.name, strategy.profile, len(windows))
        for window in windows:
            config = build_runtime_config(
                config_path,
                profile=strategy.profile,
                symbols=symbols,
                initial_capital=initial_capital,
            )
            if strategy.overrides:
                _deep_merge(config, strategy.overrides)

            window_datasets = slice_datasets(datasets, start=window.start, end_exclusive=window.end_exclusive)
            run_id = f"{strategy.name}_w{window.index:02d}"
            runner = BacktestRunner(config, runs_root)
            result = await runner.run(window_datasets, run_id=run_id)
            window_rows.append(
                WindowResultRow(
                    strategy_name=strategy.name,
                    strategy_profile=result.strategy_profile,
                    run_id=result.run_id,
                    window_index=window.index,
                    window_start=result.start,
                    window_end=result.end,
                    total_return_pct=result.total_return_pct,
                    max_drawdown_pct=result.max_drawdown_pct,
                    sharpe_ratio=result.sharpe_ratio,
                    sortino_ratio=result.sortino_ratio,
                    calmar_ratio=result.calmar_ratio,
                    annualized_volatility_pct=result.annualized_volatility_pct,
                    trade_count=result.trade_count,
                    win_rate=result.win_rate,
                    avg_trade_return_pct=result.avg_trade_return_pct,
                    profit_factor=result.profit_factor,
                )
            )

    strategy_rows = summarize_window_results(window_rows)
    _write_csv(sweep_root / "window_results.csv", window_rows)
    _write_csv(sweep_root / "strategy_summary.csv", strategy_rows)

    manifest = {
        "sweep_id": sweep_id,
        "config_path": str(config_path),
        "data_root": str(data_root),
        "symbols": list(symbols),
        "lookback_days": lookback_days,
        "window_days": window_days,
        "window_count": len(windows),
        "common_start": common_start.isoformat(),
        "common_end": common_end.isoformat(),
        "analysis_start": analysis_start.isoformat(),
        "analysis_end": analysis_end.isoformat(),
        "candle_step_seconds": candle_step.total_seconds(),
        "strategies": [asdict(spec) for spec in strategy_specs],
    }
    (sweep_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return sweep_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run non-overlapping rolling-window comparisons for local strategy profiles"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to project config")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Directory containing OHLCV parquet files")
    parser.add_argument("--report-dir", default=str(DEFAULT_SWEEP_DIR), help="Directory for rolling comparison artifacts")
    parser.add_argument("--symbols", nargs="*", default=None, help="Explicit symbol list. Defaults to the 12-symbol liquid-core basket.")
    parser.add_argument("--lookback-days", type=int, default=90, help="Length of the rolling evaluation sample in days")
    parser.add_argument("--window-days", type=int, default=10, help="Non-overlapping window length in days")
    parser.add_argument("--initial-capital", type=float, default=None, help="Override initial cash")
    parser.add_argument("--include-lstm", action="store_true", help="Include an LSTM strategy variant. Requires --lstm-model-path.")
    parser.add_argument("--lstm-model-path", default=None, help="Path to a trained LSTM/ONNX artifact")
    parser.add_argument("--name", default=None, help="Optional sweep directory name")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    return parser


async def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    lstm_model_path = Path(args.lstm_model_path).expanduser().resolve() if args.lstm_model_path else None

    available = set(list_available_symbols(data_root))
    if not available:
        raise SystemExit(f"No parquet files found under {data_root}")

    requested_symbols = args.symbols or LIQUID_CORE_SYMBOLS
    missing = sorted(symbol for symbol in requested_symbols if symbol not in available)
    symbols = [symbol for symbol in requested_symbols if symbol in available]
    if missing:
        print(f"Skipping symbols without data: {', '.join(missing)}")
    if not symbols:
        raise SystemExit("No symbols with available historical data were selected")

    sweep_root = await run_window_sweep(
        config_path=config_path,
        data_root=data_root,
        report_dir=report_dir,
        symbols=symbols,
        lookback_days=args.lookback_days,
        window_days=args.window_days,
        initial_capital=args.initial_capital,
        include_lstm=args.include_lstm,
        lstm_model_path=lstm_model_path,
        sweep_name=args.name,
    )

    print(f"Sweep artifacts: {sweep_root}")
    print(f"Strategy summary: {sweep_root / 'strategy_summary.csv'}")
    print(f"Manifest: {sweep_root / 'manifest.json'}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    return asyncio.run(_run(args))
