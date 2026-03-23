from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict

import numpy as np

from core.models import Order, PortfolioSnapshot, Side


@dataclass
class EquityPoint:
    timestamp: datetime
    nav: float
    cash: float
    drawdown: float
    exposure: float


@dataclass
class ClosedTrade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    hold_minutes: float


@dataclass
class BacktestResult:
    run_id: str
    strategy_profile: str
    symbols: list[str]
    start: datetime
    end: datetime
    initial_capital: float
    final_nav: float
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
    open_positions: int
    annualized_volatility_pct: float = 0.0


@dataclass
class SymbolSummaryRow:
    symbol: str
    trade_count: int
    win_rate: float
    net_pnl: float
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_hold_minutes: float


@dataclass
class _OpenLot:
    entry_time: datetime
    quantity: float
    entry_price: float


class TradeAnalyzer:
    """Build backtest artifacts from fill events.

    Trades are matched FIFO by symbol so summary PnL stays aligned with the
    portfolio tracker even when positions are scaled in/out repeatedly.
    """

    def __init__(self, fee_rate: float = 0.0) -> None:
        self._fee_rate = max(0.0, fee_rate)
        self._open_lots: Dict[str, Deque[_OpenLot]] = defaultdict(deque)
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[EquityPoint] = []

    def on_fill(self, order: Order) -> None:
        if order.filled_quantity <= 0 or order.filled_price is None:
            return

        fill_time = order.filled_at or order.created_at
        symbol = order.symbol
        fill_qty = float(order.filled_quantity)
        fill_price = float(order.filled_price)

        if order.side == Side.BUY:
            self._open_lots[symbol].append(
                _OpenLot(
                    entry_time=fill_time,
                    quantity=fill_qty,
                    entry_price=fill_price,
                )
            )
            return

        remaining = fill_qty
        lots = self._open_lots[symbol]
        while remaining > 1e-12 and lots:
            lot = lots[0]
            realized_qty = min(lot.quantity, remaining)
            entry_notional = lot.entry_price * realized_qty
            exit_notional = fill_price * realized_qty
            fees = (entry_notional + exit_notional) * self._fee_rate
            pnl = (fill_price - lot.entry_price) * realized_qty - fees
            pnl_pct = pnl / entry_notional if entry_notional > 0 else 0.0
            hold_minutes = (fill_time - lot.entry_time).total_seconds() / 60.0
            self.closed_trades.append(
                ClosedTrade(
                    symbol=symbol,
                    entry_time=lot.entry_time,
                    exit_time=fill_time,
                    quantity=realized_qty,
                    entry_price=lot.entry_price,
                    exit_price=fill_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    hold_minutes=hold_minutes,
                )
            )

            lot.quantity -= realized_qty
            remaining -= realized_qty
            if lot.quantity <= 1e-12:
                lots.popleft()

    def record_equity(self, timestamp: datetime, snapshot: PortfolioSnapshot, exposure: float) -> None:
        if self.equity_curve and self.equity_curve[-1].timestamp == timestamp:
            self.equity_curve[-1] = EquityPoint(
                timestamp=timestamp,
                nav=snapshot.nav,
                cash=snapshot.cash,
                drawdown=snapshot.drawdown,
                exposure=exposure,
            )
            return

        self.equity_curve.append(
            EquityPoint(
                timestamp=timestamp,
                nav=snapshot.nav,
                cash=snapshot.cash,
                drawdown=snapshot.drawdown,
                exposure=exposure,
            )
        )

    def build_result(
        self,
        *,
        run_id: str,
        strategy_profile: str,
        symbols: list[str],
        start: datetime,
        end: datetime,
        initial_capital: float,
        open_positions: int,
    ) -> BacktestResult:
        if not self.equity_curve:
            raise ValueError("No equity curve recorded")

        final_nav = self.equity_curve[-1].nav
        total_return_pct = ((final_nav / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0
        navs = np.array([point.nav for point in self.equity_curve], dtype=np.float64)
        peaks = np.maximum.accumulate(navs)
        drawdowns = (peaks - navs) / np.where(peaks > 0, peaks, 1.0)
        max_drawdown_pct = float(np.max(drawdowns) * 100.0) if len(drawdowns) else 0.0

        daily_navs = self._daily_navs()
        returns = np.diff(daily_navs) / daily_navs[:-1] if len(daily_navs) >= 2 else np.array([], dtype=np.float64)

        sharpe = 0.0
        sortino = 0.0
        calmar = 0.0
        annualized_volatility_pct = 0.0
        if len(returns) >= 2:
            mean_ret = float(np.mean(returns))
            std = float(np.std(returns, ddof=1))
            if std > 1e-10:
                sharpe = mean_ret / std * np.sqrt(365.0)
                annualized_volatility_pct = std * np.sqrt(365.0) * 100.0

            downside = returns[returns < 0]
            if len(downside) >= 1:
                downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float(np.std(downside, ddof=0))
                if downside_std > 1e-10:
                    sortino = mean_ret / downside_std * np.sqrt(365.0)

            n_days = len(daily_navs) - 1
            annualized_return = ((final_nav / initial_capital) - 1.0) * (365.0 / max(n_days, 1))
            max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
            if max_dd > 1e-10:
                calmar = annualized_return / max_dd

        closed = self.closed_trades
        trade_returns = np.array([trade.pnl_pct for trade in closed], dtype=np.float64) if closed else np.array([], dtype=np.float64)
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns <= 0]
        profit_factor = 0.0
        if closed:
            positive_pnl = sum(max(0.0, trade.pnl) for trade in closed)
            negative_pnl = abs(sum(min(0.0, trade.pnl) for trade in closed))
            if negative_pnl > 1e-10:
                profit_factor = positive_pnl / negative_pnl

        return BacktestResult(
            run_id=run_id,
            strategy_profile=strategy_profile,
            symbols=symbols,
            start=start,
            end=end,
            initial_capital=initial_capital,
            final_nav=final_nav,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            annualized_volatility_pct=float(annualized_volatility_pct),
            trade_count=len(closed),
            win_rate=float(len(wins) / len(closed)) if closed else 0.0,
            avg_trade_return_pct=float(np.mean(trade_returns) * 100.0) if len(trade_returns) else 0.0,
            avg_win_pct=float(np.mean(wins) * 100.0) if len(wins) else 0.0,
            avg_loss_pct=float(np.mean(losses) * 100.0) if len(losses) else 0.0,
            profit_factor=float(profit_factor),
            open_positions=open_positions,
        )

    def write_artifacts(self, report_dir: Path, result: BacktestResult) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)

        summary_payload = asdict(result)
        summary_payload["start"] = result.start.isoformat()
        summary_payload["end"] = result.end.isoformat()
        (report_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

        with open(report_dir / "equity_curve.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "nav", "cash", "drawdown", "exposure"])
            for point in self.equity_curve:
                writer.writerow(
                    [
                        point.timestamp.isoformat(),
                        f"{point.nav:.8f}",
                        f"{point.cash:.8f}",
                        f"{point.drawdown:.8f}",
                        f"{point.exposure:.8f}",
                    ]
                )

        with open(report_dir / "closed_trades.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "symbol",
                    "entry_time",
                    "exit_time",
                    "quantity",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "pnl_pct",
                    "hold_minutes",
                ]
            )
            for trade in self.closed_trades:
                writer.writerow(
                    [
                        trade.symbol,
                        trade.entry_time.isoformat(),
                        trade.exit_time.isoformat(),
                        f"{trade.quantity:.8f}",
                        f"{trade.entry_price:.8f}",
                        f"{trade.exit_price:.8f}",
                        f"{trade.pnl:.8f}",
                        f"{trade.pnl_pct:.8f}",
                        f"{trade.hold_minutes:.4f}",
                    ]
                )

        with open(report_dir / "symbol_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "symbol",
                    "trade_count",
                    "win_rate",
                    "net_pnl",
                    "avg_trade_return_pct",
                    "avg_win_pct",
                    "avg_loss_pct",
                    "profit_factor",
                    "avg_hold_minutes",
                ]
            )
            for row in self.symbol_summary_rows():
                writer.writerow(
                    [
                        row.symbol,
                        row.trade_count,
                        f"{row.win_rate:.8f}",
                        f"{row.net_pnl:.8f}",
                        f"{row.avg_trade_return_pct:.8f}",
                        f"{row.avg_win_pct:.8f}",
                        f"{row.avg_loss_pct:.8f}",
                        f"{row.profit_factor:.8f}",
                        f"{row.avg_hold_minutes:.4f}",
                    ]
                )

    def _daily_navs(self) -> np.ndarray:
        daily_last: Dict[str, float] = {}
        for point in self.equity_curve:
            daily_last[point.timestamp.strftime("%Y-%m-%d")] = point.nav
        ordered = [daily_last[key] for key in sorted(daily_last)]
        return np.array(ordered, dtype=np.float64)

    def symbol_summary_rows(self) -> list[SymbolSummaryRow]:
        grouped: Dict[str, list[ClosedTrade]] = defaultdict(list)
        for trade in self.closed_trades:
            grouped[trade.symbol].append(trade)

        rows: list[SymbolSummaryRow] = []
        for symbol, trades in sorted(grouped.items()):
            returns = np.array([trade.pnl_pct for trade in trades], dtype=np.float64)
            wins = returns[returns > 0]
            losses = returns[returns <= 0]
            positive_pnl = sum(max(0.0, trade.pnl) for trade in trades)
            negative_pnl = abs(sum(min(0.0, trade.pnl) for trade in trades))
            profit_factor = positive_pnl / negative_pnl if negative_pnl > 1e-10 else 0.0
            rows.append(
                SymbolSummaryRow(
                    symbol=symbol,
                    trade_count=len(trades),
                    win_rate=float(len(wins) / len(trades)) if trades else 0.0,
                    net_pnl=float(sum(trade.pnl for trade in trades)),
                    avg_trade_return_pct=float(np.mean(returns) * 100.0) if len(returns) else 0.0,
                    avg_win_pct=float(np.mean(wins) * 100.0) if len(wins) else 0.0,
                    avg_loss_pct=float(np.mean(losses) * 100.0) if len(losses) else 0.0,
                    profit_factor=float(profit_factor),
                    avg_hold_minutes=float(np.mean([trade.hold_minutes for trade in trades])) if trades else 0.0,
                )
            )
        return rows
