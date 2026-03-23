from __future__ import annotations

from datetime import datetime, timedelta

from backtest.analysis import TradeAnalyzer
from core.models import Order, Side, OrderType


def _fill(side: Side, quantity: float, price: float, ts: datetime) -> Order:
    return Order(
        symbol="BTC/USDT",
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        filled_quantity=quantity,
        filled_price=price,
        created_at=ts,
        filled_at=ts,
    )


def test_trade_analyzer_matches_fifo_reentries() -> None:
    analyzer = TradeAnalyzer(fee_rate=0.001)
    base = datetime(2026, 1, 1, 0, 0)

    analyzer.on_fill(_fill(Side.BUY, 1.0, 100.0, base))
    analyzer.on_fill(_fill(Side.SELL, 0.4, 110.0, base + timedelta(minutes=5)))
    analyzer.on_fill(_fill(Side.BUY, 0.4, 105.0, base + timedelta(minutes=10)))
    analyzer.on_fill(_fill(Side.SELL, 1.0, 120.0, base + timedelta(minutes=20)))

    total_pnl = sum(trade.pnl for trade in analyzer.closed_trades)
    expected = (
        (110.0 - 100.0) * 0.4 - ((100.0 * 0.4) + (110.0 * 0.4)) * 0.001
        + (120.0 - 100.0) * 0.6 - ((100.0 * 0.6) + (120.0 * 0.6)) * 0.001
        + (120.0 - 105.0) * 0.4 - ((105.0 * 0.4) + (120.0 * 0.4)) * 0.001
    )

    assert len(analyzer.closed_trades) == 3
    assert abs(total_pnl - expected) < 1e-9


def test_trade_analyzer_symbol_summary_rows() -> None:
    analyzer = TradeAnalyzer(fee_rate=0.0)
    base = datetime(2026, 1, 1, 0, 0)

    analyzer.on_fill(
        Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            filled_quantity=1.0,
            filled_price=100.0,
            created_at=base,
            filled_at=base,
        )
    )
    analyzer.on_fill(
        Order(
            symbol="BTC/USDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            filled_quantity=1.0,
            filled_price=110.0,
            created_at=base + timedelta(minutes=10),
            filled_at=base + timedelta(minutes=10),
        )
    )
    analyzer.on_fill(
        Order(
            symbol="ETH/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            filled_quantity=1.0,
            filled_price=200.0,
            created_at=base,
            filled_at=base,
        )
    )
    analyzer.on_fill(
        Order(
            symbol="ETH/USDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            filled_quantity=1.0,
            filled_price=190.0,
            created_at=base + timedelta(minutes=5),
            filled_at=base + timedelta(minutes=5),
        )
    )

    rows = analyzer.symbol_summary_rows()
    btc = next(row for row in rows if row.symbol == "BTC/USDT")
    eth = next(row for row in rows if row.symbol == "ETH/USDT")

    assert btc.trade_count == 1
    assert btc.net_pnl == 10.0
    assert btc.win_rate == 1.0
    assert eth.trade_count == 1
    assert eth.net_pnl == -10.0
    assert eth.win_rate == 0.0
