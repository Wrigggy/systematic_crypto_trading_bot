"""Tests for execution/sim_executor.py — SimExecutor fills, slippage, limits."""

from datetime import datetime, timedelta

import pytest

from core.models import Order, OrderStatus, OrderType, Side
from data.buffer import LiveBuffer
from execution.sim_executor import SimExecutor
from tests.conftest import make_candle


@pytest.fixture
def buffer():
    return LiveBuffer(max_candles=10)


@pytest.fixture
def executor(buffer):
    config = {"slippage_bps": 5, "fee_bps": 10}
    return SimExecutor(config, buffer)


@pytest.mark.asyncio
class TestMarketOrders:
    async def test_buy_fills_with_slippage(self, executor, buffer):
        ts0 = datetime(2025, 1, 1)
        await buffer.push_candle(make_candle(symbol="BTC/USDT", close=100.0, ts=ts0))
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        result = await executor.execute(order)
        assert result.status == OrderStatus.SUBMITTED

        await buffer.push_candle(
            make_candle(
                symbol="BTC/USDT",
                close=101.0,
                open_=100.0,
                ts=ts0 + timedelta(minutes=1),
            )
        )
        filled = await executor.get_status(order.order_id, "BTC/USDT")
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_price == pytest.approx(100.05, abs=0.01)  # next open +5bps
        assert filled.filled_quantity == 1.0

    async def test_sell_fills_with_slippage(self, executor, buffer):
        ts0 = datetime(2025, 1, 1)
        await buffer.push_candle(make_candle(symbol="BTC/USDT", close=100.0, ts=ts0))
        order = Order(
            symbol="BTC/USDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        result = await executor.execute(order)
        assert result.status == OrderStatus.SUBMITTED

        await buffer.push_candle(
            make_candle(
                symbol="BTC/USDT",
                close=99.0,
                open_=100.0,
                ts=ts0 + timedelta(minutes=1),
            )
        )
        filled = await executor.get_status(order.order_id, "BTC/USDT")
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_price == pytest.approx(99.95, abs=0.01)  # next open -5bps

    async def test_no_price_data_rejects(self, executor, buffer):
        order = Order(
            symbol="XRP/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        result = await executor.execute(order)
        assert result.status == OrderStatus.REJECTED


@pytest.mark.asyncio
class TestLimitOrders:
    async def test_limit_buy_fills_when_price_below(self, executor, buffer):
        ts0 = datetime(2025, 1, 1)
        await buffer.push_candle(make_candle(symbol="BTC/USDT", close=95.0, ts=ts0))
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0,
        )
        result = await executor.execute(order)
        assert result.status == OrderStatus.SUBMITTED
        await buffer.push_candle(
            make_candle(
                symbol="BTC/USDT",
                close=102.0,
                open_=101.0,
                low=99.0,
                high=103.0,
                ts=ts0 + timedelta(minutes=1),
            )
        )
        filled = await executor.get_status(order.order_id, "BTC/USDT")
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_price == 100.0

    async def test_limit_buy_pending_when_above(self, executor, buffer):
        ts0 = datetime(2025, 1, 1)
        await buffer.push_candle(make_candle(symbol="BTC/USDT", close=105.0, ts=ts0))
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0,
        )
        result = await executor.execute(order)
        assert result.status == OrderStatus.SUBMITTED

    async def test_limit_sell_fills_when_price_above(self, executor, buffer):
        ts0 = datetime(2025, 1, 1)
        await buffer.push_candle(make_candle(symbol="BTC/USDT", close=110.0, ts=ts0))
        order = Order(
            symbol="BTC/USDT",
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0,
        )
        result = await executor.execute(order)
        assert result.status == OrderStatus.SUBMITTED
        await buffer.push_candle(
            make_candle(
                symbol="BTC/USDT",
                close=101.0,
                open_=99.0,
                low=98.0,
                high=102.0,
                ts=ts0 + timedelta(minutes=1),
            )
        )
        filled = await executor.get_status(order.order_id, "BTC/USDT")
        assert filled.status == OrderStatus.FILLED

    async def test_limit_sell_pending_when_below(self, executor, buffer):
        ts0 = datetime(2025, 1, 1)
        await buffer.push_candle(make_candle(symbol="BTC/USDT", close=90.0, ts=ts0))
        order = Order(
            symbol="BTC/USDT",
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0,
        )
        result = await executor.execute(order)
        assert result.status == OrderStatus.SUBMITTED


@pytest.mark.asyncio
class TestCancel:
    async def test_cancel_pending(self, executor, buffer):
        await buffer.push_candle(make_candle(symbol="BTC/USDT", close=105.0))
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0,
        )
        submitted = await executor.execute(order)
        assert submitted.status == OrderStatus.SUBMITTED

        cancelled = await executor.cancel(submitted.order_id, "BTC/USDT")
        assert cancelled.status == OrderStatus.CANCELLED

    async def test_cancel_nonexistent(self, executor):
        result = await executor.cancel("fake-id", "BTC/USDT")
        assert result.status == OrderStatus.CANCELLED
