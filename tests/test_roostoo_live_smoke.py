from __future__ import annotations

import os

import pytest

from core.models import Order, OrderStatus, OrderType, Side
from execution.roostoo_executor import RoostooExecutor


def _smoke_enabled() -> bool:
    return os.getenv("RUN_ROOSTOO_SMOKE") == "1"


def _order_smoke_enabled() -> bool:
    return os.getenv("RUN_ROOSTOO_ORDER_SMOKE") == "1"


def _smoke_config() -> dict:
    api_key = os.getenv("ROOSTOO_COMP_API_KEY") or os.getenv("ROOSTOO_API_KEY", "")
    api_secret = os.getenv("ROOSTOO_COMP_API_SECRET") or os.getenv(
        "ROOSTOO_API_SECRET", ""
    )
    if not api_key or not api_secret:
        pytest.skip("Roostoo smoke credentials are not configured")
    return {
        "base_url": os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com"),
        "api_key": api_key,
        "api_secret": api_secret,
    }


@pytest.mark.asyncio
async def test_roostoo_live_balance_and_ticker_smoke():
    if not _smoke_enabled():
        pytest.skip("Set RUN_ROOSTOO_SMOKE=1 to run live Roostoo smoke tests")

    symbol = os.getenv("ROOSTOO_SMOKE_SYMBOL", "BTC/USDT")
    executor = RoostooExecutor(_smoke_config())
    try:
        server_time_ok = await executor._auth.validate_server_time(executor._base_url)
        await executor.start()
        balances = await executor.get_balance()
        ticker = await executor.get_ticker(symbol)
    finally:
        await executor.stop()

    assert server_time_ok is True
    assert symbol in executor._pair_info
    assert "USD" in balances
    assert ticker is not None
    assert ticker > 0


@pytest.mark.asyncio
async def test_roostoo_live_order_lifecycle_smoke():
    if not _smoke_enabled() or not _order_smoke_enabled():
        pytest.skip(
            "Set RUN_ROOSTOO_SMOKE=1 and RUN_ROOSTOO_ORDER_SMOKE=1 to run live order smoke"
        )

    symbol = os.getenv("ROOSTOO_ORDER_SMOKE_SYMBOL", "WLFI/USDT")
    target_notional = float(os.getenv("ROOSTOO_ORDER_SMOKE_TARGET_NOTIONAL", "5.0"))
    price_ratio = float(os.getenv("ROOSTOO_ORDER_SMOKE_PRICE_RATIO", "0.85"))
    if price_ratio <= 0.7 or price_ratio >= 1.0:
        pytest.skip("ROOSTOO_ORDER_SMOKE_PRICE_RATIO must stay inside (0.7, 1.0)")

    executor = RoostooExecutor(_smoke_config())
    try:
        await executor.start()
        assert symbol in executor._pair_info

        ticker = await executor.get_ticker(symbol)
        assert ticker is not None
        assert ticker > 0

        info = executor._pair_info[symbol]
        min_notional = float(info.get("min_notional") or 0.0)
        price = executor._round_price(symbol, ticker * price_ratio)
        quantity = executor._round_quantity(
            symbol,
            max((max(target_notional, min_notional * 1.1) / price), info["min_qty"]),
        )

        order = Order(
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
        )
        result = await executor.execute(order)
        assert result.status != OrderStatus.REJECTED

        query = await executor.get_status(result.order_id, symbol)
        assert query.status in {
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
        }

        if query.status != OrderStatus.FILLED:
            cancelled = await executor.cancel(result.order_id, symbol)
            assert cancelled.status == OrderStatus.CANCELLED
    finally:
        await executor.stop()
