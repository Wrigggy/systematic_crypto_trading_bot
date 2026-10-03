"""Causal inert bundle ingestion and explicit ablation configuration."""
import json

import numpy as np
import pytest
from datetime import datetime

from scripts.replay_research_bundle import bundle_events
from strategy.hybrid import HourlyHistory


def test_price_only_does_not_open_any_model_artifact(tmp_path):
    path = tmp_path / 'prices.npz'
    np.savez(path, price_times=[3605, 3610], prices=[[100.], [101.]],
             hourly_times=[3600], hourly_prices=[[99.]], symbols=['BTC/USDT'])
    events = list(bundle_events(path, tmp_path / 'absent_bundle', ['BTC/USDT'], False))
    assert [e.available_at for e in events] == [3601, 3605, 3610]
    assert events[1].timestamp == 3604
    assert events[0].kind == 'hourly'


def test_hourly_band_cache_invalidates_at_new_hour_and_new_data():
    history = HourlyHistory(1, 8)
    for h in range(1, 25):
        history.add('BTC/USDT', h * 3600, 100 + h, h * 3600)
    previous = history.bands('BTC/USDT', 24 * 3600)
    assert previous is not None
    assert history.bands('BTC/USDT', 24 * 3600 + 100) == previous
    assert history.bands('BTC/USDT', 25 * 3600) is None
    history.add('BTC/USDT', 25 * 3600, 200, 25 * 3600 + 1)
    assert history.bands('BTC/USDT', 25 * 3600 + 1) != previous
    assert history.bands('ETH/USDT', 25 * 3600 + 1) is None


@pytest.mark.asyncio
async def test_actual_exit_time_excludes_delayed_receipt():
    from core.models import Order, OrderStatus, OrderType, Side
    from strategy.hybrid import Holding
    from tests.test_hybrid import setup_hybrid
    clock, tracker, manager, hybrid, step = await setup_hybrid()
    now = clock[0]
    hybrid.holdings['BTC/USDT'] = Holding(now - 100, 101, 100, 100)
    order = Order(symbol='BTC/USDT', side=Side.SELL, order_type=OrderType.MARKET,
                  quantity=1, status=OrderStatus.FILLED, filled_quantity=1,
                  filled_price=101, filled_at=datetime.utcfromtimestamp(now - 20))
    hybrid._on_order(order)
    closed = next(e for e in hybrid.events if e['event'] == 'position_closed')
    assert closed['holding_seconds'] == 80
    assert closed['receipt_delay_seconds'] == 20
