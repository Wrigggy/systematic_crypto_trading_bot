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


def test_fast_score_moments_match_statistics_reference():
    from statistics import mean, pstdev
    from plugins.model_inference.forecasts import CausalScores
    from tests.test_hybrid import packet
    values = np.random.default_rng(111).normal(.00001, .00003, 200)
    scores = CausalScores(window_seconds=3000, min_samples=10, min_span_seconds=600,
                          max_gap_seconds=300)
    for i, value in enumerate(values):
        result = scores.update(packet(i * 60, float(value)))
        if i >= 10:
            past = values[max(0, i - 50):i].tolist()
            expected = (value - mean(past)) / pstdev(past)
            assert result['short'] == pytest.approx(expected, abs=1e-12)


@pytest.mark.asyncio
async def test_matrix_keeps_validation_choice_and_all_controls(tmp_path, monkeypatch):
    import time
    from scripts import run_research_matrix as matrix
    study = tmp_path / 'study'
    study.mkdir()
    (study / 'selection.json').write_text(json.dumps(dict(selected='shared',
        selection_uses='validation_only', test_opened=False)))
    manifest = dict(model_version='fixture', preprocessing_version='fixture', heads=[])
    for name in ('shared', 'grouped'):
        folder = study / f'bundle_{name}'
        folder.mkdir()
        (folder / 'bundle.json').write_text(json.dumps(manifest))
    prices = tmp_path / 'prices.npz'
    prices.with_suffix('.json').write_text(json.dumps(dict(replay_start=0, replay_end=14 * 86400)))
    seen = []
    async def fake_replay(cfg, events):
        seen.append(cfg)
        return dict(summary=dict(net_return=-.01, active_fill_days_utc8=0), events=[])
    monkeypatch.setattr(matrix, 'run_replay', fake_replay)
    monkeypatch.setattr(matrix, 'bundle_events', lambda *args: iter(()))
    monkeypatch.setattr(matrix, 'extra_metrics', lambda result: {})
    out = tmp_path / 'results'
    await matrix.run(study, prices, out, time.time() + 60)
    report = json.loads((out / 'comparison.json').read_text())
    assert report['status'] == 'complete'
    assert report['selected_by_validation'] == 'shared'
    assert len(seen) == 9
    assert all(c['strategy']['decision_interval_seconds'] == 300 for c in seen)
    assert sum(c['strategy']['fusion']['mode'] == 'price_only' for c in seen) == 2
    assert sum(c['strategy']['model_exit_enabled'] for c in seen) == 3
