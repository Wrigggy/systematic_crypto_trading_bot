import copy

import numpy as np
import pytest

from strategy.entry_quality import EntryQualityHistory
from strategy.fusion import SignalFusion
from scripts.replay_hybrid import ReplayEvent, run_replay
from scripts.run_entry_quality import apply_quality, flow_events
from scripts.publish_entry_quality import entry_metrics
from tests.test_hybrid import setup_hybrid
from tests.test_hybrid_replay import fixture


def ready(history, end=1200):
    for i in range(20):
        history.observe('BTC/USDT', end - (19 - i) * 60, 100 + i, 100,
                        40 if i < 15 else 60, end)


def test_flow_is_causal_gap_safe_and_asset_specific():
    a = EntryQualityHistory({})
    ready(a)
    snapshot = a.check('BTC/USDT', 1200)
    assert snapshot['reason'] == 'qualified'
    assert snapshot['recent_buy_share'] == .6
    assert snapshot['previous_buy_share'] == .4
    assert a.check('XRP/USDT', 1200)['reason'] == 'flow_unready_or_stale'
    b = copy.deepcopy(a)
    a.observe('BTC/USDT', 1260, 10, 100, 0, 1260)
    assert b.check('BTC/USDT', 1200) == snapshot
    assert b.check('BTC/USDT', 1260)['reason'] == 'flow_unready_or_stale'
    a.observe('BTC/USDT', 1380, 10, 100, 0, 1380)
    assert a.resets['BTC/USDT'] == 1
    assert a.check('BTC/USDT', 1380)['reason'] == 'flow_unready_or_stale'
    with pytest.raises(ValueError):
        a.observe('BTC/USDT', 1440, 10, 100, 50, 1439)
    with pytest.raises(ValueError):
        a.observe('BTC/USDT', 1440, 10, 100, 101, 1440)


def test_zero_volume_and_non_improving_flow_fail_closed():
    a = EntryQualityHistory({})
    for i in range(20):
        a.observe('BTC/USDT', (i + 1) * 60, 100, 0, 0, 1200)
    assert a.check('BTC/USDT', 1200)['reason'] == 'flow_unready_or_stale'
    b = EntryQualityHistory({})
    for i in range(20):
        b.observe('BTC/USDT', (i + 1) * 60, 100 + i, 100, 60, 1200)
    assert b.check('BTC/USDT', 1200)['reason'] == 'flow_not_improving'


def test_deep_pullback_cannot_bypass_model_floor_or_extreme_guard():
    cfg = dict(model_score_scale=1.5, entry_threshold=.75, require_long_support=False,
               rule_strength_clip=1.5, minimum_model_strength=.5, max_price_deviation_sigma=1.5)
    fusion = SignalFusion(cfg)
    weak = dict(composite=.03, long_support=0.)
    assert fusion.evaluate(98.5, 100, 1, .5, weak).reason == 'model_below_quality_floor'
    strong = dict(composite=1.5, long_support=0.)
    assert fusion.evaluate(99, 100, 1, .5, strong).eligible
    assert fusion.evaluate(98, 100, 1, .5, strong).reason == 'extreme_deviation'
    assert fusion.evaluate(98.5, 100, 1, .5, strong).rule_strength == 1.5
    assert not fusion.evaluate(99, 100, 1, .5, None).eligible
    price = SignalFusion(dict(cfg, mode='price_only'))
    assert price.evaluate(99, 100, 1, .5, None).eligible
    assert not price.evaluate(98, 100, 1, .5, None).eligible


@pytest.mark.parametrize('update', [{'minimum_model_strength': -1}, {'rule_strength_clip': 0},
                                   {'max_price_deviation_sigma': float('nan')}])
def test_invalid_score_bounds(update):
    with pytest.raises(ValueError):
        SignalFusion(update)


def test_flow_envelope_never_marks_an_unfinished_bar_as_available():
    with pytest.raises(ValueError):
        ReplayEvent(kind='flow', symbol='BTC/USDT', timestamp=120, available_at=119,
                    close=100, quote_volume=10, taker_buy_quote_volume=5)
    with pytest.raises(ValueError):
        ReplayEvent(kind='price', symbol='BTC/USDT', timestamp=120, available_at=120,
                    close=100, quote_volume=10)
    times = np.arange(60, 10 * 86400 + 1, 60)
    bars = {'BTC/USDT': np.ones((len(times), 3))}
    events = list(flow_events(times, bars, 9 * 86400, 9 * 86400 + 120))
    assert events[-1].timestamp == events[-1].available_at == 9 * 86400 + 60


@pytest.mark.asyncio
async def test_missing_or_stale_flow_blocks_entries_and_cancels_pending():
    clock, tracker, manager, hybrid, step = await setup_hybrid(
        decision_interval_seconds=300, entry_quality={'enabled': True})
    start = clock[0]
    await step(start, 99)
    assert not manager.has_pending
    ready(hybrid.entry_quality, start + 300)
    await step(start + 300, 99)
    assert manager.has_pending
    await step(start + 361, 99)
    assert not manager.has_pending
    assert tracker.snapshot().cash == 100000


@pytest.mark.asyncio
async def test_missing_flow_does_not_block_risk_exit():
    clock, tracker, manager, hybrid, step = await setup_hybrid(entry_quality={'enabled': True})
    start = clock[0]
    ready(hybrid.entry_quality, start)
    await step(start, 99)
    await step(start + 30, 98.9)
    assert hybrid.holdings
    hybrid.entry_quality.snapshots.clear()
    await step(start + 14430, 99)
    assert not hybrid.holdings
    assert tracker.get_position('BTC/USDT').quantity == 0


@pytest.mark.asyncio
async def test_replay_routes_flow_without_treating_it_as_an_execution_price():
    config, events = fixture(mode='price_only')
    config['strategy']['entry_quality'] = {'enabled': True}
    start = config['replay']['start_at']
    context = [ReplayEvent(kind='flow', symbol='BTC/USDT', timestamp=start - (19-i)*60,
        available_at=start - (19-i)*60, close=100+i, quote_volume=100,
        taker_buy_quote_volume=40 if i < 15 else 60) for i in range(20)]
    result = await run_replay(config, sorted(events + context, key=lambda e: e.available_at))
    entries = [e for e in result['events'] if e['event'] == 'position_opened']
    assert entries and entries[0]['entry_price'] < 100
    assert result['summary']['completed_holds'] == 1


def test_markouts_use_fixed_post_fill_horizon_not_actual_exit():
    times = np.arange(0, 7400, 60)
    prices = (100 + times / 7200)[:, None]
    trade = dict(opened_at=0, cost=100, quantity=1, symbol='BTC/USDT')
    metrics = entry_metrics([trade], times, prices, ['BTC/USDT'])
    assert metrics['markout_120m_bps'] == pytest.approx(100)
    assert metrics['mae_120m_bps'] == 0
    assert metrics['mfe_120m_bps'] == pytest.approx(100)
    with pytest.raises(ValueError, match='Incomplete'):
        entry_metrics([trade], times[:10], prices[:10], ['BTC/USDT'])
