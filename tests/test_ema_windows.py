import copy
import io
import zipfile
import gzip
import json

import numpy as np
import pytest
import yaml
from pathlib import Path

from scripts.download_minute_history import parse_archive, months_between
from scripts.run_ema_windows import windows, minute_events, window_config
from scripts.replay_hybrid import ReplayEvent, run_replay
from strategy.ema_pullback import EmaPullbackHistory
from tests.test_hybrid_replay import fixture
from tests.test_hybrid import setup_hybrid
from scripts.summarize_ema_windows import audit_run


def archive(bars):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        text = io.StringIO()
        np.savetxt(text, bars, delimiter=',', fmt='%.8f')
        z.writestr('test.csv', text.getvalue())
    return stream.getvalue()


@pytest.mark.parametrize('scale', [1000, 1_000_000])
def test_archive_units_ohlc_and_missing_rows(scale):
    start = 1735689600
    bars = np.array([[t * scale, 100, 102, 99, 101, 1, (t + 60) * scale - 1, 100, 5, .5, 50, 0]
                     for t in [start, start + 60]], dtype=float)
    times, _ = parse_archive(archive(bars), start, start + 120)
    assert times.tolist() == [start, start + 60]
    with pytest.raises(ValueError, match='shape'):
        parse_archive(archive(bars[:1]), start, start + 120)
    broken = bars.copy()
    broken[1, 0] = broken[0, 0]
    with pytest.raises(ValueError, match='timestamps'):
        parse_archive(archive(broken), start, start + 120)
    broken = bars.copy()
    broken[0, 2] = 90
    with pytest.raises(ValueError, match='OHLC'):
        parse_archive(archive(broken), start, start + 120)


def test_months_windows_and_closed_bar_availability():
    assert months_between('2025-12', '2026-02') == ['2025-12', '2026-01', '2026-02']
    spans = windows('2025-09-10', 25)
    assert all(end - start == 14 * 86400 for start, end in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(24))
    start, end = spans[0]
    times = np.arange(start - 8 * 86400, end + 1, 60)
    prices = np.ones((len(times), 1)) * 100
    events = list(minute_events(times, prices, ['BTC/USDT'], start, start + 120))
    assert events[-1].available_at == start + 60
    assert all(e.timestamp == e.available_at - 1 for e in events)
    changed = prices.copy()
    changed[times >= start + 60] = 1000
    other = list(minute_events(times, changed, ['BTC/USDT'], start, start + 120))
    assert events[:-1] == other[:-1]
    with pytest.raises(ValueError, match='coverage'):
        list(minute_events(times[1:], prices[1:], ['BTC/USDT'], start, end))


def test_soft_trend_removes_veto_but_keeps_recovery_and_sizes_down():
    strict = EmaPullbackHistory({'warmup_bars': 13})
    for i in range(1, 31):
        strict.observe('BTC', i * 300 - 1, 150 - i, i * 300)
    strict.observe('BTC', 9299, 121, 9300)
    soft = copy.deepcopy(strict)
    soft.trend_mode = 'soft'
    assert strict.entry_check('BTC', 121, 9300, 20) == 'trend_not_up'
    assert soft.entry_check('BTC', 121, 9300, 20) == 'qualified'
    assert soft.size_multiplier('BTC', 9300) == .5
    assert strict.size_multiplier('BTC', 9300) == 1
    soft.observe('BTC', 9599, 120, 9600)
    assert soft.entry_check('BTC', 120, 9600, 20) == 'no_completed_bar_recovery'
    assert soft.size_multiplier('BTC', 9900) == 0
    with pytest.raises(ValueError):
        EmaPullbackHistory({'trend_mode': 'anything'})


@pytest.mark.asyncio
async def test_budgeted_terminal_flatten_no_inherited_inventory():
    config, original = fixture(mode='price_only')
    start = config['replay']['start_at']
    events = [e for e in original if e.kind == 'hourly']
    events += [ReplayEvent(kind='price', symbol='BTC/USDT', timestamp=t, available_at=t,
                          close=99 if t == start else 98.9) for t in range(start, start + 900, 30)]
    config['replay'].update(end_at=start + 900, entry_cutoff_seconds=300, closeout_seconds=300)
    result = await run_replay(config, events)
    assert result['summary']['completed_holds'] == 1
    assert not result['summary']['positions']
    assert not result['summary']['pending_orders']
    assert not result['summary']['venue_fills_not_yet_reconciled']
    assert any(e.get('reason') == 'window_closeout' for e in result['events'] if e['event'] == 'position_closed')
    assert all(e['timestamp'] < start + 600 for e in result['events'] if e['event'] == 'entry_intent')
    again = await run_replay(config, events)
    assert again['summary']['final_nav'] == result['summary']['final_nav']
    assert again['summary']['initial_positions'] == []
    assert result['summary']['fees_by_role_and_asset']


def test_price_policy_does_not_claim_model_and_preserves_hold_risk():
    template = yaml.safe_load(Path('config/hybrid.yaml').read_text())
    config = window_config(template, 864000, 864000 + 14 * 86400, 'soft', True, 2)
    assert config['strategy']['fusion']['mode'] == 'price_only'
    assert config['forecast']['model_version'] == 'not_used_price_only'
    assert config['strategy']['max_holding_seconds'] == 7200
    assert config['execution']['max_requests_per_minute'] == 5
    assert config['strategy']['stop_loss_pct'] == .03


@pytest.mark.asyncio
async def test_soft_size_reaches_actual_order_quantity():
    from types import SimpleNamespace
    clock, tracker, manager, hybrid, step = await setup_hybrid()
    await step(clock[0], 100, allow_entries=False)
    hybrid.ema = SimpleNamespace(size_multiplier=lambda *args: .5)
    await hybrid._enter('BTC/USDT', 100, 101)
    order = manager.for_symbol('BTC/USDT')[0]
    assert order.quantity == 50  # 5,000 notional rather than 10,000.


@pytest.mark.asyncio
async def test_cash_auditor_accepts_closed_ledger_and_rejects_open_window(tmp_path):
    config, events = fixture()
    result = await run_replay(config, events)
    path = tmp_path / 'closed.json.gz'
    with gzip.open(path, 'wt') as stream:
        json.dump(result, stream)
    trades, error = audit_run(path)
    assert len(trades) == 1 and abs(error) < 1e-6
    config['replay']['end_at'] = config['replay']['start_at'] + 85
    result = await run_replay(config, events)
    with gzip.open(path, 'wt') as stream:
        json.dump(result, stream)
    with pytest.raises(ValueError, match='audit failed'):
        audit_run(path)
