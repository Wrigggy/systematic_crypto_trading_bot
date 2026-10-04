import copy
import json

import pytest

from strategy.ema_pullback import EmaPullbackHistory
from strategy.fusion import SignalFusion
from plugins.roostoo.credentials import load_testing_credentials
from scripts.check_roostoo import READ_ONLY, validate_quote, validate_pending, validate_orders


def test_completed_boundaries_prefix_invariance_and_gap_reset():
    a = EmaPullbackHistory({'warmup_bars': 13})
    for i in range(1, 30):
        a.observe('BTC', i * 300 - 1, 100 + i * .1, i * 300)
    middle = a.bands('BTC', 8700)
    assert middle is not None
    a.observe('BTC', 8750, 500, 8751)
    assert a.bands('BTC', 8751) == middle  # Unfinished bar never enters EMA.
    b = copy.deepcopy(a)
    a.observe('BTC', 8999, 1000, 9000)
    assert b.bands('BTC', 8700) == middle
    assert b.bands('BTC', 9000) is None  # Missing completed boundary is stale.
    a.observe('BTC', 9599, 100, 9600)
    assert a.bands('BTC', 9600) is None
    assert a.resets['BTC'] == 1
    with pytest.raises(ValueError):
        a.observe('BTC', 9899, 100, 9700)


def test_recovery_and_trend_are_additional_conditions():
    a = EmaPullbackHistory({'warmup_bars': 13})
    for i in range(1, 30):
        a.observe('BTC', i * 300 - 1, 100 + i, i * 300)
    assert a.entry_check('BTC', 95, 8700, 20) == 'qualified'
    assert a.entry_check('BTC', 200, 8700, 20) == 'insufficient_recovery_distance'
    a.observe('BTC', 8999, 128, 9000)
    assert a.entry_check('BTC', 95, 9000, 20) == 'no_completed_bar_recovery'


def test_long_veto_is_explicit_and_defaults_unchanged():
    signal = {'composite': 3., 'long_support': -1.}
    assert not SignalFusion({}).evaluate(80, 100, 10, 1, signal).eligible
    assert SignalFusion({'require_long_support': False}).evaluate(80, 100, 10, 1, signal).eligible
    assert not SignalFusion({'require_long_support': False}).evaluate(80, 100, 10, 1, None).eligible


def test_credentials_require_private_permissions_and_test_scope(tmp_path):
    path = tmp_path / 'private.json'
    path.write_text(json.dumps({'account_scope': 'general_portfolio_testing',
                                'api_key': 'fakekey', 'api_secret': 'fakesecret'}))
    path.chmod(0o644)
    with pytest.raises(ValueError, match='0600'):
        load_testing_credentials(path)
    path.chmod(0o600)
    assert load_testing_credentials(path) == ('fakekey', 'fakesecret')
    assert ('POST', '/v3/place_order') not in READ_ONLY
    assert ('POST', '/v3/cancel_order') not in READ_ONLY
    assert ('POST', '/v6/short_open') not in READ_ONLY


def test_readonly_responses_fail_closed_except_documented_empty_results():
    assert validate_quote({'Success': True, 'Data': {'BTC/USD': {'MaxBid': 100, 'MinAsk': 101}}})
    assert not validate_quote({'Success': False, 'Data': {'BTC/USD': {'MaxBid': 100, 'MinAsk': 101}}})
    assert not validate_quote({'Success': True, 'Data': {'BTC/USD': {'MaxBid': 100, 'MinAsk': float('inf')}}})
    assert not validate_pending({'Success': False, 'TotalPending': 0, 'ErrMsg': 'auth failure'})
    assert validate_pending({'Success': False, 'TotalPending': 0, 'ErrMsg': 'no pending order under this account'})
    assert validate_orders({'Success': False, 'ErrMsg': 'no order matched'})
    assert not validate_orders({'Success': False, 'ErrMsg': 'auth failure'})
    assert not validate_orders({'Success': True})


def test_private_file_is_opt_in_and_does_not_override_environment(tmp_path, monkeypatch):
    from main import _apply_env_overrides
    for name in ('ROOSTOO_COMP_API_KEY', 'ROOSTOO_COMP_API_SECRET', 'ROOSTOO_API_KEY',
                 'ROOSTOO_API_SECRET', 'ROOSTOO_TEST_CREDENTIAL_FILE'):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / 'test.json'
    path.write_text(json.dumps({'account_scope': 'general_portfolio_testing',
                                'api_key': 'filekey', 'api_secret': 'filesecret'}))
    path.chmod(0o600)
    config = {}
    _apply_env_overrides(config)
    assert not config['roostoo'].get('api_key')
    monkeypatch.setenv('ROOSTOO_TEST_CREDENTIAL_FILE', str(path))
    _apply_env_overrides(config)
    assert config['roostoo']['api_key'] == 'filekey'
    monkeypatch.setenv('ROOSTOO_COMP_API_KEY', 'compkey')
    monkeypatch.setenv('ROOSTOO_COMP_API_SECRET', 'compsecret')
    _apply_env_overrides(config)
    assert config['roostoo']['api_key'] == 'compkey'
