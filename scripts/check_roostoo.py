"""Read-only test-account check. No order placement or cancellation is implemented."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import aiohttp

from plugins.roostoo.auth import RoostooAuth
from plugins.roostoo.credentials import load_testing_credentials

BASE_URL = 'https://mock-api.roostoo.com'
READ_ONLY = {('GET', '/v3/serverTime'), ('GET', '/v3/exchangeInfo'),
             ('GET', '/v3/ticker'), ('GET', '/v3/balance'),
             ('GET', '/v3/pending_count'), ('POST', '/v3/query_order')}


def validate_quote(data):
    quote = data.get('Data', {}).get('BTC/USD', {})
    bid, ask = float(quote.get('MaxBid', 0)), float(quote.get('MinAsk', 0))
    return data.get('Success') is True and math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask


def validate_pending(data):
    count = data.get('TotalPending')
    if type(count) is not int or count < 0:
        return False
    return data.get('Success') is True or (count == 0 and
        data.get('ErrMsg') == 'no pending order under this account')


def validate_orders(data):
    rows = data.get('OrderMatched')
    if data.get('Success') is True:
        return isinstance(rows, list)
    return data.get('ErrMsg') == 'no order matched' and rows in (None, [])


async def check(path, out):
    key, secret = load_testing_credentials(path)
    auth = RoostooAuth(key, secret)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'general_portfolio_testing', 'orders_sent': 0,
              'base_url': BASE_URL, 'checks': []}
    out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    # Thirteen seconds between starts gives at most five requests per rolling minute.
    last_start = 0.
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), trust_env=True) as session:
        async def request(method, endpoint, params=None, signed=False):
            nonlocal last_start
            if (method, endpoint) not in READ_ONLY:
                raise ValueError('Read-only endpoint allowlist violation')
            await asyncio.sleep(max(0, 13 - (time.monotonic() - last_start)))
            last_start = time.monotonic()
            params, headers = dict(params or {}), {}
            if signed:
                params['timestamp'] = str(auth.get_timestamp())
                headers, encoded = auth.sign(params)
                headers['Content-Type'] = 'application/x-www-form-urlencoded'
            kwargs = {'params': params} if method == 'GET' else {'data': encoded}
            async with session.request(method, BASE_URL + endpoint, headers=headers,
                                       allow_redirects=False, **kwargs) as response:
                data = await response.json(content_type=None)
                if not isinstance(data, dict):
                    raise ValueError('Malformed API object')
                item = {'endpoint': endpoint, 'http_status': response.status,
                        'success': response.status == 200 and data.get('Success', True)}
                report['checks'].append(item)
                # Never retain raw server errors, headers, auth values or wallet balances.
                if response.status != 200:
                    save()
                    raise RuntimeError('API request rejected; see sanitized status receipt')
                return data, item
        try:
            data, item = await request('GET', '/v3/serverTime')
            item['clock_drift_ms'] = abs(auth.get_timestamp() - int(data.get('ServerTime', 0)))
            if item['clock_drift_ms'] > 60000:
                raise RuntimeError('Server clock difference exceeds authenticated request tolerance')
            data, item = await request('GET', '/v3/exchangeInfo')
            item['trade_pair_count'] = len(data.get('TradePairs', {}))
            item['is_running'] = data.get('IsRunning')
            if not item['trade_pair_count'] or item['is_running'] is not True:
                raise RuntimeError('Exchange is inactive or symbol definitions are missing')
            data, item = await request('GET', '/v3/balance', signed=True)
            if not data.get('Success'):
                raise RuntimeError('Authenticated balance check unsuccessful')
            item['wallet_asset_count'] = len(data.get('SpotWallet', {}))
            data, item = await request('GET', '/v3/ticker', {'pair': 'BTC/USD'}, signed=True)
            item['valid_bid_ask'] = validate_quote(data)
            if not item['valid_bid_ask']:
                raise RuntimeError('Invalid quote response')
            data, item = await request('GET', '/v3/pending_count', signed=True)
            item['pending_count'] = data.get('TotalPending')
            item['valid_response'] = validate_pending(data)
            if not item['valid_response']:
                raise RuntimeError('Invalid pending-order response')
            data, item = await request('POST', '/v3/query_order', {'pending_only': 'TRUE'}, signed=True)
            item['pending_rows'] = len(data.get('OrderMatched', []))
            item['empty_result'] = data.get('ErrMsg') == 'no order matched'
            item['valid_response'] = validate_orders(data)
            if not item['valid_response']:
                raise RuntimeError('Invalid order-query response')
            report['status'] = 'authenticated_read_only_checks_completed'
        except Exception as exc:
            report['status'] = 'incomplete'
            report['error_type'] = type(exc).__name__
        save()
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', type=Path, default=Path('.secrets/roostoo-test.json'))
    parser.add_argument('--output', type=Path, default=Path('logs/roostoo_readonly.json'))
    args = parser.parse_args()
    report = asyncio.run(check(args.credentials, args.output))
    raise SystemExit(0 if report['status'] == 'authenticated_read_only_checks_completed' else 1)
