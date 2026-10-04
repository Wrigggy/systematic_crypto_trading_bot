"""Smoke-check the actual executor using read operations only, never trade."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from execution.request_budget import RequestBudgetExceeded
from plugins.roostoo.credentials import load_testing_credentials
from plugins.roostoo.executor import RoostooExecutor


class ReadOnlyExecutor(RoostooExecutor):
    async def _request(self, method, endpoint, params, **kwargs):
        allowed = {'/v3/serverTime', '/v3/exchangeInfo', '/v3/pending_count',
                   '/v3/balance', '/v3/ticker'}
        if method != 'GET' or endpoint not in allowed:
            raise RuntimeError('Read-only executor endpoint violation')
        return await super()._request(method, endpoint, params, **kwargs)

    async def execute(self, order):
        raise RuntimeError('Order placement is disabled in the connection check')

    async def cancel(self, order_id, symbol):
        raise RuntimeError('Cancellation is disabled in the connection check')


async def check(credentials, output):
    key, secret = load_testing_credentials(credentials)
    executor = ReadOnlyExecutor({'api_key': key, 'api_secret': secret,
        'base_url': 'https://mock-api.roostoo.com', 'max_requests_per_minute': 5,
        'emergency_request_reserve': 1})
    receipt = {'scope': 'general_portfolio_testing', 'orders_sent': 0,
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'implementation': 'RoostooExecutor with read-only method guard', 'phase': 'startup'}

    async def perform():
        await executor.start()
        receipt['instrument_count'] = len(executor._pair_info)
        receipt['phase'] = 'balance'
        balances = await executor.get_balance()
        receipt['wallet_asset_count'] = len(balances)
        receipt['phase'] = 'quote'
        while True:
            try:
                quote = await executor.get_quote('BTC/USDT')
                receipt['quote_valid'] = 0 < quote['bid'] <= quote['ask']
                break
            except RequestBudgetExceeded:
                await asyncio.sleep(1)
        receipt['status'] = 'actual_executor_read_only_passed'
    try:
        await asyncio.wait_for(perform(), timeout=95)
    except Exception as exc:
        receipt.update(status='incomplete', error_type=type(exc).__name__)
    finally:
        await executor.stop()
        receipt['http_attempts'] = executor._budget.total
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', type=Path, default=Path('.secrets/roostoo-test.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)  # Only the sanitized receipt may leave this check.
    result = asyncio.run(check(args.credentials, args.output))
    raise SystemExit(0 if result['status'] == 'actual_executor_read_only_passed' else 1)
