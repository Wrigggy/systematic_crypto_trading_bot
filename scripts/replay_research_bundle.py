"""Integrate a real versioned multi-horizon ensemble into the offline strategy.

Consumes inert NPZ/JSON exports, never imports a model or connects to a venue.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from plugins.model_inference.forecasts import ForecastPacket
from scripts.replay_hybrid import ReplayEvent, run_replay


def bundle_events(price_path, bundle_path, symbols, include_forecasts=True):
    with np.load(price_path, allow_pickle=False) as source:
        price = {k: source[k] for k in source.files}
    declared = price['symbols'].tolist()
    if set(symbols) - set(declared):
        raise ValueError('Missing traded-asset prices.')

    def prices():
        for i, stamp in enumerate(price['price_times']):
            for symbol in symbols:
                yield ReplayEvent(kind='price', symbol=symbol, timestamp=int(stamp) - 1,
                    available_at=int(stamp), close=float(price['prices'][i, declared.index(symbol)]))

    def hours():
        for i, stamp in enumerate(price['hourly_times']):
            for symbol in symbols:
                yield ReplayEvent(kind='hourly', symbol=symbol, timestamp=int(stamp),
                    available_at=int(stamp) + 1,
                    close=float(price['hourly_prices'][i, declared.index(symbol)]))

    def forecasts():
        manifest = json.loads((bundle_path / 'bundle.json').read_text())
        with np.load(bundle_path / 'forecasts.npz', allow_pickle=False) as source:
            values, times, assets = source['prediction'], source['times'], source['symbols'].tolist()
        if values.shape != (len(times), len(assets), len(manifest['heads'])):
            raise ValueError('Forecast dimensions differ from the declared bundle.')
        if np.any(np.diff(times) != 60) or not np.isfinite(values).all():
            raise ValueError('Forecasts must be finite and regularly spaced.')
        for i, stamp in enumerate(times):
            for symbol in symbols:
                sid = assets.index(symbol.replace('/', ''))
                packet = ForecastPacket(schema_version=1, symbol=symbol, as_of=int(stamp),
                    data_cutoff=int(stamp) - 1, available_at=int(stamp) + 1,
                    valid_until=int(stamp) + 121, eligible_after=manifest['eligible_after'],
                    model_version=manifest['model_version'],
                    preprocessing_version=manifest['preprocessing_version'],
                    heads=[{k: h[k] for k in ('name', 'window_start_seconds', 'window_end_seconds', 'target')}
                           | {'prediction': float(values[i, sid, j])}
                           for j, h in enumerate(manifest['heads'])])
                yield ReplayEvent(kind='forecast', symbol=symbol, timestamp=int(stamp),
                                  available_at=int(stamp) + 1, forecast=packet)

    streams = [prices(), hours()]
    if include_forecasts:
        streams.append(forecasts())
    return heapq.merge(*streams, key=lambda e: e.available_at)


def configure(template, manifest, start, end, mode, maker, model_exit, penetration):
    cfg = copy.deepcopy(template)
    cfg['strategy']['decision_interval_seconds'] = 300
    cfg['strategy']['fusion']['mode'] = mode
    cfg['strategy']['maker_preferred'] = maker
    cfg['strategy']['model_exit_enabled'] = model_exit
    cfg['forecast'].update({k: manifest[k] for k in ('model_version', 'preprocessing_version', 'heads')})
    cfg['execution']['limit_penetration_bps'] = penetration
    cfg['replay'].update(start_at=start, end_at=end, timer_seconds=5, events_path='inert_research_bundle')
    return cfg


def extra_metrics(result):
    events = result['events']
    orders = {}
    for e in events:
        if e['event'] == 'order_event':
            order = e['order']
            orders[order['order_id']] = order
    filled = [o for o in orders.values() if o['filled_quantity'] > 0]
    holds = [e['holding_seconds'] for e in events if e['event'] == 'position_closed']
    positions, closed_pnl = {}, []
    for order in sorted(filled, key=lambda o: o['filled_at'] or ''):
        symbol = order['symbol']
        state = positions.setdefault(symbol, {'quantity': 0., 'net_cashflow': 0.})
        sign = 1 if order['side'] == 'BUY' else -1
        quantity = order['filled_quantity']
        state['quantity'] += sign * quantity
        state['net_cashflow'] -= sign * quantity * order['filled_price'] + (order['commission'] or 0.)
        if abs(state['quantity']) <= 1e-8:
            closed_pnl.append(state['net_cashflow'])
            del positions[symbol]
    gains = sum(max(v, 0) for v in closed_pnl)
    losses = -sum(min(v, 0) for v in closed_pnl)
    return {'order_count': len(orders), 'filled_order_count': len(filled),
        'order_fill_fraction': len(filled) / len(orders) if orders else None,
        'order_status_counts': dict(Counter(o['status'] for o in orders.values())),
        'filled_roles': dict(Counter(o['liquidity'] for o in filled)),
        'exit_reasons': dict(Counter(e.get('reason', 'unknown') for e in events if e['event'] == 'position_closed')),
        'hold_quantiles_seconds': dict(zip(['p10', 'p50', 'p90'], np.quantile(holds, [.1, .5, .9]).tolist())) if holds else {},
        'closed_trade_win_fraction': sum(v > 0 for v in closed_pnl) / len(closed_pnl) if closed_pnl else None,
        'closed_trade_profit_factor': gains / losses if losses > 0 else None,
        'closed_trade_net_pnl': sum(closed_pnl),
        'closed_trade_net_pnl_values': closed_pnl,
        'fee_total_usdt': sum(o['commission'] or 0 for o in filled),
        'candidate_count': sum(e['event'] == 'price_candidate' for e in events),
        'candidate_fusion_reasons': dict(Counter(e['fusion']['reason'] for e in events if e['event'] == 'price_candidate'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/hybrid.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['price_only', 'combined', 'model_filter'], default='combined')
    parser.add_argument('--taker', action='store_true')
    parser.add_argument('--model-exit', action='store_true')
    parser.add_argument('--penetration', type=float, default=2.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.bundle / 'bundle.json').read_text())
    provenance = json.loads(args.prices.with_suffix('.json').read_text())
    cfg = configure(yaml.safe_load(args.config.read_text()), manifest, provenance['replay_start'],
                    provenance['replay_end'], args.mode, not args.taker, args.model_exit, args.penetration)
    events = bundle_events(args.prices, args.bundle, list(cfg['strategy']['rules']), args.mode != 'price_only')
    result = asyncio.run(run_replay(cfg, events))
    result['summary'].update(extra_metrics(result))
    result['provenance'] = {'price_source': provenance, 'bundle': manifest,
        'independent_holdout': False, 'learned_combiner': False,
        'market_data_source': 'Cached Binance observations, not reconstructed Roostoo quotes.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'xt') as stream:
        json.dump(result, stream, allow_nan=False)
    args.output.with_suffix('.summary.json').write_text(json.dumps(result['summary'], indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
