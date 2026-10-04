"""E40: fixed entry-quality repair, matched window controls and eligible forecasts.

No tuning, model fitting, cloud compute or live trading. Freeze plan before replay.
"""
import argparse
import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import heapq
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.replay_hybrid import ReplayEvent, run_replay
from scripts.replay_research_bundle import bundle_events, extra_metrics
from scripts.run_ema_pullback import policy
from scripts.run_ema_windows import load_prices, minute_events, windows, window_config

CASES = [('baseline', False, 2.), ('quality_maker', True, 2.), ('quality_maker_5bps', True, 5.)]


def apply_quality(config, overlay, flow=True):
    cfg = copy.deepcopy(config)
    cfg['strategy']['fusion'].update(overlay['strategy']['fusion'])
    cfg['strategy']['entry_quality'] = dict(overlay['strategy']['entry_quality'], enabled=flow)
    return cfg


def load_flow(root, symbols):
    result, reference = {}, None
    for symbol in symbols:
        with np.load(root / f'{symbol.replace("/", "")}.npz', allow_pickle=False) as source:
            times = source['times'] + 60
            bars = source['bars']
        if reference is not None and not np.array_equal(times, reference):
            raise ValueError('Flow asset timestamps differ')
        if (np.any(np.diff(times) != 60) or not np.isfinite(bars).all()
                or np.any(bars[:, 7] < 0) or np.any(bars[:, 10] < 0)
                or np.any(bars[:, 10] > bars[:, 7]) or np.any(bars[:, 4] <= 0)):
            raise ValueError('Invalid flow grid, price or taker quote volume')
        reference = times
        result[symbol] = bars[:, [4, 7, 10]]
    return reference, result


def flow_events(times, values, start, end):
    if times[0] > start - 8 * 86400 or times[-1] < end:
        raise ValueError('Incomplete causal flow coverage')
    left, right = np.searchsorted(times, [start - 8 * 86400, end])
    for i in range(left, right):
        stamp = int(times[i])
        for symbol, bars in values.items():
            close, quote, buy = bars[i]
            yield ReplayEvent(kind='flow', symbol=symbol, timestamp=stamp, available_at=stamp,
                              close=close, quote_volume=quote, taker_buy_quote_volume=buy)


def save_result(out, name, result):
    result['summary'].update(extra_metrics(result))
    result['summary']['quality_rejections'] = dict(Counter(e['reason'] for e in result['events']
        if e['event'] == 'quality_entry_rejected'))
    result['summary']['flat_start_end_verified'] = not any(result['summary'][k] for k in
        ('positions', 'pending_orders', 'venue_fills_not_yet_reconciled', 'accounting_provisional'))
    with gzip.open(out / (name + '.json.gz'), 'xt') as stream:
        json.dump(result, stream, allow_nan=False)
    print(json.dumps(dict(name=name, net_return=result['summary']['net_return'],
        trades=result['summary']['completed_holds'], active_days=result['summary']['active_fill_days_utc8'])), flush=True)
    return dict(name=name, policies={'fixed_policy': result['summary']})


def run_window(task):
    root, out, index, start, end, template, overlay = task
    symbols = list(template['strategy']['rules'])
    times, prices = load_prices(root, symbols)
    ftimes, flow = load_flow(root, symbols)
    rows = []
    for name, quality, penetration in CASES:
        cfg = window_config(template, start, end, 'soft', True, penetration)
        if quality:
            cfg = apply_quality(cfg, overlay)
        streams = [minute_events(times, prices, symbols, start, end)]
        if quality:
            streams.append(flow_events(ftimes, flow, start, end))
        result = asyncio.run(run_replay(cfg, heapq.merge(*streams, key=lambda e: e.available_at)))
        row = save_result(out, f'window_{index:02d}_{name}', result)
        row.update(window=index, case=name, start=start, end=end)
        rows.append(row)
    return rows


def aggregate(records):
    output = {}
    for case, *_ in CASES:
        output[case] = {}
        for block in ('all', 'earlier_17', 'later_8'):
            rows = [r['policies']['fixed_policy'] for r in records if r.get('case') == case
                    and (block == 'all' or (r['window'] < 17) == (block == 'earlier_17'))]
            if not rows:
                continue
            net = [r['net_return'] for r in rows]
            pnl = [v for r in rows for v in r['closed_trade_net_pnl_values']]
            fees = sum(r['fee_total_usdt'] for r in rows)
            turnover = sum(r['turnover_notional'] for r in rows)
            output[case][block] = dict(windows=len(rows), mean_return=float(np.mean(net)),
                median_return=float(np.median(net)), worst_return=min(net),
                positive_windows=sum(v > 0 for v in net),
                activity_pass_windows=sum(r['meets_eight_active_days'] for r in rows),
                profitable_and_active_windows=sum(r['net_return'] > 0 and r['meets_eight_active_days'] for r in rows),
                trades=sum(r['completed_holds'] for r in rows),
                mean_active_days=float(np.mean([r['active_fill_days_utc8'] for r in rows])),
                mean_exposure=float(np.mean([r['time_weighted_exposure'] for r in rows])),
                worst_drawdown=max(r['max_drawdown'] for r in rows), fees=fees,
                gross_same_fill_pnl=sum(pnl) + fees, net_pnl=sum(pnl),
                net_pnl_per_trade=float(np.mean(pnl)) if pnl else None,
                net_bps_per_roundtrip_notional=sum(pnl) / (turnover / 2) * 10000 if turnover else None,
                profit_factor=sum(max(v, 0) for v in pnl) / -sum(min(v, 0) for v in pnl) if any(v < 0 for v in pnl) else None)
    return output


async def model_replays(bundle, prices, flow_root, out, template, overlay):
    manifest = json.loads((bundle / 'bundle.json').read_text())
    provenance = json.loads(prices.with_suffix('.json').read_text())
    start, end = provenance['replay_start'], provenance['replay_end']
    if start < manifest['eligible_after']:
        raise ValueError('Model not eligible for requested replay')
    symbols = list(template['strategy']['rules'])
    times, values = load_flow(flow_root, symbols)
    cases = [('baseline', False, False, True, 2.), ('fusion_fix', True, False, True, 2.),
             ('quality_maker', True, True, True, 2.), ('quality_maker_5bps', True, True, True, 5.),
             ('quality_taker', True, True, False, 2.)]
    configs = {}
    for name, quality, flow, maker, penetration in cases:
        cfg = policy(template, manifest, provenance, maker, penetration)
        cfg['strategy']['ema_pullback'].update(trend_mode='soft', downtrend_size_multiplier=.5)
        cfg['replay'].update(entry_cutoff_seconds=7800, closeout_seconds=600)
        configs[name] = apply_quality(cfg, overlay, flow) if quality else cfg
    report = dict(status='frozen_before_replay', independent_holdout=False, new_training=False,
                  configs=configs, model_version=manifest['model_version'], records=[])
    (out / 'model_plan.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, _, flow, _, _ in cases:
        streams = [bundle_events(prices, bundle, symbols)]
        if flow:
            streams.append(flow_events(times, values, start, end))
        result = await run_replay(configs[name], heapq.merge(*streams, key=lambda e: e.available_at))
        report['records'].append(save_result(out, 'model_' + name, result))
        report['status'] = 'complete' if len(report['records']) == len(cases) else 'running'
        (out / 'model_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--flow-data', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--prices', type=Path)
    parser.add_argument('--model-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    template = yaml.safe_load(Path('config/hybrid.yaml').read_text())
    overlay = yaml.safe_load(Path('config/entry_quality_v1.yaml').read_text())
    spans = windows('2025-09-10', 25)
    report = dict(study='E40_entry_quality', status='frozen_before_replay', overlay=overlay,
        cases=CASES, windows=spans, new_training=False, independent_holdout=False,
        selection_performed=False, primary='quality_maker', records=[],
        limitation='Historical windows are price-only. Eligible-date model replay is separate. No parameter sweep or fresh holdout.',
        representative_config=apply_quality(window_config(template, *spans[0], 'soft', True, 2.), overlay))
    (args.out / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.model_only:
        tasks = [(args.data, args.out, i, *span, template, overlay) for i, span in enumerate(spans)]
        with ProcessPoolExecutor(max_workers=2) as pool:
            for rows in pool.map(run_window, tasks):
                report['records'].extend(rows)
                report['aggregate'] = aggregate(report['records'])
                report['status'] = 'running'
                (args.out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        report['status'] = 'complete'
        (args.out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.bundle:
        asyncio.run(model_replays(args.bundle, args.prices, args.flow_data, args.out, template, overlay))


if __name__ == '__main__':
    main()
