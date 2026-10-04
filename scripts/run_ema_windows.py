"""Frozen price-policy study across independent flat-start 14-day windows.

Uses the existing hybrid coordinator, order lifecycle and request budget. There
are deliberately no model forecasts: E38 saw earlier history during training.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import time

import numpy as np
import yaml

from scripts.replay_hybrid import ReplayEvent, run_replay
from scripts.replay_research_bundle import extra_metrics
from scripts.run_ema_pullback import policy

CASES = [('strict_maker_2bps', 'strict', True, 2.),
         ('soft_maker_2bps', 'soft', True, 2.),
         ('soft_maker_5bps', 'soft', True, 5.),
         ('soft_taker', 'soft', False, 2.)]
UTC8 = timezone(timedelta(hours=8))


def windows(first, count):
    start = int(datetime.strptime(first, '%Y-%m-%d').replace(tzinfo=UTC8).timestamp())
    if count < 1:
        raise ValueError('At least one window required')
    return [(start + i * 14 * 86400, start + (i + 1) * 14 * 86400) for i in range(count)]


def load_prices(root, symbols):
    times, closes = None, []
    for symbol in symbols:
        with np.load(root / f'{symbol.replace("/", "")}.npz', allow_pickle=False) as source:
            current = source['times'] + 60
            close = source['bars'][:, 4]
        if times is not None and not np.array_equal(current, times):
            raise ValueError('Asset time grids do not align')
        if np.any(np.diff(current) != 60) or not np.isfinite(close).all() or np.any(close <= 0):
            raise ValueError('Missing minute or invalid close')
        times = current
        closes.append(close)
    return times, np.stack(closes, axis=1)


def minute_events(times, prices, symbols, start, end):
    warmup = start - 8 * 86400
    if times[0] > warmup or times[-1] < end:
        raise ValueError('Incomplete warmup or evaluation coverage')
    left, right = np.searchsorted(times, [warmup, end])
    for idx in range(left, right):
        stamp = int(times[idx])
        for j, symbol in enumerate(symbols):
            # A complete minute close first becomes visible at its end boundary.
            yield ReplayEvent(kind='price', symbol=symbol, timestamp=stamp - 1,
                              available_at=stamp, close=float(prices[idx, j]))


def window_config(template, start, end, trend, maker, penetration):
    manifest = dict(model_version='not_used_price_only', preprocessing_version='not_used_price_only',
                    heads=template['forecast']['heads'])
    config = policy(template, manifest, dict(replay_start=start, replay_end=end),
                    maker, penetration, 'price_only')
    config['strategy']['ema_pullback'].update(trend_mode=trend, downtrend_size_multiplier=.5)
    config['strategy']['model_exit_enabled'] = False
    config['replay'].update(timer_seconds=60, entry_cutoff_seconds=7800, closeout_seconds=600,
                            events_path='completed_minute_close_archive')
    return config


def run_window(task):
    root, out, index, start, end, template = task
    symbols = list(template['strategy']['rules'])
    times, prices = load_prices(root, symbols)
    records = []
    for name, trend, maker, penetration in CASES:
        tick = time.time()
        config = window_config(template, start, end, trend, maker, penetration)
        result = asyncio.run(run_replay(config, minute_events(times, prices, symbols, start, end)))
        summary = result['summary']
        summary.update(extra_metrics(result))
        flat = not any(summary[k] for k in ('positions', 'pending_orders', 'venue_fills_not_yet_reconciled'))
        summary['flat_start_end_verified'] = flat and not summary['accounting_provisional']
        # Keep an invalid window in the report; never remove it to improve metrics.
        summary['valid_window'] = summary['flat_start_end_verified']
        result['window_index'] = index
        result['case'] = name
        with gzip.open(out / f'{index:02d}_{name}.json.gz', 'xt') as stream:
            json.dump(result, stream, allow_nan=False)
        row = dict(window=index, case=name, start=start, end=end,
                   start_utc8=datetime.fromtimestamp(start, UTC8).isoformat(),
                   end_exclusive_utc8=datetime.fromtimestamp(end, UTC8).isoformat(),
                   block='earlier_17' if index < 17 else 'later_8',
                   wall_seconds=time.time() - tick, summary=summary)
        records.append(row)
        print(json.dumps({'window': index, 'case': name, 'net_return': summary['net_return'],
                          'trades': summary['completed_holds'], 'active_days': summary['active_fill_days_utc8'],
                          'flat': summary['flat_start_end_verified']}), flush=True)
    return records


def aggregate(records):
    output = {}
    for name, *_ in CASES:
        output[name] = {}
        for block in ('all', 'earlier_17', 'later_8'):
            rows = [r['summary'] for r in records if r['case'] == name and (block == 'all' or r['block'] == block)]
            if not rows:
                continue
            returns = [r['net_return'] for r in rows]
            total_trades = sum(r['completed_holds'] for r in rows)
            pnl = [p for r in rows for p in r['closed_trade_net_pnl_values']]
            output[name][block] = dict(
                windows=len(rows), valid_windows=sum(r['valid_window'] for r in rows),
                mean_net_return=float(np.mean(returns)), median_net_return=float(np.median(returns)),
                p10_net_return=float(np.quantile(returns, .1)), worst_net_return=min(returns),
                best_net_return=max(returns), positive_window_fraction=sum(v > 0 for v in returns) / len(rows),
                activity_pass_fraction=sum(r['meets_eight_active_days'] for r in rows) / len(rows),
                profitable_and_active_fraction=sum(r['meets_eight_active_days'] and r['net_return'] > 0 for r in rows) / len(rows),
                mean_active_days=float(np.mean([r['active_fill_days_utc8'] for r in rows])),
                total_completed_trades=total_trades, mean_trades_per_window=total_trades / len(rows),
                worst_window_drawdown=max(r['max_drawdown'] for r in rows),
                total_fees=sum(r['fee_total_usdt'] for r in rows),
                total_net_pnl=sum(pnl), gross_same_fills_pnl=sum(pnl) + sum(r['fee_total_usdt'] for r in rows),
                pooled_trade_win_fraction=sum(p > 0 for p in pnl) / len(pnl) if pnl else None,
                pooled_trade_profit_factor=sum(max(p, 0) for p in pnl) / -sum(min(p, 0) for p in pnl) if any(p < 0 for p in pnl) else None,
                mean_time_weighted_exposure=float(np.mean([r['time_weighted_exposure'] for r in rows])))
    return output


def run(root, out, first='2025-09-10', count=25, workers=2):
    out.mkdir(parents=True, exist_ok=False)
    template = yaml.safe_load(Path('config/hybrid.yaml').read_text())
    spans = windows(first, count)
    report = dict(study='E39_multiwindow_EMA', status='frozen_before_replay',
        new_training=False, model_forecasts_used=False, selection_performed=False,
        independent_holdout=False, data_manifest=json.loads((root / 'manifest.json').read_text()),
        primary='soft_maker_2bps', reset='100000 cash, zero positions, eight-day indicator warmup',
        split='Fixed first 17 / last 8 descriptive blocks; no candidate selection.',
        limitations=['Historical price-policy study, NOT out-of-sample model performance.',
                     'Minute closes do not reproduce second-level stops or Roostoo bid/ask/queues.',
                     'Only subsequent minute closes penetrating a limit may maker-fill; no same-bar OHLC fills.',
                     'No new entries in final 130 minutes; budgeted cash closeout starts final 10 minutes.',
                     'Non-overlapping windows are not statistically independent market regimes.',
                     'Price/model-policy design used previously inspected September 2026 diagnostics.',
                     'Cached public-feed HTTP is excluded from five-per-minute execution budget.'],
        configs={name: window_config(template, *spans[0], trend, maker, p)
                 for name, trend, maker, p in CASES}, windows=spans, records=[])
    (out / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    tasks = [(root, out, idx, start, end, template) for idx, (start, end) in enumerate(spans)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(run_window, tasks):
            report['records'].extend(rows)
            report['aggregate'] = aggregate(report['records'])
            report['status'] = 'running'
            (out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    report['status'] = 'complete'
    report['aggregate'] = aggregate(report['records'])
    (out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--first', default='2025-09-10')
    parser.add_argument('--count', type=int, default=25)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.data, args.out, args.first, args.count, args.workers)
