"""Describe frozen EMA opportunity coverage; never optimize or place orders."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from strategy.ema_pullback import EmaPullbackHistory


def diagnose(prices, comparison):
    report = json.loads(comparison.read_text())
    config = report['configs']['primary_maker']['strategy']
    ema = EmaPullbackHistory(config['ema_pullback'])
    provenance = report['provenance']
    begin, end = provenance['replay_start'], provenance['replay_end']
    counts = {s: Counter() for s in config['rules']}
    first, last = {}, {}
    with np.load(prices, allow_pickle=False) as data:
        symbols = data['symbols'].tolist()
        grid_times, grid_prices = data['price_times'], data['prices']
        for i, timestamp in enumerate(grid_times):
            timestamp = int(timestamp)
            if timestamp % 300 or timestamp >= end:
                continue
            for symbol in counts:
                price = float(grid_prices[i, symbols.index(symbol)])
                ema.observe(symbol, timestamp - 1, price, timestamp)
                if timestamp < begin:
                    continue
                first.setdefault(symbol, price)
                last[symbol] = price
                counts[symbol]['decision_grid_observations'] += 1
                bands = ema.bands(symbol, timestamp)
                if bands is None:
                    counts[symbol]['unready'] += 1
                    continue
                middle, std = bands
                state = ema.states[symbol]
                trend = state.fast > state.slow and state.fast > state.fast_history[0]
                oversold = price < middle - config['rules'][symbol]['entry_sigma'] * std
                counts[symbol]['trend_up'] += int(trend)
                counts[symbol]['oversold'] += int(oversold)
                counts[symbol]['oversold_and_trend_up'] += int(oversold and trend)
                if oversold:
                    reason = ema.entry_check(symbol, price, timestamp,
                        config['ema_pullback']['min_recovery_bps'])
                    counts[symbol]['oversold_' + reason] += 1
    return {'status': 'descriptive_only_no_policy_selection', 'independent_holdout': False,
        'grid_seconds': 300, 'start_at': begin, 'end_at': end,
        'excludes': ['model_filter', 'episode_reset', 'position_state', 'cash', 'order_fills'],
        'count_semantics': 'Repeated grid observations, not independent trade opportunities.',
        'per_asset': {s: {'counts': dict(c),
                          'first_to_last_grid_price_return': last[s] / first[s] - 1}
                      for s, c in counts.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prices', 'comparison', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.prices, args.comparison)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
