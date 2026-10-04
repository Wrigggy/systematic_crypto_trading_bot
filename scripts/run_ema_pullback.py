"""Frozen 1-2h EMA pullback replay using eligible precomputed model forecasts."""
import argparse
import asyncio
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import yaml

from scripts.replay_hybrid import run_replay
from scripts.replay_research_bundle import bundle_events, configure, extra_metrics


def policy(template, manifest, provenance, maker=True, penetration=2., mode='combined'):
    cfg = configure(template, manifest, provenance['replay_start'], provenance['replay_end'],
                    mode, maker, True, penetration)
    cfg['strategy'].update(review_after_seconds=3600, max_holding_seconds=7200)
    cfg['strategy']['ema_pullback'] = {'enabled': True, 'fast_half_life_hours': 6,
        'slow_half_life_hours': 24, 'vol_half_life_hours': 24, 'warmup_bars': 2016,
        'min_recovery_bps': 20}
    cfg['strategy']['rules'] = {s: {'entry_sigma': .5, 'exit_z': 0.}
                                for s in template['strategy']['rules']}
    cfg['strategy']['fusion'].update(model_score_scale=1.5, entry_threshold=.75,
                                     require_long_support=False)
    cfg['forecast']['heads'] = [dict(h, weight=w) for h, w in
                                zip(cfg['forecast']['heads'], [.10, .25, .40, .20, .05])]
    # Review the intended holding horizon; the 4h head still contributes to entry.
    cfg['forecast']['long_heads'] = ['h60m', 'h120m']
    return cfg


async def run(bundle, prices, out, trained_for_study=False):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / 'bundle.json').read_text())
    provenance = json.loads(prices.with_suffix('.json').read_text())
    if provenance.get('price_warmup_days', 0) < 8:
        raise ValueError('Eight days of price warmup required')
    template = yaml.safe_load(Path('config/hybrid.yaml').read_text())
    cases = [('primary_maker', True, 2., 'combined'),
             ('conservative_maker', True, 5., 'combined'),
             ('taker_sensitivity', False, 2., 'combined'),
             ('price_only_context', True, 2., 'price_only')]
    configs = {name: policy(template, manifest, provenance, maker, p, mode)
               for name, maker, p, mode in cases}
    report = {'study': 'EMA_pullback_20261004', 'status': 'frozen_before_replay',
        'primary': 'primary_maker', 'independent_holdout': False, 'new_training': trained_for_study,
        'selection_performed': False, 'configs': configs, 'records': [],
        'provenance': provenance, 'model_version': manifest['model_version'],
        'limitations': ['Previously inspected September dates, not independent confirmation.',
                        'Binance second-open price proxies, not Roostoo bid/ask.',
                        'No queue, spread, partial-fill or network-latency calibration.',
                        'Five-request execution budget excludes cached historical feed HTTP.']}
    (out / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, _, _, mode in cases:
        started = time.time()
        cfg = configs[name]
        events = bundle_events(prices, bundle, list(cfg['strategy']['rules']), mode != 'price_only')
        result = await run_replay(cfg, events)
        result['summary'].update(extra_metrics(result))
        result['summary']['ema_rejection_reasons'] = dict(Counter(e['reason'] for e in result['events']
                                                            if e['event'] == 'ema_entry_rejected'))
        result['summary']['trade_count_multiple_of_E37_primary_5'] = result['summary']['completed_holds'] / 5
        with gzip.open(out / f'{name}.json.gz', 'xt') as stream:
            json.dump(result, stream, allow_nan=False)
        report['records'].append({'name': name, 'wall_seconds': time.time() - started,
                                  'policies': {'frozen_policy': result['summary']}})
        report['status'] = 'complete' if len(report['records']) == len(cases) else 'running'
        (out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'name': name, **result['summary']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('bundle', 'prices', 'out'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--trained-for-study', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.bundle, args.prices, args.out, args.trained_for_study))
