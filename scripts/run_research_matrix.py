"""Frozen offline E37 matrix; no test-based selection or live exchange access."""
import argparse
import asyncio
import gc
import gzip
import json
from pathlib import Path
import time

import yaml

from scripts.replay_hybrid import run_replay
from scripts.replay_research_bundle import bundle_events, configure, extra_metrics


async def run(study, prices, out, deadline):
    selection = json.loads((study / 'selection.json').read_text())
    if selection['selection_uses'] != 'validation_only' or selection['test_opened']:
        raise ValueError('Require a pre-test frozen validation selection.')
    selected = selection['selected']
    variants = [n for n in ('shared', 'grouped', 'mlp') if (study / f'bundle_{n}/bundle.json').exists()]
    if selected not in variants or not {'shared', 'grouped'} <= set(variants):
        raise ValueError('Missing completed control or selected bundle.')
    cases = []
    for maker in (True, False):
        cases.append(('price_only', 'shared', maker, False, 2.))
    for name in variants:
        for maker in (True, False):
            cases.append(('combined', name, maker, False, 2.))
    for maker in (True, False):
        cases.append(('combined', selected, maker, True, 2.))
    cases.append(('combined', selected, True, True, 5.))
    out.mkdir(parents=True, exist_ok=False)
    template = yaml.safe_load(Path('config/hybrid.yaml').read_text())
    provenance = json.loads(prices.with_suffix('.json').read_text())
    report = {'study': 'E37', 'selected_by_validation': selected, 'selection': selection,
              'cases_planned': len(cases), 'independent_holdout': False, 'records': [],
              'deadline': deadline, 'status': 'running',
              'no_pnl_based_reselection': True,
              'price_only_note': 'Repeat of earlier engineering controls with unchanged rules and fills.',
              'fill_boundary': 'Binance last-second-open proxies; uncalibrated queue/spread/latency.'}
    for mode, variant, maker, review, penetration in cases:
        if time.time() >= deadline:
            raise TimeoutError('Local experiment deadline reached.')
        case = f'{mode}_{variant}_{"maker" if maker else "taker"}_review{int(review)}_p{int(penetration)}'
        bundle = study / f'bundle_{variant}'
        manifest = json.loads((bundle / 'bundle.json').read_text())
        cfg = configure(template, manifest, provenance['replay_start'], provenance['replay_end'],
                        mode, maker, review, penetration)
        events = bundle_events(prices, bundle, list(cfg['strategy']['rules']), mode != 'price_only')
        started = time.time()
        result = await run_replay(cfg, events)
        result['summary'].update(extra_metrics(result))
        result['provenance'] = {'bundle': manifest, 'prices': provenance, 'independent_holdout': False}
        with gzip.open(out / f'{case}.json.gz', 'xt') as stream:
            json.dump(result, stream, allow_nan=False)
        with (out / f'{case}.summary.json').open('x') as stream:
            json.dump(result['summary'], stream, indent=2, allow_nan=False)
        report['records'].append({'name': case, 'variant': variant, 'mode': mode,
            'maker_preferred': maker, 'model_exit_enabled': review,
            'penetration_bps': penetration, 'wall_seconds': time.time() - started,
            'policies': {'frozen_policy': result['summary']}})
        (out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'completed_case': case, 'net_return': result['summary']['net_return'],
                          'active_days': result['summary']['active_fill_days_utc8']}), flush=True)
        del result, events
        gc.collect()
    report['status'] = 'complete'
    report['finished_epoch'] = time.time()
    (out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'prices', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline', type=int, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.study, args.prices, args.out, args.deadline))
