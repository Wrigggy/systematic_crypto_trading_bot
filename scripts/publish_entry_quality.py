"""Audit and publish E40 evidence, latest resolved research config and entry diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from scripts.summarize_ema_windows import audit_run


def entry_metrics(trades, times, prices, symbols):
    rows = []
    for trade in trades:
        opened = trade['opened_at']
        entry = trade['cost'] / trade['quantity']
        index = symbols.index(trade['symbol'])
        if times[-1] < opened + 7200:
            raise ValueError('Incomplete two-hour entry diagnostic horizon')
        values = {}
        for minutes in (5, 15, 30, 60, 120):
            pos = np.searchsorted(times, opened + minutes * 60, side='right') - 1
            if pos < 0 or opened + minutes * 60 - times[pos] > 60:
                raise ValueError('Stale entry diagnostic observation')
            values[f'markout_{minutes}m_bps'] = (prices[pos, index] / entry - 1) * 10000
        left, right = np.searchsorted(times, [opened, opened + 7200], side='right')
        path = (prices[left:right, index] / entry - 1) * 10000
        values.update(mae_120m_bps=float(min(0, path.min())), mfe_120m_bps=float(max(0, path.max())))
        rows.append(values)
    return dict(trades=len(trades), **{key: float(np.mean([r[key] for r in rows]))
                                     for key in rows[0]}) if rows else {'trades': 0}


def publish(windows, models, data, price_path, out, research=None, plot=False):
    from scripts.run_ema_windows import load_prices
    out.mkdir(parents=True, exist_ok=True)
    study = json.loads((windows / 'comparison.json').read_text())
    model = json.loads((models / 'model_comparison.json').read_text())
    if study['status'] != 'complete' or model['status'] != 'complete':
        raise ValueError('Both studies must complete before publishing')
    old = json.loads(Path('docs/assets/ema_windows_E39_20261004/comparison.json').read_text())
    old_model = json.loads(Path('docs/assets/ema_windows_E39_20261004/model_comparison.json').read_text())
    symbols = ['BTC/USDT', 'XRP/USDT', 'BNB/USDT']
    times, prices = load_prices(data, symbols)
    with np.load(price_path, allow_pickle=False) as p:
        mtimes, mprices, msymbols = p['price_times'], p['prices'], p['symbols'].tolist()
    by_case, errors, baseline_diffs, diagnostics = {}, [], [], {}
    for record in study['records']:
        trades, error = audit_run(windows / (record['name'] + '.json.gz'))
        errors.append(abs(error))
        by_case.setdefault(record['case'], []).extend(trades)
        if record['case'] == 'baseline':
            reference = next(r['summary'] for r in old['records'] if r['window'] == record['window']
                             and r['case'] == 'soft_maker_2bps')
            result = record['policies']['fixed_policy']
            baseline_diffs.append(abs(reference['net_return'] - result['net_return']))
            if any(reference[k] != result[k] for k in ('completed_holds', 'active_fill_days_utc8', 'fee_total_usdt')):
                raise ValueError('Historical baseline regression')
    for case, trades in by_case.items():
        diagnostics[case] = entry_metrics(trades, times, prices, symbols)
    for record in model['records']:
        trades, error = audit_run(models / (record['name'] + '.json.gz'))
        errors.append(abs(error))
        diagnostics[record['name']] = entry_metrics(trades, mtimes, mprices, msymbols)
        if record['name'] == 'model_baseline':
            reference = old_model['records'][0]['policies']['frozen_policy']
            baseline_diffs.append(abs(reference['net_return'] - record['policies']['fixed_policy']['net_return']))
    if max(baseline_diffs) > 1e-12:
        raise ValueError('Baseline returns changed')
    audit = dict(replays=len(errors), all_flat_and_reconciled=True,
        maximum_cash_error=max(errors), maximum_baseline_return_difference=max(baseline_diffs),
        baseline_checks=26, entry_diagnostics=diagnostics,
        diagnostic_scope='Gross post-fill fixed-horizon markouts, including prices after actual early exit. Not a tradable alternative equity curve.')
    for source, name in [(windows / 'comparison.json', 'comparison.json'),
                         (models / 'model_comparison.json', 'model_comparison.json'),
                         (windows / 'plan.json', 'plan.json'), (models / 'model_plan.json', 'model_plan.json')]:
        shutil.copyfile(source, out / name)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    with (out / 'windows.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['window', 'case', 'start_epoch', 'end_epoch', 'net_return', 'trades', 'active_days', 'fees', 'drawdown'])
        for r in study['records']:
            s = r['policies']['fixed_policy']
            writer.writerow([r['window'], r['case'], r['start'], r['end'], s['net_return'],
                s['completed_holds'], s['active_fill_days_utc8'], s['fee_total_usdt'], s['max_drawdown']])
    latest = model['configs']['quality_maker']
    latest['mode'] = 'replay'
    latest['research_status'] = 'E40_frozen_candidate_not_deployment_approved'
    latest['replay']['events_path'] = None
    Path('config/strategy_latest.yaml').write_text(
        '# Generated by scripts.publish_entry_quality. Research only; explicit causal event stream required.\n'
        + yaml.safe_dump(latest, sort_keys=False))
    summary = dict(study='E40', new_training=False, selection_performed=False, local_only=True,
        live_orders_sent=0, deployment_approved=False, records=study['records'] + model['records'],
        aggregate=study['aggregate'], audit=audit,
        limitation='75 historical price-only rows and five eligible-date model rows; includes 26 exact control repetitions; all dates inspected.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if plot:
        plot_study(out)
    if research:
        dest = research / 'runs' / 'entry_quality_E40_20261005_evidence'
        dest.mkdir(parents=True, exist_ok=False)
        public = research / 'docs' / 'assets' / 'entry_quality_E40_20261005'
        public.mkdir(parents=True, exist_ok=False)
        for name in ('summary.json', 'audit.json'):
            shutil.copyfile(out / name, dest / name)
            shutil.copyfile(out / name, public / name)
    print(json.dumps(dict(aggregate=study['aggregate'], audit=audit), indent=2))


def plot_study(out):
    """Can run in the separate plotting environment without bot dependencies."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study = json.loads((out / 'comparison.json').read_text())
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), layout='constrained')
    for case in study['aggregate']:
        rows = [r for r in study['records'] if r['case'] == case]
        axes[0].plot([r['window'] + 1 for r in rows], [r['policies']['fixed_policy']['net_return']*100 for r in rows], '.-', label=case)
        axes[1].plot([r['window'] + 1 for r in rows], [r['policies']['fixed_policy']['completed_holds'] for r in rows], '.-')
    axes[0].axhline(0, color='black', linewidth=.7)
    axes[0].set_ylabel('14-day net return (%)')
    axes[0].legend()
    axes[1].set_ylabel('Completed trades')
    axes[1].set_xlabel('Non-overlapping 14-day window; cash/flat reset; UTC+8')
    axes[1].set_xticks(range(1, 26))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(17.5, color='grey', linestyle='--')
    fig.suptitle('E40 fixed entry-quality repair vs E39 control\nHistorical price-only comparison, not model out-of-sample evidence')
    fig.savefig(out / 'window_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('windows', 'models', 'data', 'prices', 'out'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--plot', action='store_true')
    a = parser.parse_args()
    publish(a.windows, a.models, a.data, a.prices, a.out, a.research_root, a.plot)
