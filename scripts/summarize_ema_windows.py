"""Audit all E39 cash ledgers and publish compact, reproducible local evidence."""
from collections import Counter, defaultdict
import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil

import numpy as np


def epoch(value):
    dt = datetime.fromisoformat(value)
    return int(dt.replace(tzinfo=timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp())


def audit_run(path):
    with gzip.open(path, 'rt') as stream:
        result = json.load(stream)
    summary = result['summary']
    final_orders = {e['order']['order_id']: e['order'] for e in result['events'] if e['event'] == 'order_event'}
    orders = sorted((o for o in final_orders.values() if o['filled_quantity'] > 0), key=lambda o: o['filled_at'])
    intents = {(e['symbol'], e['timestamp']): e for e in result['events'] if e['event'] == 'entry_intent'}
    closures = {(e['symbol'], e['closed_at']): e for e in result['events'] if e['event'] == 'position_closed'}
    positions, trades = {}, []
    for order in orders:
        symbol = order['symbol']
        quantity = order['filled_quantity']
        notional = quantity * order['filled_price']
        fee = order['commission'] or 0.
        if order['side'] == 'BUY':
            if symbol in positions:
                raise ValueError('Unexpected partial or overlapping entry; extend auditor before claiming results')
            intent = intents[(symbol, epoch(order['created_at']))]
            positions[symbol] = dict(symbol=symbol, quantity=quantity, cost=notional, fees=fee,
                opened_at=epoch(order['filled_at']), size_multiplier=intent.get('size_multiplier', 1.))
        else:
            position = positions.pop(symbol)
            if abs(position['quantity'] - quantity) > 1e-8:
                raise ValueError('Unreconciled sell size')
            close = closures[(symbol, epoch(order['filled_at']))]
            position.update(fees=position['fees'] + fee, gross_pnl=notional - position['cost'],
                            hold_seconds=close['holding_seconds'], exit_reason=close['reason'])
            position['net_pnl'] = position['gross_pnl'] - position['fees']
            trades.append(position)
    difference = sum(t['net_pnl'] for t in trades) - (summary['final_nav'] - 100000)
    if positions or abs(difference) > 1e-6 or len(trades) != summary['completed_holds']:
        raise ValueError('Cash ledger, completion count or flat reset audit failed')
    if any(summary[k] for k in ('positions', 'pending_orders', 'venue_fills_not_yet_reconciled')):
        raise ValueError('Final state not flat and reconciled')
    return trades, difference


def summarize(run, model_run, out):
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / 'comparison.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Study not complete')
    all_trades, errors = defaultdict(list), []
    for row in report['records']:
        trades, error = audit_run(run / f"{row['window']:02d}_{row['case']}.json.gz")
        all_trades[row['case']].extend(trades)
        errors.append(abs(error))
    diagnostics = {}
    for case, trades in all_trades.items():
        grouped = {}
        for field in ('symbol', 'size_multiplier', 'exit_reason'):
            grouped[field] = {}
            for value in sorted({t[field] for t in trades}, key=str):
                rows = [t for t in trades if t[field] == value]
                grouped[field][str(value)] = dict(trades=len(rows),
                    net_pnl=sum(t['net_pnl'] for t in rows), gross_pnl=sum(t['gross_pnl'] for t in rows),
                    fees=sum(t['fees'] for t in rows), win_rate=sum(t['net_pnl'] > 0 for t in rows) / len(rows))
        holds = [t['hold_seconds'] for t in trades]
        diagnostics[case] = dict(groups=grouped, pooled_hold_median_seconds=float(np.median(holds)),
            hold_fraction_1_to_2h=sum(3600 <= h <= 7200 for h in holds) / len(holds),
            exit_reasons=dict(Counter(t['exit_reason'] for t in trades)))
    audit = dict(windows_audited=len(errors), maximum_cash_reconciliation_error=max(errors),
                 all_flat_and_reconciled=True, diagnostics=diagnostics,
                 limitation='All accounting is simulated; does not verify venue matching or forecast generalization.')
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    shutil.copyfile(run / 'comparison.json', out / 'comparison.json')
    shutil.copyfile(run / 'plan.json', out / 'plan.json')
    fields = ['window', 'case', 'start_utc8', 'end_exclusive_utc8', 'net_return', 'completed_holds',
              'active_fill_days_utc8', 'max_drawdown', 'fee_total_usdt', 'median_hold_seconds', 'valid_window']
    with (out / 'windows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in report['records']:
            writer.writerow({k: (r[k] if k in r else r['summary'][k]) for k in fields})
    if model_run:
        model = json.loads((model_run / 'comparison.json').read_text())
        if model['status'] != 'complete':
            raise ValueError('Model replay not complete')
        for record in model['records']:
            audit_run(model_run / (record['name'] + '.json.gz'))
        shutil.copyfile(model_run / 'comparison.json', out / 'model_comparison.json')
    print(json.dumps(audit, indent=2))


def plot_windows(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((out / 'comparison.json').read_text())
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    for case, label in [('strict_maker_2bps', 'Strict EMA / maker'),
                        ('soft_maker_2bps', 'Soft EMA / maker (primary)'),
                        ('soft_maker_5bps', 'Soft EMA / conservative fills'),
                        ('soft_taker', 'Soft EMA / taker')]:
        rows = [r for r in report['records'] if r['case'] == case]
        x = [r['window'] + 1 for r in rows]
        axes[0].plot(x, [r['summary']['net_return'] * 100 for r in rows], marker='.', label=label)
        axes[1].plot(x, [r['summary']['completed_holds'] for r in rows], marker='.')
    axes[0].axhline(0, color='black', linewidth=.8)
    axes[0].set_ylabel('14-day net return (%)')
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].set_ylabel('Completed trades')
    axes[1].set_xlabel('Non-overlapping 14-day window (Sep 10, 2025 - Aug 25, 2026; UTC+8)')
    axes[1].set_xticks(range(1, 26))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(17.5, color='grey', linestyle='--', linewidth=1)
    fig.suptitle('E39: more trading is not more alpha\nPrice-only diagnostic; each window starts flat with 100,000; simulated fees/fills')
    fig.savefig(out / 'window_returns.png', dpi=160)
    plt.close(fig)


def export_research(out, legacy_model, research_root):
    """Generated evidence only; canonical English narrative is edited separately."""
    report = json.loads((out / 'comparison.json').read_text())
    model = json.loads((out / 'model_comparison.json').read_text())
    legacy = json.loads((legacy_model / 'comparison.json').read_text())
    dest = research_root / 'runs' / 'ema_windows_E39_20261004_evidence'
    dest.mkdir(parents=True, exist_ok=False)
    for name in ('comparison.json', 'model_comparison.json', 'audit.json', 'plan.json'):
        shutil.copyfile(out / name, dest / name)
    shutil.copyfile(legacy_model / 'comparison.json', dest / 'legacy_model_mark_to_market.json')
    records = [dict(name=f"window_{r['window']:02d}_{r['case']}",
                    policies={'price_only_flat': r['summary']}, start=r['start'], end=r['end'])
               for r in report['records']]
    records += [dict(r, name='corrected_soft_' + r['name']) for r in model['records']]
    records += [dict(r, name='superseded_legacy_end_' + r['name']) for r in legacy['records']]
    summary = dict(study='E39', new_training=False, records=records, aggregate=report['aggregate'],
        data_rows=report['data_manifest']['total_rows'], corrected_evaluation_rows=104,
        superseded_mark_to_market_rows=4, local_only=True, paid_gpu_used=False,
        production_enabled=False, live_orders_sent=0,
        limitation='100 price-only windows do not evaluate E38 forecasts out of sample; four corrected model replays reuse inspected September dates.',
        bot_evidence_directory=str(out.resolve()), source_code_repository='systematic_crypto_trading_bot')
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    published = research_root / 'docs' / 'assets' / 'ema_windows_E39_20261004'
    published.mkdir(parents=True, exist_ok=False)
    for name in ('summary.json', 'audit.json'):
        shutil.copyfile(dest / name, published / name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model-run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--legacy-model-run', type=Path)
    args = parser.parse_args()
    summarize(args.run, args.model_run, args.out)
    if args.plot:
        plot_windows(args.out)
    if args.research_root:
        if not args.legacy_model_run:
            parser.error('--legacy-model-run is required for the E39 evidence inventory')
        export_research(args.out, args.legacy_model_run, args.research_root)
