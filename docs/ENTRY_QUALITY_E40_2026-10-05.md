# E40: frozen entry-quality correction and replay

The user requested implementation, backtesting and a written latest strategy.
The change caps price-rule strength, rejects extreme displacements and prevents
deep pullbacks from bypassing independent model support, then requires improving
completed-minute taker-buy flow. See [latest strategy](STRATEGY_LATEST.md) for the
complete specification and exact configuration. Parameters were frozen before
replay; there was no threshold sweep, retraining or post-hoc candidate selection.

## Data and controls

- Reuse the E39 year of BTC/XRP/BNB spot1m bars,25non-overlapping14day flat-start
  windows,September10,2025--August25,2026 UTC+8. Three cases per window=75rows.
- Download48September1--16daily1m archives for exact flow in the eligible model
  replay:69,120new asset-minute rows,2,988,316ZIPbytes; reuse August warmup.
  Exact grid/OHLCV/ZIP integrity checks; no full-file hashing or L2 reconstruction.
- Reuse E38 ensemble forecasts and five-second price proxies for September3--16.
  Five model cases: unchanged baseline, score correction only, full primary,
  conservative maker fill and taker sensitivity. Same risk, sizing and exits.
- The historical75rows are PRICE ONLY; E38 trained on earlier history and is
  not backdated into those windows. All dates are inspected; no fresh holdout.
-26control rows exactly reproduce E39 return, with no financial redefinition.
  These repeats are counted in the80row inventory, not independent tests.

## Results: 25 price-only windows

| Fixed case | Mean14d net | Mean trades | Mean active days | Activity-pass windows | Positive windows | Worst DD |
|---|---:|---:|---:|---:|---:|---:|
| E39 baseline | -0.8630% | 48.36 | 12.60 | 25/25 | 1/25 | 2.0345% |
| Quality maker2bps, primary | -0.6107% | 37.12 | 12.20 | 25/25 | 1/25 | 1.5581% |
| Quality maker5bps | -0.3925% | 23.24 | 10.40 | 22/25 | 1/25 | 1.4012% |

Primary earlier17mean-0.6062percent; later-eight-0.6201percent,all eight negative.
Primary total net-15,267.17 across independently funded accounts,fees9,433.43,
gross same-fill PnL-5,833.74. Net per completed trade changes-17.85to-16.45USDT;
net per roundtrip-equivalent notional changes-25.52to-23.51bps. Mean exposure
falls1.793to1.399percent NAV. Thus lower total loss is partly less trading/risk;
it is not a conversion to positive entry expectancy. Conservative maker fills
alter the selected trade set and are not a monotonic worst-PnL bound.

## Results: one eligible, inspected model window

| Fixed case | Net return | Trades | Active days | Median hold | Max DD |
|---|---:|---:|---:|---:|---:|
| E39 model baseline | -0.8403% | 38 | 12 | 71.93min | 0.9287% |
| Score correction only | -0.3680% | 31 | 11 | 80.17min | 0.6093% |
| Full quality maker2bps, primary | -0.1224% | 25 | 11 | 81.10min | 0.2874% |
| Full quality maker5bps | -0.0070% | 13 | 9 | 106.10min | 0.1796% |
| Full quality taker | -0.2520% | 42 | 13 | 75.00min | 0.4208% |

Primary fees239.70USDT,net-122.43,gross same-fill+117.27;36maker and14taker fills,
92percent of holds in1--2h. Profit factor0.7505. The5bpscase is still negative,
contains only13trades and was not promoted. All five rows remain negative.

## Entry diagnostics and interpretation

Gross fixed-horizon markouts measure observed prices after actual entry fill,
regardless of whether the strategy already exited. They are not a realizable
alternative equity curve and do not include fees.

Historical primary5minute markout worsens-7.30to-7.68bps;60minute improves
-9.05to-7.91bps;120minute worsens-9.12to-9.55bps. No consistently positive
fixed-horizon edge appears. Model primary5minute improves-7.83to-2.04bps and
30minute-5.49to+2.99bps, but60minute remains-5.82bps and120minute-4.91bps.
Its average maximum adverse move over a fixed two-hour observation falls from
68.37to54.57bps. The model+flow combination improves early adverse movement on
this one period, while long-horizon improvement is mixed. Exit path and lower
turnover contribute to financial improvement. This is not proof of a generally
better predictive model or stable positive alpha.

## Engineering and evidence

All80cash/position ledgers reconcile and finish flat, maximum error7.82e-11USDT.
All26control return differences are exactly zero. Full bot suite319passed,
one existing skip and four pre-existing NumPy warnings. Tests cover flow gaps,
staleness, causal boundaries, zero volume, per-asset state, pending-buy invalidation,
uninterrupted risk exits, weak-score/extreme-dip rejection and markout timing.

The first publishing invocation used the plotting environment, which lacked
pydantic; it failed before writing evidence. Publishing was rerun in the tested
bot environment, and plotting separated into a dependency-light entry point.
No financial runs were rerun or selected because of this environment failure.

Full compressed financial logs remain under `logs/entry_quality_E40_*_20261005/`;
downloaded archives remain in ignored `data/historical/minute_202608_20260916/`.
Tracked [summary](assets/entry_quality_E40_20261005/summary.json),
[audit](assets/entry_quality_E40_20261005/audit.json),
[all windows](assets/entry_quality_E40_20261005/windows.csv), and
[plot](assets/entry_quality_E40_20261005/window_comparison.png) preserve all cases.
The companion research project indexes this same evidence, not another campaign.

## Reproduce

Run from the bot repository, using new output directories (existing runs are immutable):

```bash
.venv/bin/python -m scripts.run_entry_quality --data data/historical/minute_202509_202608 --out logs/NEW_E40_WINDOWS
.venv/bin/python -m scripts.run_entry_quality --data data/historical/minute_202509_202608 --out logs/NEW_E40_MODEL --model-only --flow-data data/historical/minute_202608_20260916 --bundle ../DLmodels/e2eCryptoModels/runs/ema_pullback_20261004/bundle_focused --prices ../DLmodels/e2eCryptoModels/runs/ema_pullback_20261004_local/prices.npz
```

No GPU, venue credentials, live orders, production switch or instance lifecycle
operation was used. The user requested a strategy repair, not guaranteed profit;
E40 remains a research candidate and should not be presented as ready for live deployment.
