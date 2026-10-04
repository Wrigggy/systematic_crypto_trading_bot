# E38: EMA pullback, short-focused forecasts and testing API connection

## Protocol and evidence boundary

The requested target is now 1--2 hours of actual holding, with more opportunities
and long-only directional exposure. This supersedes the earlier 3--4 hour target,
not the historical results. No minimum holding lock-in, forced participation,
continuous two-sided market making, live hybrid deployment or automatic orders.

The frozen policy uses completed five-minute observations, fast/slow EMA half-lives
of 6/24 hours, residual-volatility half-life 24 hours and seven days of uninterrupted
EMA warmup. Require a half-standard-deviation pullback below the fast middle,
fast above slow, rising fast middle over one hour, a recovering completed bar,
and at least 20bps distance to the recovery middle. Price and model jointly score
entries (equal branch weights), not model-only trading. Long-head support is no
longer a hard entry veto; positive weighted model support remains required.

Forecast heads remain 15/30/60/120/240 minutes. Their fixed score weights are
10/25/40/20/5 percent, not fitted on replay PnL. Review 60/120-minute model support
after one hour, confirm weakness for five minutes, and exit by two hours. Recovery,
3-percent stop/trailing stop or account risk may exit earlier. Position sizing is
10-percent NAV, with the retained 50-percent portfolio and three-position caps.

Passive entries use venue-side bid and exits use ask with outward tick rounding.
Limits are not post-only guarantees. Entry expiry is 120 seconds; cancel and
reconcile without chasing. Stops/deadlines and failed passive exits may use taker.
Actual role and commission receipts, not order type, decide maker accounting.
The shared rolling budget is five HTTP requests/minute, including reads.

Four cases are frozen: primary maker (2bps later-price penetration), stricter
maker (5bps), taker sensitivity, and price-only EMA context. All use 14 Singapore
days, September 3--16, cash 100,000 and flat initial positions, with external
warmup. Symbols are BTC/XRP/BNB. Fees are modeled as 5/10bps maker/taker with zero
slippage. These are reused, inspected dates, not an independent holdout. Historical
Binance second-open proxies do not validate Roostoo quotes, spread, queue position,
latency or real fill probability. The execution budget excludes cached feed HTTP.

## Implementation

- `strategy/ema_pullback.py`: causal completed-boundary history, missing-bar reset,
  residual moments, trend/bounce/distance checks; no unfinished-bar updates.
- `strategy/hybrid.py`: opt-in EMA entry middle and checks, including pending-buy
  invalidation. The old policy is unchanged when the option is disabled.
- `scripts/run_ema_pullback.py`: immutable four-case replay output with fill,
  cost, activity, holding, risk and rejection metrics. It never selects a winner.
- `scripts/check_roostoo.py`: separate read-only connectivity check.

The initial `logs/ema_pullback_20261004` replay is preserved. A final repetition
in `logs/ema_pullback_20261004_final` includes consistent pending-entry invalidation;
it is an engineering confirmation, not another independent financial experiment.

Reference-model results reproduce after that correction:

| Frozen case | Net return | Max drawdown | Closed trades | Active fill days | Median hold |
|---|---:|---:|---:|---:|---:|
| Primary maker, 2bps | -0.19557% | 0.22342% | 2 | 1 | 120min |
| Conservative maker, 5bps | -0.19380% | 0.22459% | 2 | 1 | 120min |
| Taker sensitivity | -0.34604% | 0.40102% | 3 | 1 | 120min |
| Price-only EMA | -0.17892% | 0.31409% | 3 | 2 | 120min |

All four fail the activity requirement and requested fivefold trade increase.
Primary maker has two maker entries and two deadline taker exits, $29.83 modeled
fees, zero winning trades and no unresolved/terminal inventory. Its 100-percent
order-fill fraction is a four-order proxy simulation, not evidence of live hit rate.
Of 1,519 repeated model-qualified checks, 1,509 fail the trend gate and seven fail
bar recovery. This identifies a narrow intersection of oversold and still-rising
EMA conditions. Price-only still trades only three times: new model weights alone
cannot be assumed to solve the price-policy opportunity bottleneck.

The independent-of-model diagnostic confirms only1/1/2 fully qualified grid
observations for BTC/XRP/BNB out of4,032each. This is coverage on the fixed grid,
not a formal bound for every possible order/holding path or an independent sample
count. See `assets/ema_pullback_20261004/opportunity_diagnostics.json`.

Tracked receipt: `assets/ema_pullback_20261004/reference_comparison.json`.
The official [event page](https://luma.com/coghwiyt), checked October4, still lists
0.05/0.10-percent maker/taker fees and eight active days. Five HTTP requests/minute
and at least one daily fill on those days remain user-supplied clarifications.
The [official API reference](https://github.com/roostoo/Roostoo-API-Documents)
defines LIMIT orders and fill/commission receipts; passive placement is not proof
of post-only behavior or guaranteed execution.

## Testing API connection

Authenticated read-only checks completed October 4 at 13:54 UTC: server time,
exchange definitions (88 pairs), balance, BTC/USD valid bid/ask, pending count and
pending-order query. Empty-order API responses carry `Success=false`; only the
documented empty-result messages are accepted, not arbitrary HTTP200 failures.
The same fail-closed empty-account contract is now enforced by the executor's
startup check. Zero counts accompanying authentication errors are not accepted.
No placement, cancellation or short operation exists in this checker. No orders
were submitted. Receipt: `assets/ema_pullback_20261004/roostoo_readonly.json`.

Credentials are in the local ignored `.secrets/roostoo-test.json` with private
permissions; this is plaintext local storage, not a secret vault. Never commit,
print or transfer this file to a GPU machine. The testing scope is distinct from
competition credentials. For a read-only recheck:

```bash
.venv/bin/python -m scripts.check_roostoo
```

The main runner can explicitly load this testing file through
`ROOSTOO_TEST_CREDENTIAL_FILE`; existing complete environment credential pairs
take priority. This does not enable live mode, approve trading, provide feature-
exact live forecasts, or bypass the main hybrid deployment guard.

## New model authorization

The user authorized up to four hours on existing instance53113057, hostname
68941bac6e6e, RTX5090, and requires it to remain running. Bound window:
October4 13:49:41--17:49:41 UTC (21:49:41--October5 01:49:41 Singapore).
Process-only guard, no instance lifecycle call. Training cutoff is one hour
before the ceiling; forecast export cutoff is thirty minutes before it.

Research runner: sibling `e2eCryptoModels/scripts/run_pullback_training.py`.
New two-seed shared GRUs retain 11,596,261 parameters, 224 training days, eight
assets, three input scales and 53 non-L2 context features. No data/capacity expansion.
Train-only per-asset input moments, train-only per-head label scaling, common4h
purge and validation-only early stopping remain. Weighted MSE uses the same
10/25/40/20/5-percent horizon priorities. Training uses strict FP32, unlike the
older TF32-enabled reference: this is not a pure objective-weight ablation.
No adaptive search or post-hoc policy reselection is scheduled. Two seeds and
their validation-selected checkpoints are retained regardless of replay PnL.

## Completed new-model results

Both seeds finished with validation-only early stopping. Seed42 selected11,250
updates and stopped19,250; seed43 selected750 and stopped8,750. Weighted validation
MSE compared with the reweighted E35 checkpoints is2.011160vs2.014330 and
2.021719vs2.018321 respectively: +0.157percent and-0.168percent relative improvement.
The seed mean is essentially unchanged, not a seed-stable validation gain.
Each model can sample5,134,056 strongly overlapping asset/time rows; this is not
that many independent observations or a claim of traversing every training row.

| New-model frozen case | Net return | Max drawdown | Closed trades | Active days |
|---|---:|---:|---:|---:|
| Primary maker, 2bps | -0.13645% | 0.23406% | 3 | 2 |
| Conservative maker, 5bps | -0.13469% | 0.23230% | 3 | 2 |
| Taker sensitivity | -0.29701% | 0.42091% | 4 | 2 |
| Price-only EMA, repeated control | -0.17892% | 0.31409% | 3 | 2 |

Primary net PnL is-$136.45 on100k; three trade PnLs are-$95.30,-$100.26,+$59.11.
The improvement over the old model is entirely the additional winning trade.
Win fraction33.3percent, profit factor0.3023, median hold120minutes,2of3holds within
1--2hours. There are four maker fills and two deadline taker fills; fees$39.84.
Adding fees back on the same fills still leaves-$96.61 price PnL. No open positions,
pending orders or unreconciled simulated fills remain. None meets8active days or
fivefold frequency. Taker fills alter trade paths, so the maker/taker PnL difference
must not be attributed solely to the fee-rate difference.

DailyIC is computed per Singapore day/per asset, then averaged over days and
equal-weighted over assets. All dates remain inspected and labels overlap.

| Head | Old DailyIC, trade3 | New DailyIC, trade3 | New Daily RankIC, trade3 | New DailyIC, all8 |
|---|---:|---:|---:|---:|
| 15m | 0.01084 | 0.05261 | 0.03188 | 0.04987 |
| 30m | 0.01786 | 0.05582 | 0.03321 | 0.06862 |
| 60m | 0.02536 | 0.09974 | 0.08230 | 0.09823 |
| 120m | 0.04982 | 0.08247 | 0.09933 | 0.08323 |
| 240m | 0.00367 | 0.05577 | 0.08085 | 0.07346 |

This is a useful retrospective ranking diagnostic, not a fresh holdout or proven
tradable advantage.60/120/240minute ensemble MSE still loses to the constant
train-mean prediction. New weights, precision, GPU hardware and panel placement
differ from E35; do not claim a clean causal loss-weight effect.

All23manifest-listed files/392,028,244bytes are backed up locally with size,
readability and scaler/label lineage checks, not full-file hashes. Six ensemble
asset/time endpoint probes pass CPU/GPU parity, max1.217e-9. Remote training and
exports finished14:45:56UTC; workerEXITED and GPU0percent utilization. At14:50UTC
the owned guard was stopped; instance53113057 remained SSH-responsive. No instance
Stop/Inactivate/Destroy command was issued. The four-hour ceiling was not exhausted
and does not schedule another experiment. Keeping the instance open can incur charges.

Financial receipt: `assets/ema_pullback_20261004/focused_comparison.json`.
Full models, loss curves, metrics and lifecycle are in sibling e2eCryptoModels:
`runs/ema_pullback_20261004` and `docs/assets/ema_pullback_20261004`.
Both source repositories retain the failed reference and all four fixed cases.

The actual `RoostooExecutor` also passed a separately guarded read-only smoke
check at14:55UTC: startup88instruments, balance and a fresh valid BTC quote,
five HTTP attempts, zero orders/cancels. It retains the normal rolling budget
and emergency reserve. Receipt: `assets/ema_pullback_20261004/executor_readonly.json`.
This is not a deployed hybrid live feed. No automatic trading has been started.

## Verification so far

Final bot regression suite: 300 passed, one optional skip, four pre-existing small-sample
NumPy metric warnings. Research local suite: 184 passed, two CUDA skips. Remote
startup suite: 24 passed. The discarded 12-update throughput/label canary has
120 direct TWAP probes with maximum error2.114e-9 and mean step time55.0ms.
These are engineering checks, not financial performance or live fill validation.
