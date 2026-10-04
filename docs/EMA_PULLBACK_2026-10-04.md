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
| Conservative maker, 5bps | -0.19380% | See receipt | 2 | 1 | 120min |
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

Training/backup completion and measured results will be added only after observed.
