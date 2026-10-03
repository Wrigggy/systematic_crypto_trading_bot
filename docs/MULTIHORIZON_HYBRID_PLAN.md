# Multi horizon model and mean reversion implementation plan

## Scope and decisions

Implement a research-only hybrid: the old price mean-reversion signal and new
multi-horizon model signals jointly determine entry strength and candidate ranking.
Neither an independent model-only strategy nor merely a binary model filter is
the primary policy. Deterministic risk and execution manage orders. The user's current
target is an actual holding duration of three to four hours, not a mandatory
minimum. Predictions have multiple horizons, not one fixed three-to-four-hour label.
Decision frequency remains undecided and must be supplied explicitly. No training,
remote machine access, deployment or trading is authorized by this implementation.
Ask the user for an available machine and a compute budget before training.

The teammate reference is Wrigggy/e2eCryptoModels commit
`8c3184afcb0ff9532d05a4aa8fd006269f60ad18`. Its E34 rules are retrospective research,
not deployment-approved. Preserve its original results. Its main strategy uses
hourly observations and eight-hour rebalances; our higher-frequency policy is an
explicit adaptation, not an exact E34 reproduction.

## Implementation order

1. Finish the interrupted maker order lifecycle work and regression tests.
2. Add a strict multi-horizon forecast contract, causal per-asset score processing,
   and a model-output provider that fails closed without valid artifacts.
3. Add a separate hybrid strategy and executable runtime/replay entry point.
4. Verify synthetic end-to-end decisions, fills, cancellations and risk exits.
5. Stop before new model training. Financial experiments require real eligible
   forecasts, data and separately frozen decision cadence and model horizons.

## Model contract and training handoff

Every forecast contains symbol, prediction time, data cutoff, availability time,
expiry, model version, preprocessing version, checkpoint eligibility time and a
set of horizon-specific predictions. Labels explicitly identify their TWAP window
or point-return definition. Predictions are log returns in original units, not
standardized training labels. Missing heads, future data, stale packets, unknown
versions or decisions before checkpoint eligibility prohibit new entries.

The contract supports either independent horizon models or a shared encoder with
multiple heads. The initial untrained recipe proposes 15, 30, 60, 120 and 240 minute
horizons, using a TWAP over the latter half of each horizon. These are configurable
research defaults, not a user-approved optimum. Existing 30-to-60-minute and
two-to-three-hour models do not magically supply all requested heads.

For new training, retain the multiscale one-second, ten-second and minute branches
and optional non-L2 context. Fit input and per-head target scalers on training data
only. Purge labels crossing each split using the longest target and execution
delay. Require forward-only forecasts for any learned second-stage combiner.
First use fixed normalized weights and a sign-consistency gate; a learned linear
or small MLP combiner is a later, separately evaluated experiment.

Score history is maintained separately by asset, model version and horizon.
The current value is excluded from its own mean and standard deviation. The
24-hour window is time-based; cadence, minimum history and missingness checks are
explicit. Use all eligible predictions, not just price-entry candidates. A model
or preprocessing change invalidates existing score history.

## Price strategy and holding behavior

Start with BTC, XRP and BNB, which have both existing model coverage and E34 mean
reversion rules. Keep the wider training universe distinct from the trade universe.
Do not silently add ZEC or turn SOL/NEAR trend rules into mean reversion.

The E34 price statistic samples 60 completed hourly closes eight hours apart over
20 days. BTC and XRP enter below one standard deviation; BNB below two. BTC exits
above minus half a standard deviation; XRP and BNB above the mean. The hybrid
recomputes this statistic using the latest closed hour and compares the current
completed observation with it. This intraday re-anchoring is explicitly not E34's
fixed eight-hour decision grid. Require continuous hourly history.

The original lower-band breach remains necessary. Convert the depth of price
dislocation into a continuous rule strength, normalized by each asset's entry
band width. Separately combine causal, per-asset, per-horizon model z-scores into
a model strength. Both feed a fixed-weight joint score. Clip each strength before
combining so one extreme observation cannot dominate without bound.

Initial research defaults use equal rule/model weights, model z-score scale three,
strength clipping at three and joint entry threshold one. These are explicit
proposals, not historical user thresholds transplanted into different units and
not optimized weights. Require positive model support and nonnegative aggregate
long-horizon support. Here support means relative to the model's own past score
distribution, not a guaranteed positive absolute return forecast. No absolute
predicted-return entry gate is used. Rank simultaneous eligible assets by joint
strength before allocating shared cash. Position size remains fixed for attribution.

Consequently, deeper price dislocation may qualify with more modest model support;
a shallow dislocation needs stronger model support. Positive model predictions
alone cannot bypass the original oversold condition. A short-horizon fluctuation
alone does not close a position. Freeze the price-recovery target at the entry
intent and retain it when the first fill is received. Time held starts from the
first known execution, including partial fills, not order submission.
Replay preserves the simulated execution time even when API limits delay the
receipt. Roostoo exposes a completion timestamp, not a documented first-partial
timestamp; record the time source and use observation time if execution time is
unavailable. Do not claim exact first-fill timing from a completion-only receipt.

Exit on price recovery, price risk, or a four-hour deadline. Starting at three
hours, sustained loss of long-horizon support may trigger an orderly exit; its
confirmation duration is elapsed time, not an implicit number of samples. Missing
forecasts prohibit entries but never disable risk/deadline exits. Re-entry after an
exit requires a reset of the oversold episode. No arbitrary three-hour lock-in.
The three-to-four-hour interval is a measured target, not an achieved result.

Sizing is fixed and explicit for the first comparison; do not invent Kelly priors
or train a sizing policy. Initial research defaults are 10% NAV per position,
15% per-asset cap, 50% portfolio cap and three positions. Expose parameters in the
configuration. Price risk defaults must be reported separately from E34, which did
not validate this stop policy. Always report actual hold distributions.

## Execution contract

Ordinary entries and exits prefer passive venue quotes: buy at/below bid, sell
at/above ask, outward-rounded to instrument ticks. Explicit target prices are
respected when more passive. A limit order is not guaranteed to be a maker fill;
use actual exchange role and commission receipts. Never claim guaranteed touch fills.

Entry orders expire after 120 seconds by default, or earlier on signal invalidation.
Cancel and reconcile; do not automatically chase with a market buy. Ordinary exits
may wait 120 seconds, then cancel/reconcile and market-sell only remaining inventory.
Stops, deadlines and circuit breakers prefer taker execution after conflicting
orders are reconciled. Requested expiry and actual cancellation/fill time differ.

Every Roostoo HTTP attempt shares five requests per rolling minute, with risk work
prioritized. Quote, balance, submit, status and cancel all count. No endpoint or
urgent order bypasses the platform ceiling. Reserve capacity for risk handling;
uncertain cancellations block replacement rather than risking double sells.

Treat acknowledgement, partial fill, full fill, cancellation and unknown outcomes
separately. Book cumulative quantity/notional/fees by delta. Do not retry ambiguous
placement POSTs without a documented idempotency mechanism. Persist unresolved
orders before submission and fail closed on restart until explicit reconciliation.
No unconditional participation or seed trade.

## Replay and comparisons

Use 14-day cash-start windows with external indicator and forecast-history warmup.
Do not call E34 continuous-position windows equivalent to cash-start competitions.
Keep the original E34 audit baseline, adapted price-only control, binary model
filter and joint-signal fusion separate. The latter three share the price-recovery,
stop and four-hour deadline mechanics, so price-only is not mislabeled an exact
E34 reproduction. Price-only never consults models, including for exits. For an
entry-only comparison, explicitly set `model_exit_enabled: false` in all three
arms. Then compare model-exit enabled versus disabled with identical entry logic
to attribute changes from the three-hour review separately.
Compare taker and maker-preferred execution with
identical signals and risk. Use shared cash, exposure and request constraints.

The engineering replay consumes timestamped completed bars and causal model
packets through the same hybrid coordinator and order manager. Its fills and
request costs are modeled, not a certification of Roostoo matching. Never fill an
entry from a bar preceding its submission. Report pending orders, incomplete exits,
unknown outcomes, and fees; mark positions at the terminal observation rather than
fabricating a terminal fill. An incomplete liquidation is not a flat portfolio.

Financial evaluation must report per-asset per-day IC and rank IC, candidate-only
diagnostics and counts, net return, drawdown, exposure, turnover, holding duration,
active fill days, fill rate, waiting/cancellation time and maker/taker fee attribution.
Calendar is UTC+8. At least eight days with actual fills is a user-supplied contest
gate, not eight days with a position. Do not generate trades solely to pass it.

## Acceptance and evidence boundary

Require deterministic tests for future-data rejection, past-only normalization,
missing horizons, duplicated forecasts, multi-asset isolation, cold start, first
partial-fill timing, no forced minimum hold, four-hour exit without model data,
maker timeout, failed cancellation, fill/cancel races, fee currencies, request
budget, idempotent accounting and restart safety. Run the pre-existing test suite.

Synthetic tests and execution-code readiness are not model quality, an empirical
holding duration, profitability or live deployment evidence. The hybrid config
must not run until decision cadence, forecast provenance and source paths are
explicit. New training is the next authority boundary; machine selection belongs
to the user. Preserve all historical experiments and do not stop any rented machine.

## Local implementation and usage

The bot now contains `strategy/fusion.py` for the joint signal,
`strategy/hybrid.py` for holding and risk decisions, and
`plugins/model_inference/forecasts.py` for causal model packets. The executable
offline path is `scripts/replay_hybrid.py`. The main trading runner explicitly
rejects the hybrid template; it cannot fall through to the old alpha engine or
silently use a random model. No live hybrid inference feed is implemented yet.

Copy the `config/hybrid.yaml` template to an experiment-specific file. Set an
explicit decision interval, artifact/preprocessing identities, event source and
start/end times. Use a 14-day evaluation window with preceding external warmup.
Run from the repository root:

```bash
.venv/bin/python -m scripts.replay_hybrid --config config/my_hybrid.yaml --output logs/my_hybrid_replay.json
```

The source is newline-delimited JSON sorted by `available_at`, not by prediction
or bar time. Each event has `kind`, `symbol`, `timestamp`, and `available_at`.
`price` adds `close` for a completed observation; `hourly` adds `close` with an
hour-end timestamp; `forecast` adds the complete versioned `forecast` packet.
Forecast envelope timestamps and symbol must exactly match the packet. Warmup
events initialize the indicators and model score history without creating trades.
Never manufacture missing model heads or rename old weights to satisfy the manifest.

The replay models a shared request ceiling. Exchange-side limit fills may precede
the next affordable status poll. Output therefore distinguishes reconciled
inventory from venue fills not yet acknowledged; accounting is flagged provisional
when these differ. Terminal positions are marked, not automatically liquidated.
Taker fills use the latest fresh observed close and configured slippage. Maker
fills require a later observed close to penetrate the limit. This is an engineering
model without calibrated queue position, partial fills, book spread or latency;
do not treat its PnL as a validated reproduction of the competition engine.

The output includes the effective config, decision/order audit trail, NAV series,
return, drawdown, exposure, turnover, fee attribution, actual fill days, completed
holding durations, open holding ages, and unresolved orders. DailyIC requires
matured labels and remains an explicit downstream research evaluation, not an
invented number in a replay that contains prices and predictions only.

Local synthetic tests cover fusion sensitivity to both branches, missing-model
behavior, strict causal packets, warmup, fill timing, early recovery, maker timeout,
deadline/stop exits, partial-fill accounting, cancellation races and request limits.
The regression suite currently passes 280 tests with one optional test skipped;
four pre-existing NumPy small-sample metric warnings remain. These checks establish
engineering behavior only.

## Training boundary

Not completed: new multi-horizon training, a feature/scaler-identical production
inference adapter, real eligible forecast exports, calibrated financial replay,
learned fusion, or live deployment. The current deliverable is the joint strategy,
forecast contract and tested offline integration; the forecast producer still
requires research-side work and new weights.

Before requesting a paid run, verify the available machine and cache, reconcile
the research repository with its newer teammate commits, then implement the
multi-target dataset/model/export path there. Freeze horizons, train/validation/test
dates, longest-label purge, target scaling, two seeds and the early-stopping metric
before training. Compare fixed fusion weights without fitting them on the final
14-day test. Report DailyIC by horizon and asset/day, price-candidate conditional
IC with coverage, forecast disagreement, fills and hold distributions as well as
PnL. Machine choice and a new compute budget must come from the user.
