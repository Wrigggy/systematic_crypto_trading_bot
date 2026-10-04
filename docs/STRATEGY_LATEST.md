# Latest strategy: E40 entry-quality mean reversion plus multi-horizon forecasts

Updated October 5, 2026, Asia/Singapore. Status: implemented and replay-tested
research candidate, NOT deployment-approved. It improves inspected-period losses
but has not established positive net expectancy. No live orders or cloud training
were performed. Older E37/E38/E39 results are preserved, not overwritten.

## Configuration and executable path

- Full resolved model-strategy configuration: `config/strategy_latest.yaml`.
- Small frozen entry overlay: `config/entry_quality_v1.yaml`.
- Strategy: `strategy/hybrid.py`, `strategy/fusion.py`,
  `strategy/ema_pullback.py`, `strategy/entry_quality.py`.
- End-to-end offline runner: `python -m scripts.run_entry_quality`.
- Current evidence: [E40 results](ENTRY_QUALITY_E40_2026-10-05.md).

The full configuration is generated from the executed primary model case, not a
different post-hoc winner. It includes an exact forecast/preprocessing identity.
`replay.events_path` is deliberately unset: the generic replay CLI requires an
explicit causal event stream; the study runner supplies audited adapters. The
regular live entry point still rejects the hybrid engine. Do not enable it by
removing this guard: a feature-exact live forecast and completed-flow adapter,
venue reconciliation and explicit trading approval are still required.

## Inputs, timestamps and normalization

Trade BTC/USDT, XRP/USDT and BNB/USDT long-only. This is directional execution,
not simultaneous two-sided market making. The E38 ensemble is reused, not retrained.

1. Price observations: historical windows use completed minute closes; eligible
   model replay uses the original previous-second-open proxy sampled every five
   seconds. They are different data-resolution experiments, not interchangeable.
2. EMA context: completed five-minute boundaries only; fast/slow half-lives6h/24h,
   residual-volatility half-life24h,2,016 completed bars of warmup. The eight-day
   replay prefix is indicator history only, never pre-existing capital/positions.
3. Trade flow: exact completed-minute close, total quote volume and taker-buy
   quote volume. This is exchange-reported aggressor-side volume, not an L2 book
   or a candle-color estimate. The minute is available only after its end.
4. Forecasts: existing15/30/60/120/240minute latter-half-TWAP heads, inverse-scaled
   model outputs. The versioned packet checks eligibility, timestamps, missing
   heads, staleness and preprocessing identity. Per-asset/head rolling24h score
   normalization excludes the current prediction. Original model input scalers
   remain train-only frozen per-asset scalers; they are not rolling live fits.

Missing flow minutes reset its20minute window. Duplicate identical observations
are idempotent; conflicting/out-of-order or future observations are rejected.
Zero-volume groups, less than20continuous minutes or flow age60seconds and above
block new entries. A missing model never silently becomes a price-only strategy.

## Entry rules, evaluated every five minutes

All the following must hold; a weak model cannot be compensated by a deeper dip.

1. Price is below fast EMA by more than0.5residual standard deviations, but no
   more than1.5standard deviations. More extreme displacement is rejected.
2. The latest completed five-minute close is above the previous completed close.
   The entry-time EMA is at least20bps above the observed entry reference price.
   This is geometric room, NOT a calibrated expected-profit assertion.
3. Recent five-minute taker-buy quote share is at least50percent and improves
   by at least5percentage points over the preceding, non-overlapping15minutes.
   The completed-minute price also rises over those recent five minutes.
   Shares are quote-volume-weighted, not an average of minute percentages.
4. Price-rule strength is capped at1.5, rather than letting ever-deeper dips
   increase strength up to the previous3.0cap.
5. Model composite uses head weights10/25/40/20/5percent, score scale1.5 and a
   scaled minimum0.5; equivalently its standardized composite must reach0.75.
   This is a relative score requirement, not a predicted-return or probability
   threshold. The older long-head nonnegative hard veto stays disabled.
6. Rule/model weights remain0.5/0.5, joint threshold0.75. Joint strength ranks
   simultaneous candidates for shared cash/slots. Price eligibility and the
   independent model-quality floor remain mandatory; missing data is rejected.

The broad25-window study explicitly disables the model as a named price-only
diagnostic. It tests the displacement guard, ranking cap and flow confirmation;
it does NOT test the independent model floor or historical model generalization.
No learned quality classifier, all-head agreement veto, new market-state veto,
EMA half-life optimization or future-looking target is implemented in E40.

## Position sizing and exits

Normal target is10percent NAV; fast EMA at or below slow EMA reduces it to5percent.
The one-hour EMA slope is not an entry veto. Retain15percent single-asset and
50percent portfolio caps, three positions maximum, available-cash checks and
the existing shared pending-order restrictions.

Holding starts on actual fill, not order submission. The recovery target is the
entry-time EMA and stays frozen for that position. There is no mandatory minimum
hold. Exit at recovery, a3percent hard/trailing stop, the two-hour holding
deadline, or a5percent intraday drawdown brake. After one hour, persistent weak
60/120minute model support for five minutes also exits the combined strategy.
Price-only controls do not use this model review. Stops/deadlines remain active
even if model or flow observations disappear.

New flow/fusion/EMA invalidation cancels pending buys before the next scheduled
entry decision. It does not cancel necessary sells or suppress risk liquidation.
Order acknowledgement is not a fill; partial, unknown and cancelled orders must
be reconciled before inventory replacement.

## Execution and competition accounting

Prefer maker entries and ordinary recovery exits, with120second order/exit wait.
Risk/deadline exits and ordinary exits that time out can become taker; this is not
a maker-only promise. Existing venue execution uses fresh best bid/ask and passive
tick rounding; LIMIT is not a proof of maker classification or guaranteed fill.

Replay fees are5bps maker and10bps taker, with zero added slippage. Maker fills
require a later observed price to penetrate the limit by2bps; a5bps sensitivity
uses the same strategy. Those2/5bps are SIMULATED FILL CONDITIONS, not different
fees or live quote offsets. Spread, queues, partial fills and latency are not
calibrated to Roostoo; use actual Role/commission receipts for live accounting.

The shared budget is five modeled execution HTTP requests per minute, including
quotes, placement, status and cancellation, with a risk reserve. Cached historical
public-feed HTTP is excluded, so this does not certify a complete live data/API
request schedule. No per-tick cancel/replace loop bypasses the budget.

Every14day experiment starts with100,000cash and zero holdings. Entries stop in
the final130minutes; budgeted closeout begins in the final10minutes. Eight active
days mean days with actual simulated fills in UTC+8, not emitted signals. There
are no forced trades to satisfy activity. Audit rejects unclosed inventory or
unreconciled fills instead of fabricating cost-free terminal liquidation.

## Current acceptance status

The primary historical price-only variant averages-0.6107percent per14days with
37.12completed trades and12.2active days;25/25windows pass activity but only1/25
is profitable. Existing-model primary on inspected September3--16 is-0.1224percent,
25trades/11days,median81.1minutes,0.2874percent maximum drawdown. It is an
improvement from E39, not a profitable or independently validated strategy.

No live configuration is switched, no new GPU run is scheduled, and no machine
lifecycle changes are authorized. The next research question is durable net edge,
not simply how to create more signals. A genuine multi-window model study needs
eligible chronological fold forecasts rather than applying E38 to its training
history. Existing viewed dates cannot be relabeled as untouched holdout.
