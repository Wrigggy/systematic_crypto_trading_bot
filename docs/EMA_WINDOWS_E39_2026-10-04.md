# E39: independent 14-day EMA windows and a softer trend gate

## Scope and data

The user requested more historical data, multiple 14-day windows and revised
trading logic. This is local CPU-only research. No paid machine, new model
training, API order, production switch, instance lifecycle action or credential
transfer was performed.

Downloaded 36 official Binance spot monthly 1-minute archives for BTCUSDT,
XRPUSDT and BNBUSDT, September 2025 through August 2026 inclusive. There are
525,600 bars per asset, 1,576,800 asset-minute rows total, 68,273,890 downloaded
ZIP bytes and approximately 143 MiB including parsed arrays. Raw 12-column bars
are retained. ZIP CRC, finite OHLCV, exact minute grid, timestamp units and
month boundaries pass; no missing minutes are filled and no full-file hashes
are required. Source: [Binance public-data documentation](https://github.com/binance/binance-public-data).

The 25 complete, non-overlapping windows cover September 10, 2025 00:00 through
August 26, 2026 00:00 exclusive, Singapore time. Each starts at 100,000 cash and
zero positions, with eight days of causal indicator warmup. First 17 and last
eight are descriptive chronological blocks, not an adaptive selection process
or untouched holdout. Incomplete tails are not padded. Flat state resets do not
make market regimes statistically independent.

## Frozen policy modification

- Retain completed-five-minute indicators: fast EMA half-life 6h, slow 24h,
  residual-volatility half-life 24h and seven-day indicator warmup.
- Strict control still requires fast EMA above slow EMA and above its level
  one hour ago. The new opt-in `trend_mode: soft` removes those hard entry vetoes.
- Soft mode allocates 10% NAV when fast EMA is above slow, 5% otherwise; no
  increase above the existing position/exposure limits. This changes both
  eligibility and sizing, not a one-factor slope ablation.
- Keep 0.5-sigma price pullback, recovery versus the prior completed five-minute
  close, and at least 20 bps distance to the entry-time EMA target. No forced
  activity trades. The target is frozen at entry, not moved using future data.
- Five-minute decisions; price-only history uses one-minute closed-price/risk
  updates. Maximum hold 2h, 3% hard/trailing stop, 5% daily drawdown brake.
- Maker entries/ordinary exits expire after 120 seconds. Risk/deadline and
  timed-out exits may use taker. Fees are 5/10 bps, zero added slippage. A later
  completed minute close must penetrate the limit by 2 bps, or 5 bps in the
  conservative fill sensitivity. This is not a claim about actual bid/ask,
  spread, queues, latency or maker certainty.
- Shared five-per-minute execution-request budget remains. Cached public-feed
  requests are not included. New entries stop 130 minutes before window end;
  budgeted cancellation/liquidation begins ten minutes before end. No free
  terminal fill or inherited inventory is permitted.

The broad study explicitly uses `price_only` mode. E38 trained on much of this
history; inserting its forecasts into earlier windows would be leakage.
The production combined-signal default and fail-closed missing-forecast behavior
are unchanged. Separate eligible-date combined replays are described below.

## Completed 100-window results

All figures below are per independent 14-day account, not annualized returns or
a continuously compounded wealth path. All 100 rows start and finish flat and
reconcile cash against the fill ledger, maximum error 0.000000000121 USDT.

| Price-only policy | Mean net return | Median | Positive windows | Mean completed trades | Mean active days | At least 8 days | Worst window DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict EMA, maker | -0.0447% | 0.0000% | 7/25 | 1.32 | 1.36 | 0/25 | 0.3959% |
| Soft EMA, maker 2bps (primary) | -0.8630% | -0.7686% | 1/25 | 48.36 | 12.60 | 25/25 | 2.0345% |
| Soft EMA, maker 5bps | -0.6943% | -0.6717% | 1/25 | 33.76 | 11.88 | 25/25 | 1.8249% |
| Soft EMA, taker | -1.4227% | -1.5104% | 1/25 | 87.64 | 13.24 | 25/25 | 2.6583% |

Primary early-17 mean is -0.9263%, later-eight -0.7286%; all eight later primary
windows are negative. No policy was selected after viewing these results.
The conservative scenario is not a guaranteed pessimistic PnL bound: changing
which orders fill changes the entire path, sometimes reducing losses.

Primary completes 1,209 trades, 40.94% winners, profit factor 0.3978, median
hold 120 minutes, 88.50% within 1--2h. Average time-weighted NAV exposure is
1.793%. Its much higher activity also changes risk exposure; tiny strict-control
drawdown is not evidence of superior capital efficiency.

Across the 25 separately funded accounts, primary gross same-fill PnL is
-9,329.83, fees 12,245.75, net -21,575.57 USDT. These are diagnostic sums, not
one account's realized historical wealth path. Both gross signal selection and
costs are problems. All three assets lose before fees. Both full-size and
half-size trend groups lose before fees; removing only the downtrend trades is
not supported as a sufficient repair.

899/1,209 primary trades (74.36%) reach the two-hour deadline; their combined
net PnL is -25,859.71. Recovery/exit-timeout trades contribute +10,298.11, while
30 price stops lose -6,013.97. Exit groups are outcome-conditioned diagnostics,
not filters that could have been selected at entry. The six-hour EMA target may
be mismatched to a two-hour holding cap; that is a next-test hypothesis, not a
demonstrated remedy or reason to remove risk deadlines.

## Model boundary and engineering repeats

The same soft mode is also replayed with the existing E38 two-seed ensemble on
the already inspected September 3--16 period and the original five-second price
proxy. This reuses predictions and does not extend model training or prove
multi-window model generalization. The initial four replays used legacy terminal
mark-to-market handling; taker and price-only ended with inventory. The strict
cash-ledger auditor rejected those as flat-end evidence. All four are retained
at `logs/ema_soft_model_E39_20261004/` and superseded by the explicitly budgeted
flat-end run `logs/ema_soft_model_E39_20261004_flat/`. They must not be added as
independent financial tests. The final four model-context results are in
`assets/ema_windows_E39_20261004/model_comparison.json`.

| Corrected September model-context case | Net return | Trades | Active days | Median hold |
|---|---:|---:|---:|---:|
| Soft EMA + E38, maker 2bps | -0.8403% | 38 | 12 | 71.93 min |
| Soft EMA + E38, maker 5bps | -0.6105% | 27 | 10 | 76.10 min |
| Soft EMA + E38, taker | -1.0326% | 71 | 12 | 70.00 min |
| Soft EMA without model, maker 2bps | -0.9897% | 60 | 13 | 120.00 min |

All four corrected rows finish flat and reconcile. The model improves this one
soft-policy replay by about 0.1494 percentage points versus its price-only
control, but both lose; this is not evidence of stable model value. E38's prior
strict+model primary was -0.1365%, three trades and two days, and did not meet
activity. Increased turnover is not a successful replacement. Do not compare
one-minute historical rows directly to this five-second replay as a controlled
resolution ablation.

## Engineering verification

The full bot suite passes 308 tests with one existing skip and four existing
NumPy warnings. New tests cover milliseconds/microseconds, duplicate/missing
bars, causal minute availability, non-overlapping windows, soft trend sizing
reaching order quantity, retained recovery checks, budgeted final liquidation,
flat restart and the auditor rejecting an open terminal position. An initial
test stub lacked the new sizing method; it was corrected. A new sizing fixture
initially omitted a price observation and was corrected before acceptance.
The original legacy-end model audit failed as described above; its artifacts
remain, and the corrected financial replay is separately recorded.

## Reproduction and artifacts

```bash
python -m scripts.download_minute_history --out data/historical/minute_202509_202608
python -m scripts.run_ema_windows --data data/historical/minute_202509_202608 --out logs/NEW_IMMUTABLE_RUN --workers 2
python -m scripts.summarize_ema_windows --run logs/ema_windows_E39_20261004 --model-run logs/ema_soft_model_E39_20261004_flat --out docs/assets/ema_windows_E39_20261004 --plot
```

Plotting additionally requires matplotlib; the existing research environment has
it. Raw archives/arrays and complete compressed event/equity logs stay ignored
locally. Tracked evidence contains the frozen plan, all 100 row summaries,
aggregate results, CSV, ledger audit, plot and four corrected model-context rows.
The companion research ledger indexes the same evidence; it is not another run.

- [All windows and configurations](assets/ema_windows_E39_20261004/comparison.json)
- [Window CSV](assets/ema_windows_E39_20261004/windows.csv)
- [Cash and trade attribution audit](assets/ema_windows_E39_20261004/audit.json)
- [Window return/activity plot](assets/ema_windows_E39_20261004/window_returns.png)

This rejects the soft-EMA policy as a deployment candidate despite solving
activity coverage. A proper multi-window model test requires chronologically
trained fold checkpoints and eligible predictions, not backdating E38. No new
GPU work or automatic successor experiment is dispatched.
