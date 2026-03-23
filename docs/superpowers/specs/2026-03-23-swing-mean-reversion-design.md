# Swing Mean-Reversion Strategy Design

**Date:** 2026-03-23
**Profile name:** `swing_mean_reversion_v1`
**Status:** Approved

## Problem Statement

All existing strategy profiles (capital_preservation, regime_trend, trend_pullback, core_satellite) suffer from:
- Tight stop losses (0.8%) causing frequent whipsaw exits
- Low win rates (23%) with unfavorable payoff ratios (avg loss 2x avg win)
- Excessive trade churn (26 trades in 10 days on 6 coins with 2-position cap)
- Mean return near zero or negative across 90-day rolling window backtests

The competition format (10-day, long-only, crypto) rewards strategies that:
- Trade less frequently but with higher conviction
- Buy at statistical extremes and sell at mean-reversion targets
- Use wider stops to avoid noise-driven exits

## Strategy Overview

**Philosophy:** Wait for price to reach statistical extremes (lower Bollinger Band) while the broader trend remains intact, then enter long and ride the mean reversion back toward the middle/upper band.

**Universe:** 12 major coins (BTC, ETH, SOL, BNB, XRP, LINK, ADA, AVAX, DOT, DOGE, LTC, UNI)
**Hold period:** 4-24 hours (target ~8h average)
**Max concurrent positions:** 3
**Max portfolio exposure:** 50%
**Max single position:** 12%
**Stop loss:** 2.5% from entry
**Target risk-reward:** TP1 at 1.2R (3% partial exit), TP2 at 2.0R (5% full exit) vs 2.5% SL

## Entry Logic

### Primary Signal: Bollinger Band Dip Buy

Entry triggers when ALL conditions are met on 15-minute resampled candles:

1. **Bollinger Band touch:** `close <= lower_band` (20-period, 2.0σ)
2. **RSI oversold:** `RSI(14) < 35` (on 15m candles)
3. **Higher-TF trend intact:** 1h EMA(12) > EMA(26) — confirms dip within uptrend
4. **Volume confirmation:** current 15m volume > 1.2x 20-period average
5. **Volatility filter:** realized volatility < 0.025 (avoids panic selloffs)

### New Factor: `bollinger_mean_reversion`

Added as an explicit call in `FactorEngine.evaluate()`'s hard-coded observation list (there is no plugin/registry — each factor is called inline).

The method signature: `_bollinger_mean_reversion(self, features, candles_15m, candles_1h)`.

**1h EMA computation:** The factor computes 1h EMA(12) and EMA(26) inline from `candles_1h` closes using numpy EMA (same approach as `_trend_alignment`).

**Bollinger Band computation:** Computed inline from `candles_15m` closes: SMA(20), then upper/lower = SMA ± 2σ. Uses `max(band_width * 0.5, 1e-6)` as denominator to guard against division by zero during Bollinger squeezes.

**Minimum band width filter:** Requires `bb_width > 0.02` to avoid low-conviction entries during Bollinger squeezes where tiny price moves trigger touches.

- **BULLISH** when price <= lower band AND RSI < 35 AND 1h trend bullish AND bb_width > 0.02
  - Strength = `clamp((lower_band - close) / max(band_width * 0.5, 1e-6))` — deeper dip = stronger signal
- **BEARISH** when price >= upper band (contributes to exit scoring)
  - Strength = `clamp((close - upper_band) / max(band_width * 0.5, 1e-6))`
- **NEUTRAL** otherwise (including during band squeezes with bb_width <= 0.02)

### New Features Required

Add to `FeatureVector` (computed on 15m resampled candles):
- `bb_upper`: Upper Bollinger Band (20-period, 2σ)
- `bb_middle`: Middle Bollinger Band (20-period SMA)
- `bb_lower`: Lower Bollinger Band (20-period, 2σ)
- `bb_width`: Band width = (upper - lower) / middle
- `bb_pctb`: %B = (close - lower) / (upper - lower)

The existing `MultiResampler` already produces 15m candles. The feature extractor computes Bollinger Bands on these resampled candles.

### Cross-Symbol Ranking

When multiple coins signal simultaneously:
1. Rank by `bb_pctb` ascending (most oversold first)
2. Break ties by RSI ascending (lower RSI = more extreme)
3. Enter top 1 per cycle (prevent correlated entries)

### Order Type Selection

- RSI < 25 (deeply oversold) → MARKET order (urgency)
- RSI 25-35 → LIMIT order at current price - 3 bps offset (get better fill at extreme)

## Exit Logic

### Three-Tier Graduated Exit

**IMPORTANT: The existing `_check_graduated_exit` in `StrategyLogic.on_signal()` uses alpha_score and triggers when alpha DROPS below thresholds. This is the wrong direction for mean-reversion exits, which should trigger when exit_score RISES above thresholds (price reaches BB middle/upper band).**

**Solution:** Add a new method `_check_graduated_factor_exit()` to `StrategyLogic` that operates on `FactorSnapshot.exit_score` in the `on_factors()` HOLDING branch. This method:
1. Checks `exit_score` against graduated tiers (ascending thresholds)
2. Sells partial or full position when exit_score rises above each tier threshold
3. Tracks which tiers have been triggered (reusing `_exit_tier_reached`)

This means **`strategy/logic.py` MUST be modified** (moved to "Files to Modify" below).

| Tier | Trigger | Action |
|------|---------|--------|
| TP1 | `exit_score >= 0.40` (BB middle band reached) | Sell 50% of position |
| TP2 | `exit_score >= 0.70` (BB upper band reached) | Sell remaining 50% |
| Stop | Price drops 2.5% from entry | Sell 100% (hard stop via RiskShield) |
| Time | Position held > 24h with no TP hit | Sell 100% (avoid bagholding) |
| Trend | 1h EMA(12) crosses below EMA(26) → `blocker_score >= 0.30` | Sell 100% (thesis invalidated) |

### 24h Time-Based Exit

The time-based exit is implemented in the new `_check_graduated_factor_exit()` method in `StrategyLogic`. It tracks `_entry_time` (set on fill) and compares against the current factor timestamp. If `timestamp - entry_time > 24h` and no TP tier has been reached, emit a full exit intent.

### Trailing Stop Interaction

**Note:** The `trailing_stop_pct: 0.025` can prematurely exit mean-reversion trades during choppy recovery paths toward the BB middle band. For this profile, set `trailing_stop_pct: 0.035` (wider than the 2.5% hard stop) so that the trailing stop only engages after the position is already well in profit. The hard stop at 2.5% handles the downside.

### Exit Factor Integration

The `bollinger_mean_reversion` factor in BEARISH mode (price at upper band) contributes to `exit_score`. Combined with the existing `overextension_exit` factor, this creates natural sell pressure when price reaches the upper band.

The trend reversal exit is handled by the existing `trend_alignment` factor going BEARISH, which contributes to `blocker_score` and triggers exit when above threshold.

## Position Sizing

### Kelly-Scaled Fixed Fraction

- **Base size:** 5% of NAV
- **Kelly-scaled range:** 3% (low conviction) to 12% (max conviction)
- **Conviction intensity** = how deep below lower band + RSI extremity:
  - `conviction = 0.6 * bb_depth_score + 0.4 * rsi_extremity_score`
  - `bb_depth_score = clamp((lower_band - close) / max(band_width * 0.5, 1e-6))`
  - `rsi_extremity_score = clamp((35 - RSI) / 15)` (RSI 35→0, RSI 20→1.0)
- **Kelly params:** `kelly_fraction: 0.35`, `estimated_win_rate: 0.58`, `estimated_payoff: 1.8`

### Risk Limits

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_portfolio_exposure` | 0.50 | Allow up to 50% deployed |
| `max_single_exposure` | 0.12 | Single position cap |
| `max_active_positions` | 3 | Concentration + diversification balance |
| `top_n_entries_per_cycle` | 1 | One entry per decision cycle |
| `trailing_stop_pct` | 0.025 | 2.5% trailing stop |
| `atr_stop_multiplier` | 3.0 | Wide ATR stop (3x ATR) |
| `daily_drawdown_limit` | 0.04 | 4% daily drawdown circuit breaker |
| `base_stop_loss_pct` | 0.025 | 2.5% hard stop |

## Factor Weights

The strategy de-emphasizes momentum/breakout factors and heavily weights the new Bollinger factor:

| Factor | Weight | Rationale |
|--------|--------|-----------|
| `bollinger_mean_reversion` | 0.40 | Primary entry signal |
| `trend_alignment` | 0.25 | Higher-TF trend filter |
| `volume_confirmation` | 0.12 | Volume spike at dip |
| `volatility_regime` | 0.10 | Avoid volatile regimes |
| `market_regime` | 0.08 | Market breadth context |
| `liquidity_balance` | 0.05 | Order book confirmation |
| `momentum_impulse` | 0.00 | Disabled (anti-mean-reversion) |
| `breakout_confirmation` | 0.00 | Disabled (not relevant) |
| `perp_crowding` | 0.00 | Disabled (minimal impact at this TF) |
| `pullback_reentry` | 0.00 | Disabled (different entry logic) |
| `overextension_exit` | 0.00 | Replaced by BB upper band exit |

## Config Profile

```yaml
# Strategy profile: swing_mean_reversion_v1
# Added to STRATEGY_PROFILES dict in main.py or config/default.yaml

swing_mean_reversion_v1:
  symbols:
    - BTC/USDT
    - ETH/USDT
    - SOL/USDT
    - BNB/USDT
    - XRP/USDT
    - LINK/USDT
    - ADA/USDT
    - AVAX/USDT
    - DOT/USDT
    - DOGE/USDT
    - LTC/USDT
    - UNI/USDT

  alpha:
    engine: rule_based
    multi_timeframes: [15, 60]  # Required: 15m and 1h candles for BB and trend filter

  strategy:
    use_model_overlay: false
    position_size_pct: 0.05
    base_size_pct: 0.03
    max_size_pct: 0.12
    kelly_fraction: 0.35
    estimated_win_rate: 0.58
    estimated_payoff: 1.80
    confirmation_bars: 1          # single bar confirmation (BB touch is already extreme)
    min_entry_score: 0.65
    max_blocker_score: 0.30
    min_exit_score: 0.35
    min_supporting_factors: 2
    min_supporting_categories: 2
    require_trend_alignment: true
    urgent_entry_score: 0.90
    signal_horizon_minutes: 480   # 8h signal horizon
    exit_horizon_minutes: 60      # 1h exit horizon
    base_stop_loss_pct: 0.025     # 2.5% stop loss
    take_profit_1_rr: 1.2         # TP1 at 1.2R = 3%
    take_profit_2_rr: 2.0         # TP2 at 2.0R = 5%
    neutral_entry_size_multiplier: 0.50
    risk_off_entry_size_multiplier: 0.00
    min_volatility_size_multiplier: 0.30
    max_active_positions: 3
    top_n_entries_per_cycle: 1
    min_volume_ratio: 1.20
    min_order_book_imbalance: 1.02
    max_funding_rate: 0.0005
    max_taker_ratio: 1.20
    max_open_interest_change: 0.03
    max_volatility: 0.025

    # Bollinger-specific parameters
    bb_period: 20
    bb_std_dev: 2.0
    bb_rsi_oversold: 35
    bb_rsi_deeply_oversold: 25
    bb_max_hold_hours: 24
    bb_min_width: 0.02          # Minimum BB width to avoid squeeze false signals

    factor_weights:
      bollinger_mean_reversion: 0.40
      trend_alignment: 0.25
      volume_confirmation: 0.12
      volatility_regime: 0.10
      market_regime: 0.08
      liquidity_balance: 0.05
      momentum_impulse: 0.00
      breakout_confirmation: 0.00
      perp_crowding: 0.00
      pullback_reentry: 0.00
      overextension_exit: 0.00

    exit_tiers:
      - threshold: 0.40    # When exit_score > 0.40 (BB middle band reached)
        sell_pct: 0.50      # Sell 50%
      - threshold: 0.70    # When exit_score > 0.70 (BB upper band reached)
        sell_pct: 1.00      # Sell remaining 100%

  regime:
    enabled: true
    risk_on_threshold: 0.25
    neutral_threshold: 0.08
    breadth_min_symbols: 6

  risk:
    max_portfolio_exposure: 0.50
    max_single_exposure: 0.12
    trailing_stop_pct: 0.035    # Wider than hard stop to avoid premature exits during choppy recovery
    atr_stop_multiplier: 3.0    # Wide ATR stop; may be wider than hard stop in low-vol, which is intentional
    daily_drawdown_limit: 0.04
    max_orders_per_minute: 6

  trend:
    vol_target_floor: 0.015
```

## Implementation Plan (High Level)

### Files to Modify

1. **`features/extractor.py`** — Add `compute_bollinger_bands()` method. Add BB fields to `FeatureVector`.
2. **`core/models.py`** — Add `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`, `bb_pctb` fields to `FeatureVector`. These are factor-engine-only fields and should NOT be added to `FEATURE_NAMES` (LSTM feature vector).
3. **`strategy/factor_engine.py`** — Add `_bollinger_mean_reversion()` factor method. Add explicit call in `evaluate()` observation list (no registry — inline call like all other factors).
4. **`strategy/logic.py`** — Add `_check_graduated_factor_exit()` method for factor-score-based graduated exits (existing `_check_graduated_exit` only works on alpha_score in wrong direction). Add `_entry_time` tracking for 24h time-based exit.
5. **`config/default.yaml`** — Add `swing_mean_reversion_v1` profile section. Ensure `alpha.multi_timeframes: [15, 60]` is present.
6. **`main.py`** — Add profile to `STRATEGY_PROFILES` dict (if profiles are stored there).

### Files Unchanged

- `strategy/monitor.py` — Already supports cross-symbol ranking, position limits, entry gating.
- `risk/risk_shield.py` — Already supports trailing stop, ATR stop, circuit breaker.
- `execution/` — No changes needed.
- `data/` — `MultiResampler` already provides 15m and 1h candles (requires `alpha.multi_timeframes: [15, 60]` in config).

### Backtest Validation

After implementation, validate using the backtest harness:
1. Single-run on the latest 10-day window (Mar 7-17)
2. Rolling 90-day window sweep (9 × 10-day windows) comparing against existing profiles
3. Target metrics: positive mean return, >50% win rate, profit factor > 1.5, max DD < 3%

## Expected Characteristics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Trades per 10 days | 8-15 | Selective entries at BB extremes |
| Win rate | 55-65% | Mean reversion has natural edge at extremes |
| Avg win / avg loss | 1.5-2.0x | TP at 3-5% vs SL at 2.5% |
| Profit factor | >1.5 | win_rate × payoff_ratio > 1 |
| Max drawdown | <3% | Wide stops + position limits |
| Sharpe ratio | >0 | Positive returns with controlled vol |
