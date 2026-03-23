# Swing Mean-Reversion Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Bollinger Band mean-reversion swing strategy (`swing_mean_reversion_v1`) to the trading competition system and validate it via backtest.

**Architecture:** Add Bollinger Band features to the extractor, create a new `bollinger_mean_reversion` factor in the factor engine, add graduated factor-based exits to the strategy logic, register the profile in `main.py`, fix the backtest monitor's missing buy-candidate ranking, and run comparative backtests.

**Tech Stack:** Python, numpy, pydantic, pytest, asyncio

**Spec:** `docs/superpowers/specs/2026-03-23-swing-mean-reversion-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `core/models.py` | No change | BB data stored in existing `FeatureVector.raw` dict — no schema change needed |
| `features/extractor.py` | Modify | Add `compute_bollinger_bands()` static method |
| `strategy/factor_engine.py` | Modify | Add `_bollinger_mean_reversion()` factor; add call in `evaluate()` |
| `strategy/logic.py` | Modify | Add `_check_graduated_factor_exit()` for factor-score-based graduated exits; add `_entry_time` for time-based exit |
| `main.py` | Modify | Add `swing_mean_reversion_v1` to `STRATEGY_PROFILES` |
| `tests/test_factor_engine.py` | Modify | Add tests for new bollinger factor |
| `tests/test_strategy.py` | Modify | Add tests for graduated factor exit and time-based exit |
| `tests/test_features.py` | Modify | Add test for Bollinger Band computation |
| `../trading_competition_backtest/tc_backtest/backtest_monitor.py` | Modify | Fix missing buy-candidate ranking (port from parent `StrategyMonitor`) |

---

### Task 1: Add Bollinger Band computation to FeatureExtractor

**Files:**
- Modify: `features/extractor.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing test for `compute_bollinger_bands`**

```python
# In tests/test_features.py, add:
class TestBollingerBands:
    def test_bollinger_bands_basic(self):
        """BB(20,2) on constant-price series should have zero width."""
        from features.extractor import FeatureExtractor
        # 25 candles at price=100
        closes = [100.0] * 25
        upper, middle, lower = FeatureExtractor.compute_bollinger_bands(closes, period=20, num_std=2.0)
        assert middle == pytest.approx(100.0)
        assert upper == pytest.approx(100.0)  # zero std → bands collapse
        assert lower == pytest.approx(100.0)

    def test_bollinger_bands_trending(self):
        """BB on trending data should have middle = SMA, bands symmetric."""
        from features.extractor import FeatureExtractor
        closes = [float(i) for i in range(1, 26)]  # 1..25
        upper, middle, lower = FeatureExtractor.compute_bollinger_bands(closes, period=20, num_std=2.0)
        # SMA of last 20 values (6..25) = 15.5
        assert middle == pytest.approx(15.5)
        assert upper > middle
        assert lower < middle
        assert (upper - middle) == pytest.approx(middle - lower, abs=0.01)

    def test_bollinger_bands_insufficient_data(self):
        """Return zeros when not enough data."""
        from features.extractor import FeatureExtractor
        closes = [100.0] * 5
        upper, middle, lower = FeatureExtractor.compute_bollinger_bands(closes, period=20, num_std=2.0)
        assert upper == 0.0 and middle == 0.0 and lower == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/test_features.py::TestBollingerBands -v`
Expected: FAIL — `AttributeError: type object 'FeatureExtractor' has no attribute 'compute_bollinger_bands'`

- [ ] **Step 3: Implement `compute_bollinger_bands` and add BB data to `raw` dict**

In `features/extractor.py`, add static method after `compute_volume_zscore`:

```python
@staticmethod
def compute_bollinger_bands(
    closes: List[float], period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float]:
    """Bollinger Bands: (upper, middle, lower).

    Returns (0, 0, 0) if insufficient data.
    """
    if len(closes) < period:
        return 0.0, 0.0, 0.0
    window = closes[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window))
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower
```

Also update `extract()` to include BB data in the `raw` dict. After the existing `raw = {...}` block (around line 80-95), add:

```python
# Bollinger Bands on the candle series (will use 15m resampled candles when available)
bb_upper, bb_middle, bb_lower = self.compute_bollinger_bands(
    closes, period=20, num_std=2.0
)
bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.0
bb_pctb = (
    (closes[-1] - bb_lower) / (bb_upper - bb_lower)
    if (bb_upper - bb_lower) > 1e-10
    else 0.5
)
raw["bb_upper"] = bb_upper
raw["bb_middle"] = bb_middle
raw["bb_lower"] = bb_lower
raw["bb_width"] = bb_width
raw["bb_pctb"] = bb_pctb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/test_features.py::TestBollingerBands -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/kevinwu/Coding/trading_competition
git add features/extractor.py tests/test_features.py
git commit -m "feat: add Bollinger Band computation to FeatureExtractor"
```

---

### Task 2: Add `bollinger_mean_reversion` factor to FactorEngine

**Files:**
- Modify: `strategy/factor_engine.py`
- Test: `tests/test_factor_engine.py`

- [ ] **Step 1: Write failing tests for the new factor**

```python
# In tests/test_factor_engine.py, add:

class TestBollingerMeanReversionFactor:
    def test_bullish_when_price_at_lower_band_and_rsi_oversold(self):
        """Factor should be BULLISH when close <= lower BB, RSI < 35, uptrend on 1h."""
        engine = FactorEngine({
            "strategy": {
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "bb_rsi_oversold": 35,
                "bb_min_width": 0.02,
                "factor_weights": {"bollinger_mean_reversion": 0.40},
            },
        })
        # Features with RSI=28 (oversold) and BB data showing price at lower band
        features = _feature_vector(
            rsi=28.0,
            raw={
                "breakout_distance": 0.0,
                "trend_slope": 0.001,
                "volume_zscore": 0.5,
                "bb_upper": 105.0,
                "bb_middle": 100.0,
                "bb_lower": 95.0,
                "bb_width": 0.10,
                "bb_pctb": -0.05,  # below lower band
            },
        )
        # 1h candles with uptrend (EMA12 > EMA26)
        candles_1h = _candles(95.0, 0.5, 30)  # rising prices
        snapshot = engine.evaluate(features, candles_1h=candles_1h)
        bb_obs = next(obs for obs in snapshot.observations if obs.name == "bollinger_mean_reversion")
        assert bb_obs.bias == FactorBias.BULLISH
        assert bb_obs.strength > 0.0

    def test_bearish_when_price_at_upper_band(self):
        """Factor should be BEARISH when close >= upper BB."""
        engine = FactorEngine({
            "strategy": {
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "bb_rsi_oversold": 35,
                "bb_min_width": 0.02,
                "factor_weights": {"bollinger_mean_reversion": 0.40},
            },
        })
        features = _feature_vector(
            rsi=72.0,
            raw={
                "breakout_distance": 0.0,
                "trend_slope": 0.001,
                "volume_zscore": 0.5,
                "bb_upper": 105.0,
                "bb_middle": 100.0,
                "bb_lower": 95.0,
                "bb_width": 0.10,
                "bb_pctb": 1.05,  # above upper band
            },
        )
        snapshot = engine.evaluate(features)
        bb_obs = next(obs for obs in snapshot.observations if obs.name == "bollinger_mean_reversion")
        assert bb_obs.bias == FactorBias.BEARISH
        assert bb_obs.strength > 0.0

    def test_neutral_during_band_squeeze(self):
        """Factor should be NEUTRAL when bb_width < min threshold (squeeze)."""
        engine = FactorEngine({
            "strategy": {
                "bb_min_width": 0.02,
                "factor_weights": {"bollinger_mean_reversion": 0.40},
            },
        })
        features = _feature_vector(
            rsi=28.0,
            raw={
                "breakout_distance": 0.0,
                "trend_slope": 0.001,
                "volume_zscore": 0.5,
                "bb_upper": 100.5,
                "bb_middle": 100.0,
                "bb_lower": 99.5,
                "bb_width": 0.01,  # squeeze: width < 0.02
                "bb_pctb": -0.05,
            },
        )
        snapshot = engine.evaluate(features)
        bb_obs = next(obs for obs in snapshot.observations if obs.name == "bollinger_mean_reversion")
        assert bb_obs.bias == FactorBias.NEUTRAL

    def test_neutral_when_rsi_not_oversold(self):
        """Factor should be NEUTRAL when RSI > oversold threshold even if at lower band."""
        engine = FactorEngine({
            "strategy": {
                "bb_rsi_oversold": 35,
                "bb_min_width": 0.02,
                "factor_weights": {"bollinger_mean_reversion": 0.40},
            },
        })
        features = _feature_vector(
            rsi=50.0,  # not oversold
            raw={
                "breakout_distance": 0.0,
                "trend_slope": 0.001,
                "volume_zscore": 0.5,
                "bb_upper": 105.0,
                "bb_middle": 100.0,
                "bb_lower": 95.0,
                "bb_width": 0.10,
                "bb_pctb": -0.05,  # at lower band but RSI not oversold
            },
        )
        snapshot = engine.evaluate(features)
        bb_obs = next(obs for obs in snapshot.observations if obs.name == "bollinger_mean_reversion")
        assert bb_obs.bias == FactorBias.NEUTRAL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/test_factor_engine.py::TestBollingerMeanReversionFactor -v`
Expected: FAIL — no observation named `bollinger_mean_reversion`

- [ ] **Step 3: Implement `_bollinger_mean_reversion` in FactorEngine**

In `strategy/factor_engine.py`:

1. Add config params in `__init__` (after the existing `_enable_overextension_exit` block):
```python
self._bb_period: int = strategy_cfg.get("bb_period", 20)
self._bb_std_dev: float = strategy_cfg.get("bb_std_dev", 2.0)
self._bb_rsi_oversold: float = strategy_cfg.get("bb_rsi_oversold", 35.0)
self._bb_rsi_deeply_oversold: float = strategy_cfg.get("bb_rsi_deeply_oversold", 25.0)
self._bb_min_width: float = strategy_cfg.get("bb_min_width", 0.02)
```

2. Add the factor to the observation list in `evaluate()` (insert after `self._volatility_regime(features)` on line ~129):
```python
self._bollinger_mean_reversion(features, candles_1h),
```

3. Add the method (after `_volatility_regime`):
```python
def _bollinger_mean_reversion(
    self,
    features: FeatureVector,
    candles_1h: Optional[List[OHLCV]] = None,
) -> FactorObservation:
    bb_upper = float(features.raw.get("bb_upper", 0.0))
    bb_middle = float(features.raw.get("bb_middle", 0.0))
    bb_lower = float(features.raw.get("bb_lower", 0.0))
    bb_width = float(features.raw.get("bb_width", 0.0))
    bb_pctb = float(features.raw.get("bb_pctb", 0.5))
    close = features.ema_fast  # proxy for current close from features

    # Use bb_pctb to detect position relative to bands
    # pctb <= 0 means at or below lower band, pctb >= 1 means at or above upper band

    # Check 1h trend: compute EMA(12) and EMA(26) on 1h closes
    hourly_trend_bullish = True  # default if no 1h data
    if candles_1h and len(candles_1h) >= 30:
        closes_1h = [c.close for c in candles_1h]
        ema12 = self._compute_ema_value(closes_1h, 12)
        ema26 = self._compute_ema_value(closes_1h, 26)
        hourly_trend_bullish = ema12 > ema26

    bias = FactorBias.NEUTRAL
    strength = 0.0

    if bb_width < self._bb_min_width or bb_middle <= 0:
        # Squeeze or no data — stay neutral
        pass
    elif bb_pctb <= 0.0 and features.rsi < self._bb_rsi_oversold and hourly_trend_bullish:
        # BULLISH: price at/below lower band, RSI oversold, uptrend intact
        denom = max(bb_width * 0.5, 1e-6)
        depth = max(0.0, -bb_pctb)  # how far below 0 (lower band)
        rsi_extremity = _clamp((self._bb_rsi_oversold - features.rsi) / 15.0)
        strength = _clamp(0.6 * min(depth / 0.5, 1.0) + 0.4 * rsi_extremity)
        bias = FactorBias.BULLISH
    elif bb_pctb >= 1.0:
        # BEARISH: price at/above upper band (for exit scoring)
        overshoot = bb_pctb - 1.0
        strength = _clamp(overshoot / 0.5)
        bias = FactorBias.BEARISH

    return FactorObservation(
        symbol=features.symbol,
        name="bollinger_mean_reversion",
        category="entry_timing",
        timestamp=features.timestamp,
        bias=bias,
        strength=strength if bias != FactorBias.NEUTRAL else 0.0,
        value=bb_pctb,
        threshold=0.0,
        horizon_minutes=480,
        expected_move_bps=80.0 + 200.0 * strength,
        thesis=(
            f"BB %B={bb_pctb:.2f}, width={bb_width:.4f}, RSI={features.rsi:.1f}, "
            f"1h_trend={'up' if hourly_trend_bullish else 'down'}"
        ),
        invalidate_condition="Price returns to middle band or 1h trend reverses",
        metadata={
            "bb_pctb": bb_pctb,
            "bb_width": bb_width,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "rsi": features.rsi,
            "hourly_trend_bullish": hourly_trend_bullish,
        },
    )

@staticmethod
def _compute_ema_value(values: list[float], period: int) -> float:
    """Compute EMA of a list, return final value."""
    if not values:
        return 0.0
    multiplier = 2.0 / (period + 1)
    ema = values[0]
    for val in values[1:]:
        ema = val * multiplier + ema * (1 - multiplier)
    return ema
```

4. Update the default `_factor_weights` dict in `__init__` to include:
```python
"bollinger_mean_reversion": 0.0,  # default off; enabled by profile
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/test_factor_engine.py::TestBollingerMeanReversionFactor -v`
Expected: PASS

- [ ] **Step 5: Run all existing factor engine tests to ensure no regressions**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/test_factor_engine.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/kevinwu/Coding/trading_competition
git add strategy/factor_engine.py tests/test_factor_engine.py
git commit -m "feat: add bollinger_mean_reversion factor to FactorEngine"
```

---

### Task 3: Add graduated factor-based exits and time-based exit to StrategyLogic

**Files:**
- Modify: `strategy/logic.py`
- Test: `tests/test_strategy.py`

- [ ] **Step 1: Write failing tests for graduated factor exit**

```python
# In tests/test_strategy.py, add these tests.
# IMPORTANT: Reuse the existing _factor_snapshot() helper already in this file
# (defined at line ~79, accepts keyword args: ts, regime, entry_score, exit_score,
# blocker_score, observations, supporting_factors, blocking_factors).
# Do NOT redefine _factor_snapshot — it would shadow the existing one and break tests.
#
# The existing file already imports: datetime, Position, PortfolioSnapshot, Side,
# StrategyState, StrategyLogic, FactorBias, FactorObservation, FactorSnapshot.

class TestGraduatedFactorExit:
    @staticmethod
    def _holding_portfolio(cash=50000.0, qty=1.0, price=100000.0):
        """Helper for tests needing a HOLDING portfolio."""
        pos = Position(
            symbol="BTC/USDT", quantity=qty, entry_price=price,
            current_price=price, state=StrategyState.HOLDING,
        )
        return PortfolioSnapshot(
            timestamp=datetime(2026, 1, 1),
            cash=cash,
            nav=cash + qty * price,
            positions=[pos],
            daily_pnl=0.0,
            peak_nav=cash + qty * price,
            drawdown=0.0,
            daily_drawdown=0.0,
        )

    def test_partial_exit_at_first_tier(self):
        """When exit_score exceeds first tier, sell 50%."""
        config = {
            "strategy": {
                "min_entry_score": 0.65,
                "min_exit_score": 0.35,
                "max_blocker_score": 0.30,
                "exit_tiers": [
                    {"threshold": 0.40, "sell_pct": 0.50},
                    {"threshold": 0.70, "sell_pct": 1.00},
                ],
            },
            "alpha": {},
        }
        logic = StrategyLogic("BTC/USDT", config)
        logic._state = StrategyState.HOLDING
        logic._entry_price = 95000.0
        logic._entry_time = datetime(2026, 1, 1, 0, 0)

        portfolio = self._holding_portfolio()

        # exit_score=0.45 > first tier threshold 0.40
        factors = _factor_snapshot(exit_score=0.45, blocker_score=0.0)
        intent = logic.on_factors(factors, portfolio, current_price=100000.0)

        assert intent is not None
        assert intent.direction == Side.SELL
        assert intent.quantity == pytest.approx(0.5, rel=0.01)  # 50% of 1.0

    def test_time_based_exit_after_max_hold(self):
        """When position held > max hours, trigger full exit."""
        config = {
            "strategy": {
                "min_entry_score": 0.65,
                "min_exit_score": 0.35,
                "max_blocker_score": 0.30,
                "bb_max_hold_hours": 24,
                "exit_tiers": [],
            },
            "alpha": {},
        }
        logic = StrategyLogic("BTC/USDT", config)
        logic._state = StrategyState.HOLDING
        logic._entry_price = 95000.0
        logic._entry_time = datetime(2026, 1, 1, 0, 0)  # 25h ago

        portfolio = self._holding_portfolio(qty=1.0, price=96000.0)

        # Low exit_score that wouldn't normally trigger, but timestamp is 25h later
        factors = _factor_snapshot(exit_score=0.10, blocker_score=0.10)
        factors.timestamp = datetime(2026, 1, 2, 1, 0)  # 25h after entry

        intent = logic.on_factors(factors, portfolio, current_price=96000.0)
        assert intent is not None
        assert intent.direction == Side.SELL
        assert intent.quantity == pytest.approx(1.0)

    def test_existing_blocker_exit_still_works(self):
        """Existing exit via blocker_score >= max_blocker_score should still function."""
        config = {
            "strategy": {
                "min_entry_score": 0.65,
                "min_exit_score": 0.35,
                "max_blocker_score": 0.30,
                "exit_tiers": [
                    {"threshold": 0.40, "sell_pct": 0.50},
                    {"threshold": 0.70, "sell_pct": 1.00},
                ],
            },
            "alpha": {},
        }
        logic = StrategyLogic("BTC/USDT", config)
        logic._state = StrategyState.HOLDING
        logic._entry_price = 95000.0
        logic._entry_time = datetime(2026, 1, 1, 0, 0)

        portfolio = self._holding_portfolio()

        # Low exit_score (below tiers) but high blocker_score → should trigger full exit
        factors = _factor_snapshot(exit_score=0.10, blocker_score=0.50)
        intent = logic.on_factors(factors, portfolio, current_price=100000.0)

        assert intent is not None
        assert intent.direction == Side.SELL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/test_strategy.py::TestGraduatedFactorExit -v`
Expected: FAIL

- [ ] **Step 3: Implement graduated factor exit and time-based exit in `StrategyLogic`**

In `strategy/logic.py`:

1. Add `_entry_time` and `_bb_max_hold_hours` to `__init__`:
```python
self._entry_time: Optional[datetime] = None
self._bb_max_hold_hours: float = strategy_cfg.get("bb_max_hold_hours", 0)
```

2. Set `_entry_time` in `on_fill()` when a BUY fills:
```python
# Inside on_fill(), after the existing self._record_entry_fill(order) line,
# when transitioning to HOLDING:
self._entry_time = order.created_at if hasattr(order, 'created_at') and order.created_at else datetime.utcnow()
```

3. In the HOLDING branch of `on_factors()` (around line 341), replace the existing exit logic with graduated factor exit support:

```python
if self._state == StrategyState.HOLDING:
    pos_qty = self._position_quantity(portfolio)
    if pos_qty <= 0:
        self.force_flat()
        return None

    # Time-based exit: if held longer than max hold hours
    if (
        self._bb_max_hold_hours > 0
        and self._entry_time is not None
        and factors.timestamp > self._entry_time + timedelta(hours=self._bb_max_hold_hours)
        and self._exit_tier_reached == 0  # no TP hit yet
    ):
        intent = self._build_exit_intent(
            factors, portfolio, pos_qty, current_price,
            reasoning=reasoning + [f"Max hold time {self._bb_max_hold_hours}h exceeded"],
        )
        self._state = StrategyState.EXIT_PENDING
        self._reset_exit_fill_tracking()
        return intent

    # Graduated factor exit: check exit_score tiers
    if self._exit_tiers:
        for i, tier in enumerate(self._exit_tiers):
            if i <= self._exit_tier_reached - 1:
                continue
            if effective_exit >= tier["threshold"]:
                sell_pct = tier["sell_pct"]
                sell_qty = pos_qty * sell_pct if sell_pct < 1.0 else pos_qty
                sell_qty = min(sell_qty, pos_qty)
                self._exit_tier_reached = i + 1

                intent = self._build_exit_intent(
                    factors, portfolio, sell_qty, current_price,
                    reasoning=reasoning + [f"Exit tier {i+1}: exit_score {effective_exit:.3f} >= {tier['threshold']}"],
                )
                if sell_pct >= 1.0 or sell_qty >= pos_qty - 1e-12:
                    self._state = StrategyState.EXIT_PENDING
                else:
                    self._state = StrategyState.EXIT_PENDING
                self._reset_exit_fill_tracking()
                logger.info(
                    "[%s] graduated factor exit tier %d: exit_score=%.3f, sell_pct=%.2f, qty=%.6f",
                    self._symbol, i + 1, effective_exit, sell_pct, sell_qty,
                )
                return intent

    # Fallback: original exit logic (blocker or exit score threshold)
    if effective_exit < self._min_exit_score and effective_blocker < self._max_blocker_score:
        return None

    # ... rest of existing exit logic stays the same
```

4. Add helper method `_build_exit_intent()` to avoid code duplication:

```python
def _build_exit_intent(
    self,
    factors: FactorSnapshot,
    portfolio: PortfolioSnapshot,
    qty: float,
    current_price: float,
    reasoning: list[str] | None = None,
) -> StrategyIntent:
    size_notional = qty * current_price
    return StrategyIntent(
        signal_time=factors.timestamp,
        symbol=self._symbol,
        direction=Side.SELL,
        thesis=self._compose_exit_thesis(factors.blocking_factors, reasoning or []),
        entry_type=OrderType.MARKET,
        entry_price=None,
        size_pct=size_notional / portfolio.nav if portfolio.nav > 0 else 0.0,
        size_notional=size_notional,
        quantity=qty,
        signal_horizon=self._format_horizon(self._exit_horizon_minutes),
        expected_move="Protect capital",
        stop_loss="Exit immediately",
        take_profit="N/A",
        invalidate_condition="Exit cancelled only if factors clear",
        urgency=UrgencyLevel.HIGH,
        confidence=min(1.0, max(0.0, factors.exit_score)),
        factor_names=factors.blocking_factors,
        reasoning=(reasoning or [])[:4],
    )
```

5. Reset `_entry_time` in `force_flat()`:
```python
self._entry_time = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/test_strategy.py::TestGraduatedFactorExit -v`
Expected: PASS

- [ ] **Step 5: Run all strategy tests to ensure no regressions**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/test_strategy.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/kevinwu/Coding/trading_competition
git add strategy/logic.py tests/test_strategy.py
git commit -m "feat: add graduated factor-based exits and time-based exit to StrategyLogic"
```

---

### Task 4: Add `swing_mean_reversion_v1` strategy profile to main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add the profile to `STRATEGY_PROFILES` dict**

In `main.py`, add a new entry to `STRATEGY_PROFILES` (after the existing profiles):

```python
"swing_mean_reversion_v1": {
    "symbols": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
        "XRP/USDT", "LINK/USDT", "ADA/USDT", "AVAX/USDT",
        "DOT/USDT", "DOGE/USDT", "LTC/USDT", "UNI/USDT",
    ],
    "alpha": {
        "engine": "rule_based",
        "resample_minutes": 5,
        "multi_timeframes": [15, 60],
    },
    "strategy": {
        "profile": "swing_mean_reversion_v1",
        "use_model_overlay": False,
        "position_size_pct": 0.05,
        "base_size_pct": 0.03,
        "max_size_pct": 0.12,
        "kelly_fraction": 0.35,
        "estimated_win_rate": 0.58,
        "estimated_payoff": 1.80,
        "confirmation_bars": 1,
        "min_entry_score": 0.65,
        "max_blocker_score": 0.30,
        "min_exit_score": 0.35,
        "min_supporting_factors": 2,
        "min_supporting_categories": 2,
        "require_trend_alignment": True,
        "urgent_entry_score": 0.90,
        "signal_horizon_minutes": 480,
        "exit_horizon_minutes": 60,
        "base_stop_loss_pct": 0.025,
        "take_profit_1_rr": 1.2,
        "take_profit_2_rr": 2.0,
        "neutral_entry_size_multiplier": 0.50,
        "risk_off_entry_size_multiplier": 0.00,
        "min_volatility_size_multiplier": 0.30,
        "max_active_positions": 3,
        "top_n_entries_per_cycle": 1,
        "min_volume_ratio": 1.20,
        "min_order_book_imbalance": 1.02,
        "max_funding_rate": 0.0005,
        "max_taker_ratio": 1.20,
        "max_open_interest_change": 0.03,
        "max_volatility": 0.025,
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "bb_rsi_oversold": 35,
        "bb_rsi_deeply_oversold": 25,
        "bb_max_hold_hours": 24,
        "bb_min_width": 0.02,
        "exit_tiers": [
            {"threshold": 0.40, "sell_pct": 0.50},
            {"threshold": 0.70, "sell_pct": 1.00},
        ],
        "factor_weights": {
            "bollinger_mean_reversion": 0.40,
            "trend_alignment": 0.25,
            "volume_confirmation": 0.12,
            "volatility_regime": 0.10,
            "market_regime": 0.08,
            "liquidity_balance": 0.05,
            "momentum_impulse": 0.00,
            "breakout_confirmation": 0.00,
            "perp_crowding": 0.00,
            "pullback_reentry": 0.00,
            "overextension_exit": 0.00,
        },
    },
    "regime": {
        "enabled": True,
        "benchmark_symbols": ["BTC/USDT", "ETH/USDT"],
        "risk_on_threshold": 0.25,
        "neutral_threshold": 0.08,
        "breadth_min_symbols": 6,
        "volatility_ceiling": 0.020,
    },
    "risk": {
        "max_portfolio_exposure": 0.50,
        "max_single_exposure": 0.12,
        "trailing_stop_pct": 0.035,
        "atr_stop_multiplier": 3.0,
        "daily_drawdown_limit": 0.04,
        "max_orders_per_minute": 6,
    },
    "trend": {
        "breakout_lookback": 24,
        "trend_slope_lookback": 24,
        "volume_zscore_window": 24,
        "vol_target_floor": 0.015,
    },
},
```

- [ ] **Step 2: Run full test suite to check no breakage**

Run: `cd /Users/kevinwu/Coding/trading_competition && .venv/bin/python3 -m pytest tests/ -v --timeout=30`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/kevinwu/Coding/trading_competition
git add main.py
git commit -m "feat: add swing_mean_reversion_v1 strategy profile"
```

---

### Task 5: Fix BacktestMonitor missing buy-candidate ranking

**Critical bug:** `BacktestMonitor.process_backtest_iteration()` submits every buy intent immediately without the cross-symbol ranking, position cap, or top-N entry gating that the live `StrategyMonitor._process_iteration()` uses. This means backtest results don't reflect live behavior.

**Files:**
- Modify: `/Users/kevinwu/Coding/trading_competition_backtest/tc_backtest/backtest_monitor.py`
- Test: `/Users/kevinwu/Coding/trading_competition_backtest/tests/test_backtest_monitor.py` (new)

- [ ] **Step 1: Write a test that verifies buy-candidate ranking is applied**

Create `/Users/kevinwu/Coding/trading_competition_backtest/tests/test_backtest_monitor.py`:

```python
"""Test that BacktestMonitor uses buy-candidate ranking like the live monitor."""
import pytest

def test_backtest_monitor_has_buy_candidate_ranking():
    """BacktestMonitor.process_backtest_iteration should collect buy intents
    and rank them through _rank_buy_candidates rather than submitting immediately."""
    import inspect
    from tc_backtest.backtest_monitor import BacktestMonitor
    source = inspect.getsource(BacktestMonitor.process_backtest_iteration)
    # The method must reference buy_candidates and ranking
    assert "buy_candidates" in source, "BacktestMonitor should collect buy candidates"
    assert "_rank_buy_candidates" in source, "BacktestMonitor should rank buy candidates"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kevinwu/Coding/trading_competition_backtest && ../trading_competition/.venv/bin/python3 -m pytest tests/test_backtest_monitor.py -v`
Expected: FAIL — `buy_candidates` not in source

- [ ] **Step 3: Refactor `process_backtest_iteration` to use buy-candidate ranking**

In `tc_backtest/backtest_monitor.py`, replace lines 172-195 (the intent submission block) with buy-candidate collection and ranking matching the live monitor's pattern:

Replace:
```python
        if intent is not None:
            instruction = strategy.build_instruction(intent, current_price=candles[-1].close)
            order = instruction.to_order()
            order.created_at = current_time
            if hasattr(self._risk_shield, "validate_at"):
                ...
            else:
                ...
            if validated is not None:
                ...
            else:
                strategy.on_cancel(order)
```

With:
```python
            if intent is not None:
                if intent.direction == Side.BUY:
                    buy_candidates.append({
                        "symbol": symbol,
                        "strategy": strategy,
                        "intent": intent,
                        "current_price": candles[-1].close,
                        "factor_snapshot": factor_snapshot,
                        "current_time": current_time,
                    })
                    continue

                instruction = strategy.build_instruction(intent, current_price=candles[-1].close)
                order = instruction.to_order()
                order.created_at = current_time
                if hasattr(self._risk_shield, "validate_at"):
                    validated = self._risk_shield.validate_at(
                        order, self._tracker, current_time=current_time,
                        market_price=candles[-1].close,
                    )
                else:
                    validated = self._risk_shield.validate(
                        order, self._tracker, market_price=candles[-1].close,
                    )
                if validated is not None:
                    if hasattr(self._order_manager, "submit_at"):
                        await self._order_manager.submit_at(validated, current_time)
                    else:
                        await self._order_manager.submit(validated)
                else:
                    strategy.on_cancel(order)
```

Also add `Side` to the module-level import from `core.models` (line 10 of backtest_monitor.py):
```python
from core.models import OHLCV, Order, OrderStatus, Side, StrategyState
```

And after the symbol loop (before stop_orders), add:
```python
        # Buy candidate ranking (mirrors live StrategyMonitor behavior)
        ranked = self._rank_buy_candidates(buy_candidates)
        selected = ranked[:self._top_n_entries_per_cycle]
        for skipped in ranked[self._top_n_entries_per_cycle:]:
            self._cancel_buy_candidate(skipped, "not in top entry cohort")

        for idx, candidate in enumerate(selected):
            unsubmitted = {c["symbol"] for c in selected[idx:]}
            if self._active_symbol_count(exclude_symbols=unsubmitted) >= self._max_active_positions:
                self._cancel_buy_candidate(candidate, "max active position cap")
                continue
            strategy = candidate["strategy"]
            intent = candidate["intent"]
            current_price = candidate["current_price"]
            ct = candidate["current_time"]
            instruction = strategy.build_instruction(intent, current_price=current_price)
            order = instruction.to_order()
            order.created_at = ct
            if hasattr(self._risk_shield, "validate_at"):
                validated = self._risk_shield.validate_at(
                    order, self._tracker, current_time=ct, market_price=current_price,
                )
            else:
                validated = self._risk_shield.validate(
                    order, self._tracker, market_price=current_price,
                )
            if validated is not None:
                if hasattr(self._order_manager, "submit_at"):
                    await self._order_manager.submit_at(validated, ct)
                else:
                    await self._order_manager.submit(validated)
            else:
                strategy.on_cancel(order)
```

Also add `buy_candidates: list = []` at the top of `process_backtest_iteration`, and import `Side` at the module level.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kevinwu/Coding/trading_competition_backtest && ../trading_competition/.venv/bin/python3 -m pytest tests/test_backtest_monitor.py -v`
Expected: PASS

- [ ] **Step 5: Run existing backtest tests**

Run: `cd /Users/kevinwu/Coding/trading_competition_backtest && ../trading_competition/.venv/bin/python3 -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/kevinwu/Coding/trading_competition_backtest
git add tc_backtest/backtest_monitor.py tests/test_backtest_monitor.py
git commit -m "fix: add buy-candidate ranking to BacktestMonitor (matches live behavior)"
```

---

### Task 6: Run backtests and compare strategies

**Files:** No code changes — execution only.

- [ ] **Step 1: Run single backtest with `swing_mean_reversion_v1` on latest 10-day window**

```bash
cd /Users/kevinwu/Coding/trading_competition_backtest
../trading_competition/.venv/bin/python3 run_backtest.py \
  --profile swing_mean_reversion_v1 \
  --start 2026-03-07 \
  --end 2026-03-17 \
  --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT LINK/USDT ADA/USDT AVAX/USDT DOT/USDT DOGE/USDT LTC/USDT UNI/USDT \
  --name swing_mr_latest \
  --verbose
```

Check output for: return %, win rate, trade count, max drawdown.

- [ ] **Step 2: Run 90-day rolling window sweep comparing all profiles**

```bash
cd /Users/kevinwu/Coding/trading_competition_backtest
../trading_competition/.venv/bin/python3 run_window_sweep.py \
  --lookback-days 90 \
  --window-days 10 \
  --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT LINK/USDT ADA/USDT AVAX/USDT DOT/USDT DOGE/USDT LTC/USDT UNI/USDT \
  --name swing_comparison \
  --verbose
```

Note: The window sweep runs conservative, normal, trend_pullback strategies by default. To add swing_mean_reversion, check if `run_window_sweep.py` or `window_sweep.py` needs the new profile added to its strategy specs.

- [ ] **Step 3: Analyze results and produce comparison table**

Read `artifacts/window_sweeps/swing_comparison_*/strategy_summary.csv` and compare:
- Mean return
- Win rate
- Profit factor
- Max drawdown
- Trade count

- [ ] **Step 4: Commit backtest artifacts and push**

```bash
cd /Users/kevinwu/Coding/trading_competition
git push origin codex/core-satellite-competition-20260322
cd /Users/kevinwu/Coding/trading_competition_backtest
git add -A && git commit -m "feat: swing mean-reversion backtest results" && git push
```
