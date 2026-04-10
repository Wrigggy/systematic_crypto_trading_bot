# Systematic Crypto Trading Bot — Open-Source Redesign

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `trading_competition` (fork) into `systematic_crypto_trading_bot` — a portfolio-showcase crypto trading system with a pluggable expression-tree alpha contract.

**Architecture:** Fork the original repo. Add an `alpha/` module that loads expression-tree alphas from JSON, evaluates them via an AST evaluator ported from AlphaGen, and normalizes to trading signals. Extract Roostoo and model inference into `plugins/`. Delete competition-specific code. Rewrite README as architecture-first showcase.

**Tech Stack:** Python 3.10+, Pydantic, numpy, asyncio, websockets, ccxt, PyTorch/ONNX (plugin only)

**Spec:** `docs/superpowers/specs/2026-04-10-trading-system-open-source-redesign.md`

---

## File Map

### New files to create

| File | Responsibility |
|---|---|
| `alpha/__init__.py` | Package init |
| `alpha/contract.py` | `AlphaSpec` Pydantic model, JSON loader, schema validation |
| `alpha/expression.py` | Expression AST: parser, evaluator, operators (ported from AlphaGen, numpy-based) |
| `alpha/normalizer.py` | `rolling_zscore` and `cross_sectional` normalization |
| `alpha/registry.py` | Load all alpha JSONs at startup, validate features, produce `FactorObservation` per tick |
| `alphas/schema.json` | JSON Schema for alpha contract |
| `alphas/builtin/momentum_impulse.json` | Converted from current hardcoded factor |
| `alphas/builtin/trend_alignment.json` | Converted from current hardcoded factor |
| `alphas/builtin/volume_confirmation.json` | Converted from current hardcoded factor |
| `alphas/builtin/mean_reversion.json` | Converted from current hardcoded factor |
| `alphas/examples/model_checkpoint_example.json` | Example for model plugin |
| `strategy/sizing.py` | Position sizing extracted from `logic.py` |
| `strategy/optimizer.py` | Portfolio allocation (score_tilted, equal_weight) |
| `plugins/__init__.py` | Package init |
| `plugins/roostoo/__init__.py` | Package init |
| `plugins/roostoo/executor.py` | Moved from `execution/roostoo_executor.py` |
| `plugins/roostoo/auth.py` | Moved from `data/roostoo_auth.py` |
| `plugins/roostoo/README.md` | Plugin docs |
| `plugins/model_inference/__init__.py` | Package init |
| `plugins/model_inference/evaluator.py` | Moved from `models/inference.py` (model parts only) |
| `plugins/model_inference/model_wrapper.py` | Moved from `models/model_wrapper.py` |
| `plugins/model_inference/README.md` | Plugin docs |
| `tests/test_alpha_contract.py` | Alpha contract loading and validation tests |
| `tests/test_expression.py` | Expression parser and evaluator tests |
| `tests/test_normalizer.py` | Normalization tests |
| `tests/test_registry.py` | Registry integration tests |
| `tests/test_optimizer.py` | Portfolio optimizer tests |
| `config/examples/paper_btc_only.yaml` | Minimal config example |
| `config/examples/multi_symbol_live.yaml` | Full config example |

### Files to modify

| File | Change |
|---|---|
| `core/models.py` | Add `FactorObservation`, `FactorSnapshot`, `StrategyIntent`, `TradeInstruction` Pydantic models |
| `strategy/monitor.py` | Replace `AlphaEngine` dependency with `AlphaRegistry`, wire optimizer |
| `strategy/logic.py` | Consume `FactorSnapshot` instead of `Signal`, extract sizing to `sizing.py` |
| `main.py` | Replace `AlphaEngine` init with `AlphaRegistry`, add `--alphas` flag, plugin loading |
| `config/default.yaml` | Simplified config per spec |
| `features/extractor.py` | Add `available_features()` class method for registry validation |

### Files to delete

| File | Reason |
|---|---|
| `models/inference.py` | Replaced by `alpha/registry.py` (rule-based) + `plugins/model_inference/` |
| `models/lstm_model.py` | Training belongs in research repo |
| `models/transformer_model.py` | Training belongs in research repo |
| `models/train.py` | Training belongs in research repo |
| `models/icir_tracker.py` | Future work |
| `models/__init__.py` | Package removed |
| `execution/roostoo_executor.py` | Moved to plugin |
| `data/roostoo_auth.py` | Moved to plugin |
| `strategy/trade_tracker.py` | Over-engineered for MVP |
| `scripts/upload_model_to_hf.py` | No models in core |

---

## Task 1: Fork & Initial Setup

**Files:**
- Create: GitHub fork
- Modify: `pyproject.toml`, `.git/config`

- [ ] **Step 1: Fork the repo on GitHub**

```bash
gh repo fork qunzhongwang/trading_competition --clone=false --fork-name systematic_crypto_trading_bot
gh repo clone <your-username>/systematic_crypto_trading_bot
cd systematic_crypto_trading_bot
```

- [ ] **Step 2: Update pyproject.toml metadata**

Change the `[project]` section:

```toml
[project]
name = "systematic-crypto-trading-bot"
description = "Factor-first crypto trading system with pluggable expression-tree alpha contract"
```

- [ ] **Step 3: Verify existing tests pass**

Run: `uv sync && uv run pytest -q`
Expected: All existing tests pass (this is our baseline)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: rename project to systematic_crypto_trading_bot"
```

---

## Task 2: Add Domain Models (FactorObservation, FactorSnapshot, StrategyIntent, TradeInstruction)

**Files:**
- Modify: `core/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for new models**

Add to `tests/test_models.py`:

```python
from core.models import FactorObservation, FactorSnapshot, StrategyIntent, TradeInstruction
from datetime import datetime


class TestFactorObservation:
    def test_create_bullish(self):
        obs = FactorObservation(
            name="momentum_impulse",
            symbol="BTC/USDT",
            bias="BULLISH",
            strength=0.8,
            timestamp=datetime(2026, 1, 1),
        )
        assert obs.bias == "BULLISH"
        assert 0.0 <= obs.strength <= 1.0

    def test_strength_clamped(self):
        obs = FactorObservation(
            name="test",
            symbol="BTC/USDT",
            bias="NEUTRAL",
            strength=1.5,  # over 1.0
            timestamp=datetime(2026, 1, 1),
        )
        assert obs.strength == 1.0


class TestFactorSnapshot:
    def test_aggregate_observations(self):
        obs1 = FactorObservation(
            name="alpha1", symbol="BTC/USDT", bias="BULLISH",
            strength=0.8, timestamp=datetime(2026, 1, 1),
        )
        obs2 = FactorObservation(
            name="alpha2", symbol="BTC/USDT", bias="BEARISH",
            strength=0.3, timestamp=datetime(2026, 1, 1),
        )
        snap = FactorSnapshot(
            symbol="BTC/USDT",
            timestamp=datetime(2026, 1, 1),
            observations=[obs1, obs2],
            entry_score=0.5,
            exit_score=-0.1,
            confidence=0.7,
        )
        assert len(snap.observations) == 2
        assert snap.entry_score == 0.5


class TestStrategyIntent:
    def test_create_long_intent(self):
        intent = StrategyIntent(
            symbol="BTC/USDT",
            direction="LONG",
            target_weight=0.35,
            thesis="Momentum + volume aligned",
            timestamp=datetime(2026, 1, 1),
        )
        assert intent.direction == "LONG"


class TestTradeInstruction:
    def test_create_instruction(self):
        instr = TradeInstruction(
            symbol="BTC/USDT",
            side="BUY",
            quantity=0.12,
            order_type="LIMIT",
            limit_price=65000.0,
            timestamp=datetime(2026, 1, 1),
        )
        assert instr.quantity == 0.12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py::TestFactorObservation -v`
Expected: FAIL — `ImportError: cannot import name 'FactorObservation'`

- [ ] **Step 3: Implement new models**

Add to `core/models.py` after the `Signal` class:

```python
class Bias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Direction(str, Enum):
    LONG = "LONG"
    FLAT = "FLAT"
    EXIT = "EXIT"


class FactorObservation(BaseModel):
    """Output of a single alpha evaluation for one symbol."""
    name: str
    symbol: str
    bias: Bias
    strength: float  # [0.0, 1.0]
    timestamp: datetime
    thesis: str = ""

    def model_post_init(self, __context) -> None:
        object.__setattr__(self, "strength", max(0.0, min(1.0, self.strength)))


class FactorSnapshot(BaseModel):
    """Aggregated signal across all alphas for one symbol."""
    symbol: str
    timestamp: datetime
    observations: List[FactorObservation] = Field(default_factory=list)
    entry_score: float = 0.0  # weighted composite, positive = bullish
    exit_score: float = 0.0   # negative = should exit
    confidence: float = 0.0   # [0.0, 1.0]


class StrategyIntent(BaseModel):
    """What the strategy wants to do, before sizing/risk."""
    symbol: str
    direction: Direction
    target_weight: float = 0.0  # desired portfolio weight [0.0, 1.0]
    thesis: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradeInstruction(BaseModel):
    """Sized, concrete order to submit. Output of optimizer."""
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All PASS including existing model tests

- [ ] **Step 5: Commit**

```bash
git add core/models.py tests/test_models.py
git commit -m "feat: add FactorObservation, FactorSnapshot, StrategyIntent, TradeInstruction domain models"
```

---

## Task 3: Expression Evaluator (numpy-based, ported from AlphaGen)

**Files:**
- Create: `alpha/__init__.py`, `alpha/expression.py`
- Test: `tests/test_expression.py`

The original AlphaGen evaluator uses PyTorch tensors over `(days, stocks)`. We port it to **numpy arrays** operating on `(time,)` for a single symbol, since the trading system processes per-symbol candle streams.

- [ ] **Step 1: Write failing tests**

Create `tests/test_expression.py`:

```python
import numpy as np
import pytest
from alpha.expression import ExpressionParser, ExpressionEvaluator


class TestExpressionParser:
    def setup_method(self):
        self.parser = ExpressionParser()

    def test_parse_feature(self):
        expr = self.parser.parse("$close")
        assert expr.kind == "feature"
        assert expr.value == "close"

    def test_parse_constant(self):
        expr = self.parser.parse("1.5")
        assert expr.kind == "constant"
        assert expr.value == 1.5

    def test_parse_unary(self):
        expr = self.parser.parse("Abs($close)")
        assert expr.kind == "operator"
        assert expr.name == "Abs"
        assert len(expr.children) == 1

    def test_parse_binary(self):
        expr = self.parser.parse("Mul($close, $volume)")
        assert expr.kind == "operator"
        assert expr.name == "Mul"
        assert len(expr.children) == 2

    def test_parse_rolling(self):
        expr = self.parser.parse("Mean($close, 20)")
        assert expr.kind == "operator"
        assert expr.name == "Mean"
        assert expr.window == 20

    def test_parse_nested(self):
        expr = self.parser.parse("Mul(Delta($close, 5), Div($volume, Mean($volume, 20)))")
        assert expr.kind == "operator"
        assert expr.name == "Mul"

    def test_parse_invalid_feature(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            self.parser.parse("$nonexistent")

    def test_parse_invalid_operator(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            self.parser.parse("FakeOp($close)")

    def test_referenced_features(self):
        expr = self.parser.parse("Mul(Delta($close, 5), $volume)")
        assert expr.referenced_features() == {"close", "volume"}


class TestExpressionEvaluator:
    def setup_method(self):
        self.parser = ExpressionParser()
        # 100 bars of synthetic OHLCV data
        np.random.seed(42)
        n = 100
        base = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        self.data = {
            "open": base + np.random.randn(n) * 0.1,
            "high": base + abs(np.random.randn(n)) * 0.5,
            "low": base - abs(np.random.randn(n)) * 0.5,
            "close": base.copy(),
            "volume": np.random.exponential(1000, n),
            "vwap": base + np.random.randn(n) * 0.05,
        }
        self.evaluator = ExpressionEvaluator()

    def test_evaluate_feature(self):
        expr = self.parser.parse("$close")
        result = self.evaluator.evaluate(expr, self.data)
        np.testing.assert_array_equal(result, self.data["close"])

    def test_evaluate_constant(self):
        expr = self.parser.parse("1.5")
        result = self.evaluator.evaluate(expr, self.data)
        assert np.all(result == 1.5)
        assert len(result) == len(self.data["close"])

    def test_evaluate_abs(self):
        expr = self.parser.parse("Abs(Sub($close, $open))")
        result = self.evaluator.evaluate(expr, self.data)
        expected = np.abs(self.data["close"] - self.data["open"])
        np.testing.assert_allclose(result, expected)

    def test_evaluate_mean_rolling(self):
        expr = self.parser.parse("Mean($close, 20)")
        result = self.evaluator.evaluate(expr, self.data)
        # First 19 values should be NaN (insufficient window)
        assert np.all(np.isnan(result[:19]))
        # Value at index 19 should be mean of first 20 closes
        expected = np.mean(self.data["close"][:20])
        np.testing.assert_allclose(result[19], expected, rtol=1e-10)

    def test_evaluate_delta(self):
        expr = self.parser.parse("Delta($close, 5)")
        result = self.evaluator.evaluate(expr, self.data)
        assert np.isnan(result[4])  # not enough history
        expected = self.data["close"][5] - self.data["close"][0]
        np.testing.assert_allclose(result[5], expected)

    def test_evaluate_nested(self):
        """Mul(Delta($close, 5), Div($volume, Mean($volume, 20)))"""
        expr = self.parser.parse("Mul(Delta($close, 5), Div($volume, Mean($volume, 20)))")
        result = self.evaluator.evaluate(expr, self.data)
        assert len(result) == 100
        # First 19 bars are NaN due to Mean window
        assert np.all(np.isnan(result[:19]))
        # After warmup, values are finite
        valid = result[19:]
        assert np.all(np.isfinite(valid))

    def test_evaluate_csrank(self):
        """CSRank should return the value unchanged for single-symbol."""
        expr = self.parser.parse("CSRank($close)")
        result = self.evaluator.evaluate(expr, self.data)
        # For single symbol, CSRank returns 0.5 (middle of [0,1])
        np.testing.assert_allclose(result, np.full(100, 0.5))

    def test_nan_propagation(self):
        """NaN in input should propagate through operators."""
        self.data["close"][50] = np.nan
        expr = self.parser.parse("Abs($close)")
        result = self.evaluator.evaluate(expr, self.data)
        assert np.isnan(result[50])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_expression.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'alpha'`

- [ ] **Step 3: Create alpha package and expression module**

Create `alpha/__init__.py`:

```python
```

Create `alpha/expression.py`:

```python
"""Expression-tree AST: parser and numpy-based evaluator.

Ported from AlphaGen (https://github.com/RL-MLDM/alphagen).
Adapted from PyTorch tensors (days, stocks) to numpy arrays (time,)
for per-symbol real-time evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

# Valid OHLCV features the expression can reference
VALID_FEATURES = {"open", "close", "high", "low", "volume", "vwap"}

# Operator registry: name → (category, n_args)
# Categories: unary (1 expr), binary (2 expr), rolling (1 expr + window),
#             pair_rolling (2 expr + window)
_OPERATORS: Dict[str, tuple] = {
    # Unary
    "Abs": ("unary", 1),
    "Sign": ("unary", 1),
    "Log": ("unary", 1),
    "CSRank": ("unary", 1),
    # Binary
    "Add": ("binary", 2),
    "Sub": ("binary", 2),
    "Mul": ("binary", 2),
    "Div": ("binary", 2),
    "Pow": ("binary", 2),
    "Greater": ("binary", 2),
    "Less": ("binary", 2),
    # Rolling (expr + window)
    "Ref": ("rolling", 1),
    "Mean": ("rolling", 1),
    "Sum": ("rolling", 1),
    "Std": ("rolling", 1),
    "Var": ("rolling", 1),
    "Skew": ("rolling", 1),
    "Kurt": ("rolling", 1),
    "Max": ("rolling", 1),
    "Min": ("rolling", 1),
    "Med": ("rolling", 1),
    "Mad": ("rolling", 1),
    "Rank": ("rolling", 1),
    "Delta": ("rolling", 1),
    "WMA": ("rolling", 1),
    "EMA": ("rolling", 1),
    # Pair rolling (2 expr + window)
    "Cov": ("pair_rolling", 2),
    "Corr": ("pair_rolling", 2),
}


@dataclass
class ExprNode:
    """AST node for an expression tree."""
    kind: str  # "feature", "constant", "operator"
    value: object = None  # feature name (str) or constant value (float)
    name: str = ""  # operator name
    children: List["ExprNode"] = field(default_factory=list)
    window: Optional[int] = None  # for rolling operators

    def referenced_features(self) -> Set[str]:
        """Return set of feature names used by this expression."""
        if self.kind == "feature":
            return {self.value}
        result: Set[str] = set()
        for child in self.children:
            result |= child.referenced_features()
        return result


_TOKEN_PATTERN = re.compile(r"([+-]?\d+\.?\d*|\$\w+|\w+|[(),])")


class ExpressionParser:
    """Parse expression strings like 'Mul(Delta($close, 5), $volume)' into ExprNode AST."""

    def parse(self, expr_str: str) -> ExprNode:
        tokens = [t for t in _TOKEN_PATTERN.findall(expr_str) if t.strip()]
        node, pos = self._parse_tokens(tokens, 0)
        if pos != len(tokens):
            raise ValueError(f"Unexpected tokens after position {pos}: {tokens[pos:]}")
        return node

    def _parse_tokens(self, tokens: List[str], pos: int) -> tuple:
        """Recursive descent parser. Returns (ExprNode, next_pos)."""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[pos]

        # Feature: $close, $volume, etc.
        if token.startswith("$"):
            feat_name = token[1:].lower()
            if feat_name not in VALID_FEATURES:
                raise ValueError(f"Unknown feature: {token}. Valid: {sorted(VALID_FEATURES)}")
            return ExprNode(kind="feature", value=feat_name), pos + 1

        # Numeric constant
        try:
            value = float(token)
            return ExprNode(kind="constant", value=value), pos + 1
        except ValueError:
            pass

        # Operator: Name(args...)
        if token in _OPERATORS:
            category, n_expr_args = _OPERATORS[token]
            if pos + 1 >= len(tokens) or tokens[pos + 1] != "(":
                raise ValueError(f"Operator {token} must be followed by '('")
            pos += 2  # skip operator name and '('

            children = []
            window = None

            for i in range(n_expr_args):
                if i > 0:
                    if pos >= len(tokens) or tokens[pos] != ",":
                        raise ValueError(f"Expected ',' after arg {i} in {token}")
                    pos += 1  # skip comma
                child, pos = self._parse_tokens(tokens, pos)
                children.append(child)

            # Rolling and pair_rolling operators have a window arg
            if category in ("rolling", "pair_rolling"):
                if pos >= len(tokens) or tokens[pos] != ",":
                    raise ValueError(f"Rolling operator {token} requires a window argument")
                pos += 1  # skip comma
                try:
                    window = int(float(tokens[pos]))
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid window for {token}: {tokens[pos] if pos < len(tokens) else 'EOF'}")
                pos += 1

            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError(f"Expected ')' to close {token}")
            pos += 1  # skip ')'

            return ExprNode(kind="operator", name=token, children=children, window=window), pos

        raise ValueError(f"Unknown operator: {token}. Valid: {sorted(_OPERATORS.keys())}")


class ExpressionEvaluator:
    """Evaluate an ExprNode AST over numpy arrays.

    Input: dict mapping feature names to 1D numpy arrays of equal length.
    Output: 1D numpy array of the same length. NaN where insufficient history.
    """

    def evaluate(self, node: ExprNode, data: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(data.values())))

        if node.kind == "feature":
            return data[node.value].astype(np.float64).copy()

        if node.kind == "constant":
            return np.full(n, node.value, dtype=np.float64)

        if node.kind == "operator":
            return self._eval_operator(node, data, n)

        raise ValueError(f"Unknown node kind: {node.kind}")

    def _eval_operator(self, node: ExprNode, data: Dict[str, np.ndarray], n: int) -> np.ndarray:
        name = node.name
        children_vals = [self.evaluate(c, data) for c in node.children]
        w = node.window

        # --- Unary ---
        if name == "Abs":
            return np.abs(children_vals[0])
        if name == "Sign":
            return np.sign(children_vals[0])
        if name == "Log":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(np.abs(children_vals[0]))
        if name == "CSRank":
            # Cross-sectional rank: for single symbol, always 0.5
            result = np.full(n, 0.5, dtype=np.float64)
            result[np.isnan(children_vals[0])] = np.nan
            return result

        # --- Binary ---
        if name == "Add":
            return children_vals[0] + children_vals[1]
        if name == "Sub":
            return children_vals[0] - children_vals[1]
        if name == "Mul":
            return children_vals[0] * children_vals[1]
        if name == "Div":
            with np.errstate(divide="ignore", invalid="ignore"):
                result = children_vals[0] / children_vals[1]
                result[~np.isfinite(result)] = np.nan
                return result
        if name == "Pow":
            with np.errstate(invalid="ignore"):
                return np.power(children_vals[0], children_vals[1])
        if name == "Greater":
            return (children_vals[0] > children_vals[1]).astype(np.float64)
        if name == "Less":
            return (children_vals[0] < children_vals[1]).astype(np.float64)

        # --- Rolling ---
        arr = children_vals[0]
        if name == "Ref":
            result = np.full(n, np.nan, dtype=np.float64)
            if w >= 0 and w < n:
                result[w:] = arr[:n - w]
            return result
        if name == "Delta":
            result = np.full(n, np.nan, dtype=np.float64)
            if w < n:
                result[w:] = arr[w:] - arr[:n - w]
            return result

        # Generic rolling window operations
        if name in ("Mean", "Sum", "Std", "Var", "Skew", "Kurt", "Max", "Min", "Med", "Mad", "Rank", "WMA", "EMA"):
            return self._rolling(name, arr, w, n)

        # --- Pair rolling ---
        if name in ("Cov", "Corr"):
            return self._pair_rolling(name, children_vals[0], children_vals[1], w, n)

        raise ValueError(f"Unimplemented operator: {name}")

    def _rolling(self, op: str, arr: np.ndarray, w: int, n: int) -> np.ndarray:
        result = np.full(n, np.nan, dtype=np.float64)

        if op == "EMA":
            alpha = 2.0 / (w + 1)
            result[0] = arr[0]
            for i in range(1, n):
                if np.isnan(arr[i]):
                    result[i] = result[i - 1]
                elif np.isnan(result[i - 1]):
                    result[i] = arr[i]
                else:
                    result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
            # Mark first w-1 as NaN (warmup)
            result[:w - 1] = np.nan
            return result

        if op == "WMA":
            weights = np.arange(1, w + 1, dtype=np.float64)
            weight_sum = weights.sum()
            for i in range(w - 1, n):
                window = arr[i - w + 1:i + 1]
                if np.any(np.isnan(window)):
                    continue
                result[i] = np.dot(window, weights) / weight_sum
            return result

        for i in range(w - 1, n):
            window = arr[i - w + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            if op == "Mean":
                result[i] = np.mean(window)
            elif op == "Sum":
                result[i] = np.sum(window)
            elif op == "Std":
                result[i] = np.std(window, ddof=1) if len(window) > 1 else 0.0
            elif op == "Var":
                result[i] = np.var(window, ddof=1) if len(window) > 1 else 0.0
            elif op == "Skew":
                from scipy.stats import skew
                result[i] = skew(window, bias=False)
            elif op == "Kurt":
                from scipy.stats import kurtosis
                result[i] = kurtosis(window, bias=False)
            elif op == "Max":
                result[i] = np.max(window)
            elif op == "Min":
                result[i] = np.min(window)
            elif op == "Med":
                result[i] = np.median(window)
            elif op == "Mad":
                result[i] = np.mean(np.abs(window - np.mean(window)))
            elif op == "Rank":
                from scipy.stats import rankdata
                ranks = rankdata(window)
                result[i] = ranks[-1] / len(window)
        return result

    def _pair_rolling(self, op: str, a: np.ndarray, b: np.ndarray, w: int, n: int) -> np.ndarray:
        result = np.full(n, np.nan, dtype=np.float64)
        for i in range(w - 1, n):
            wa = a[i - w + 1:i + 1]
            wb = b[i - w + 1:i + 1]
            if np.any(np.isnan(wa)) or np.any(np.isnan(wb)):
                continue
            if op == "Cov":
                result[i] = np.cov(wa, wb, ddof=1)[0, 1]
            elif op == "Corr":
                c = np.corrcoef(wa, wb)[0, 1]
                result[i] = c if np.isfinite(c) else np.nan
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_expression.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/__init__.py alpha/expression.py tests/test_expression.py
git commit -m "feat: add expression-tree parser and numpy evaluator (ported from AlphaGen)"
```

---

## Task 4: Alpha Contract (JSON Schema + Loader)

**Files:**
- Create: `alpha/contract.py`, `alphas/schema.json`
- Test: `tests/test_alpha_contract.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_alpha_contract.py`:

```python
import json
import pytest
from pathlib import Path
from alpha.contract import AlphaSpec, load_alpha, validate_alpha_features


class TestAlphaSpec:
    def test_load_valid_expression_tree(self, tmp_path):
        alpha_json = {
            "alpha_id": "test_momentum",
            "version": "1.0.0",
            "type": "expression_tree",
            "description": "Test alpha",
            "expression": "Delta($close, 5)",
            "normalization": {"method": "rolling_zscore", "lookback": 20},
            "weight_hint": 0.5,
            "horizon": "intraday",
            "meta": {"author": "test"},
        }
        f = tmp_path / "test.json"
        f.write_text(json.dumps(alpha_json))

        spec = load_alpha(f)
        assert spec.alpha_id == "test_momentum"
        assert spec.type == "expression_tree"
        assert spec.expression == "Delta($close, 5)"
        assert spec.normalization.method == "rolling_zscore"
        assert spec.weight_hint == 0.5

    def test_missing_expression_fails(self, tmp_path):
        alpha_json = {
            "alpha_id": "bad",
            "version": "1.0.0",
            "type": "expression_tree",
            "normalization": {"method": "rolling_zscore", "lookback": 20},
        }
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(alpha_json))

        with pytest.raises(Exception):  # Pydantic ValidationError
            load_alpha(f)

    def test_invalid_normalization_method_fails(self, tmp_path):
        alpha_json = {
            "alpha_id": "bad",
            "version": "1.0.0",
            "type": "expression_tree",
            "expression": "$close",
            "normalization": {"method": "invalid_method", "lookback": 20},
        }
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(alpha_json))

        with pytest.raises(Exception):
            load_alpha(f)

    def test_model_type_accepted(self, tmp_path):
        alpha_json = {
            "alpha_id": "lstm_v1",
            "version": "1.0.0",
            "type": "model",
            "description": "LSTM alpha",
            "compute": {
                "checkpoint": "artifacts/model.onnx",
                "input_features": ["rsi", "momentum"],
                "sequence_length": 240,
            },
            "normalization": {"method": "rolling_zscore", "lookback": 20},
            "weight_hint": 0.2,
            "meta": {"author": "test"},
        }
        f = tmp_path / "model.json"
        f.write_text(json.dumps(alpha_json))

        spec = load_alpha(f)
        assert spec.type == "model"
        assert spec.compute.checkpoint == "artifacts/model.onnx"


class TestFeatureValidation:
    def test_valid_features_pass(self):
        available = {"close", "volume", "open", "high", "low", "vwap"}
        errors = validate_alpha_features("Delta($close, 5)", available)
        assert errors == []

    def test_missing_feature_detected(self):
        available = {"close"}
        errors = validate_alpha_features("Mul($close, $volume)", available)
        assert len(errors) == 1
        assert "volume" in errors[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alpha_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'alpha.contract'`

- [ ] **Step 3: Implement contract module**

Create `alpha/contract.py`:

```python
"""Alpha contract: Pydantic models for the JSON alpha specification."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from alpha.expression import ExpressionParser, VALID_FEATURES


class AlphaType(str, Enum):
    EXPRESSION_TREE = "expression_tree"
    MODEL = "model"


class NormalizationMethod(str, Enum):
    ROLLING_ZSCORE = "rolling_zscore"
    CROSS_SECTIONAL = "cross_sectional"


class NormalizationConfig(BaseModel):
    method: NormalizationMethod = NormalizationMethod.ROLLING_ZSCORE
    lookback: int = 20


class ValidationMetrics(BaseModel):
    ic: Optional[float] = None
    icir: Optional[float] = None
    decay_halflife: Optional[float] = None
    backtest_sharpe: Optional[float] = None
    validated_on: Optional[str] = None
    judge_score: Optional[float] = None
    judge_narrative: Optional[str] = None


class ModelComputeConfig(BaseModel):
    checkpoint: str
    input_features: List[str] = Field(default_factory=list)
    sequence_length: int = 240


class AlphaMeta(BaseModel):
    author: str = "unknown"
    source_repo: Optional[str] = None


class AlphaSpec(BaseModel):
    """Specification for a single alpha signal."""
    alpha_id: str
    version: str = "1.0.0"
    type: AlphaType
    description: str = ""

    # For expression_tree type
    expression: Optional[str] = None

    # For model type
    compute: Optional[ModelComputeConfig] = None

    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    validation: Optional[ValidationMetrics] = None
    weight_hint: float = 1.0
    horizon: str = "intraday"
    meta: AlphaMeta = Field(default_factory=AlphaMeta)

    @model_validator(mode="after")
    def check_type_fields(self):
        if self.type == AlphaType.EXPRESSION_TREE and not self.expression:
            raise ValueError("expression_tree type requires 'expression' field")
        if self.type == AlphaType.MODEL and not self.compute:
            raise ValueError("model type requires 'compute' field")
        return self


def load_alpha(path: Path) -> AlphaSpec:
    """Load and validate an alpha spec from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return AlphaSpec(**data)


def validate_alpha_features(expression: str, available_features: set) -> List[str]:
    """Check that all features referenced in an expression are available.

    Returns list of error messages (empty = valid).
    """
    parser = ExpressionParser()
    try:
        node = parser.parse(expression)
    except ValueError as e:
        return [f"Parse error: {e}"]

    referenced = node.referenced_features()
    missing = referenced - available_features
    return [f"Missing feature: {feat}" for feat in sorted(missing)]
```

Create `alphas/schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AlphaSpec",
  "description": "Specification for a pluggable alpha signal",
  "type": "object",
  "required": ["alpha_id", "type"],
  "properties": {
    "alpha_id": {"type": "string"},
    "version": {"type": "string", "default": "1.0.0"},
    "type": {"enum": ["expression_tree", "model"]},
    "description": {"type": "string"},
    "expression": {"type": "string"},
    "compute": {
      "type": "object",
      "properties": {
        "checkpoint": {"type": "string"},
        "input_features": {"type": "array", "items": {"type": "string"}},
        "sequence_length": {"type": "integer"}
      }
    },
    "normalization": {
      "type": "object",
      "properties": {
        "method": {"enum": ["rolling_zscore", "cross_sectional"]},
        "lookback": {"type": "integer", "minimum": 2}
      }
    },
    "validation": {"type": "object"},
    "weight_hint": {"type": "number", "minimum": 0},
    "horizon": {"type": "string"},
    "meta": {"type": "object"}
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alpha_contract.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/contract.py alphas/schema.json tests/test_alpha_contract.py
git commit -m "feat: add alpha contract schema and JSON loader"
```

---

## Task 5: Normalizer

**Files:**
- Create: `alpha/normalizer.py`
- Test: `tests/test_normalizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_normalizer.py`:

```python
import numpy as np
import pytest
from alpha.normalizer import RollingZScoreNormalizer, CrossSectionalNormalizer


class TestRollingZScoreNormalizer:
    def test_basic_normalization(self):
        norm = RollingZScoreNormalizer(lookback=10)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        result = norm.update("BTC/USDT", values[-1], values)
        # After 10+ values, z-score should be computable
        assert np.isfinite(result)

    def test_warmup_returns_zero(self):
        norm = RollingZScoreNormalizer(lookback=20)
        values = np.array([1.0, 2.0, 3.0])
        result = norm.update("BTC/USDT", values[-1], values)
        assert result == 0.0  # insufficient history

    def test_constant_values_return_zero(self):
        norm = RollingZScoreNormalizer(lookback=5)
        values = np.full(10, 5.0)
        result = norm.update("BTC/USDT", values[-1], values)
        assert result == 0.0  # zero std → zero z-score

    def test_separate_symbols(self):
        norm = RollingZScoreNormalizer(lookback=5)
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        v2 = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
        z1 = norm.update("BTC/USDT", v1[-1], v1)
        z2 = norm.update("ETH/USDT", v2[-1], v2)
        assert z1 > 0  # 100 is way above recent mean
        assert z2 < 0  # 95 is below recent mean


class TestCrossSectionalNormalizer:
    def test_basic_normalization(self):
        norm = CrossSectionalNormalizer()
        scores = {"BTC/USDT": 10.0, "ETH/USDT": 5.0, "SOL/USDT": 0.0}
        result = norm.normalize(scores)
        # BTC should be positive, SOL negative
        assert result["BTC/USDT"] > 0
        assert result["SOL/USDT"] < 0

    def test_single_symbol_returns_zero(self):
        norm = CrossSectionalNormalizer()
        scores = {"BTC/USDT": 42.0}
        result = norm.normalize(scores)
        assert result["BTC/USDT"] == 0.0

    def test_equal_values_return_zero(self):
        norm = CrossSectionalNormalizer()
        scores = {"BTC/USDT": 5.0, "ETH/USDT": 5.0}
        result = norm.normalize(scores)
        assert result["BTC/USDT"] == 0.0
        assert result["ETH/USDT"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_normalizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'alpha.normalizer'`

- [ ] **Step 3: Implement normalizer**

Create `alpha/normalizer.py`:

```python
"""Signal normalization: raw expression output → z-score.

Two methods following AlphaGen's normalize_by_day pattern:
- rolling_zscore: per-symbol time-series normalization
- cross_sectional: across-symbols normalization per tick
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class RollingZScoreNormalizer:
    """Per-symbol rolling z-score: (value - rolling_mean) / rolling_std."""

    def __init__(self, lookback: int = 20):
        self._lookback = lookback

    def update(self, symbol: str, current_value: float, history: np.ndarray) -> float:
        """Compute z-score for current_value given its recent history.

        Args:
            symbol: Symbol name (for logging, not used in computation).
            current_value: The latest raw alpha value.
            history: Array of recent raw values including current_value.

        Returns:
            Z-score float. Returns 0.0 if insufficient history or zero std.
        """
        if len(history) < self._lookback:
            return 0.0

        window = history[-self._lookback:]
        mean = np.nanmean(window)
        std = np.nanstd(window, ddof=1)

        if std < 1e-12 or not np.isfinite(std):
            return 0.0

        z = (current_value - mean) / std
        if not np.isfinite(z):
            return 0.0
        return float(z)


class CrossSectionalNormalizer:
    """Cross-sectional z-score: normalize across all symbols at one point in time."""

    def normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores across symbols.

        Args:
            scores: Dict mapping symbol → raw alpha value.

        Returns:
            Dict mapping symbol → z-score. Returns 0.0 for all if <= 1 symbol or zero std.
        """
        if len(scores) <= 1:
            return {sym: 0.0 for sym in scores}

        values = np.array(list(scores.values()), dtype=np.float64)
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1)

        if std < 1e-12 or not np.isfinite(std):
            return {sym: 0.0 for sym in scores}

        result = {}
        for sym, val in scores.items():
            z = (val - mean) / std
            result[sym] = float(z) if np.isfinite(z) else 0.0
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_normalizer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/normalizer.py tests/test_normalizer.py
git commit -m "feat: add rolling z-score and cross-sectional normalizers"
```

---

## Task 6: Alpha Registry (loads JSONs, evaluates, produces FactorObservation)

**Files:**
- Create: `alpha/registry.py`
- Modify: `features/extractor.py` (add `available_features` class method)
- Test: `tests/test_registry.py`

- [ ] **Step 1: Add available_features to extractor**

Add to `features/extractor.py` after the `FEATURE_NAMES` list:

```python
    @classmethod
    def available_ohlcv_features(cls) -> set:
        """Return the set of OHLCV feature names available for alpha expressions."""
        return {"open", "close", "high", "low", "volume", "vwap"}
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_registry.py`:

```python
import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime
from alpha.registry import AlphaRegistry
from core.models import OHLCV, FactorObservation, FactorSnapshot


def _make_candles(n: int = 100, symbol: str = "BTC/USDT") -> list:
    np.random.seed(42)
    base = 65000.0 + np.cumsum(np.random.randn(n) * 100)
    candles = []
    for i in range(n):
        candles.append(OHLCV(
            symbol=symbol,
            open=base[i] + np.random.randn() * 10,
            high=base[i] + abs(np.random.randn()) * 50,
            low=base[i] - abs(np.random.randn()) * 50,
            close=float(base[i]),
            volume=float(np.random.exponential(100)),
            timestamp=datetime(2026, 1, 1, 0, i),
        ))
    return candles


def _write_alpha(tmp_path: Path, alpha_id: str, expression: str, weight: float = 1.0) -> Path:
    data = {
        "alpha_id": alpha_id,
        "version": "1.0.0",
        "type": "expression_tree",
        "expression": expression,
        "normalization": {"method": "rolling_zscore", "lookback": 20},
        "weight_hint": weight,
        "horizon": "intraday",
        "meta": {"author": "test"},
    }
    f = tmp_path / f"{alpha_id}.json"
    f.write_text(json.dumps(data))
    return f


class TestAlphaRegistry:
    def test_load_single_alpha(self, tmp_path):
        _write_alpha(tmp_path, "test_delta", "Delta($close, 5)")
        registry = AlphaRegistry(alpha_dir=tmp_path)
        assert len(registry.alphas) == 1
        assert "test_delta" in registry.alphas

    def test_load_multiple_alphas(self, tmp_path):
        _write_alpha(tmp_path, "alpha1", "Delta($close, 5)", weight=0.6)
        _write_alpha(tmp_path, "alpha2", "Mean($volume, 10)", weight=0.4)
        registry = AlphaRegistry(alpha_dir=tmp_path)
        assert len(registry.alphas) == 2

    def test_evaluate_produces_factor_snapshot(self, tmp_path):
        _write_alpha(tmp_path, "test_delta", "Delta($close, 5)")
        registry = AlphaRegistry(alpha_dir=tmp_path)
        candles = _make_candles(100)

        snapshot = registry.evaluate("BTC/USDT", candles)
        assert isinstance(snapshot, FactorSnapshot)
        assert snapshot.symbol == "BTC/USDT"
        assert len(snapshot.observations) == 1
        assert isinstance(snapshot.observations[0], FactorObservation)
        assert snapshot.observations[0].name == "test_delta"

    def test_entry_score_is_weighted(self, tmp_path):
        _write_alpha(tmp_path, "a1", "Delta($close, 5)", weight=0.7)
        _write_alpha(tmp_path, "a2", "Mean($close, 10)", weight=0.3)
        registry = AlphaRegistry(alpha_dir=tmp_path)
        candles = _make_candles(100)

        snapshot = registry.evaluate("BTC/USDT", candles)
        assert np.isfinite(snapshot.entry_score)

    def test_invalid_feature_raises_at_load(self, tmp_path):
        data = {
            "alpha_id": "bad",
            "version": "1.0.0",
            "type": "expression_tree",
            "expression": "$nonexistent",
            "normalization": {"method": "rolling_zscore", "lookback": 20},
        }
        (tmp_path / "bad.json").write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Unknown feature"):
            AlphaRegistry(alpha_dir=tmp_path)

    def test_empty_directory(self, tmp_path):
        registry = AlphaRegistry(alpha_dir=tmp_path)
        assert len(registry.alphas) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'alpha.registry'`

- [ ] **Step 4: Implement registry**

Create `alpha/registry.py`:

```python
"""Alpha registry: loads alpha JSONs, evaluates per tick, produces FactorSnapshot.

This is the main interface between the alpha contract layer and the strategy layer.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from alpha.contract import AlphaSpec, AlphaType, NormalizationMethod, load_alpha
from alpha.expression import ExpressionParser, ExpressionEvaluator, ExprNode
from alpha.normalizer import RollingZScoreNormalizer, CrossSectionalNormalizer
from core.models import OHLCV, Bias, FactorObservation, FactorSnapshot

logger = logging.getLogger(__name__)

# System-level signal constants
NOISE_THRESHOLD = 0.3  # |z| below this = NEUTRAL
MAX_STRENGTH_Z = 3.0   # z=3 maps to strength=1.0


class _LoadedAlpha:
    """Internal: a parsed and ready-to-evaluate alpha."""

    def __init__(self, spec: AlphaSpec, ast: ExprNode):
        self.spec = spec
        self.ast = ast


class AlphaRegistry:
    """Loads alpha JSONs at startup, evaluates all alphas per symbol per tick."""

    def __init__(
        self,
        alpha_dir: Optional[Path] = None,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        alphas_cfg = cfg.get("alphas", {})
        self._noise_threshold = alphas_cfg.get("signal", {}).get("noise_threshold", NOISE_THRESHOLD)
        self._max_strength_z = alphas_cfg.get("signal", {}).get("max_strength_z", MAX_STRENGTH_Z)
        norm_lookback = alphas_cfg.get("normalization_lookback", 20)

        self._parser = ExpressionParser()
        self._evaluator = ExpressionEvaluator()
        self._rolling_normalizers: Dict[str, RollingZScoreNormalizer] = {}
        self._cross_normalizer = CrossSectionalNormalizer()
        self._default_norm_lookback = norm_lookback

        # Per-alpha, per-symbol history of raw values for normalization
        self._history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        self._alphas: Dict[str, _LoadedAlpha] = {}

        if alpha_dir is not None:
            self._load_directory(alpha_dir)

    @property
    def alphas(self) -> Dict[str, AlphaSpec]:
        return {k: v.spec for k, v in self._alphas.items()}

    def _load_directory(self, alpha_dir: Path) -> None:
        """Load all .json files from directory, parse expressions, validate."""
        if not alpha_dir.exists():
            logger.warning("Alpha directory does not exist: %s", alpha_dir)
            return

        for f in sorted(alpha_dir.glob("*.json")):
            spec = load_alpha(f)

            if spec.type == AlphaType.EXPRESSION_TREE:
                ast = self._parser.parse(spec.expression)
                self._alphas[spec.alpha_id] = _LoadedAlpha(spec, ast)
                logger.info(
                    "Loaded alpha: %s (features: %s, weight: %.2f)",
                    spec.alpha_id,
                    sorted(ast.referenced_features()),
                    spec.weight_hint,
                )
            elif spec.type == AlphaType.MODEL:
                logger.info(
                    "Skipping model alpha %s (requires model_inference plugin)",
                    spec.alpha_id,
                )

    def evaluate(self, symbol: str, candles: List[OHLCV]) -> FactorSnapshot:
        """Evaluate all loaded alphas for a symbol, return aggregated FactorSnapshot."""
        now = candles[-1].timestamp if candles else datetime.utcnow()

        # Build OHLCV arrays for expression evaluation
        data = {
            "open": np.array([c.open for c in candles], dtype=np.float64),
            "high": np.array([c.high for c in candles], dtype=np.float64),
            "low": np.array([c.low for c in candles], dtype=np.float64),
            "close": np.array([c.close for c in candles], dtype=np.float64),
            "volume": np.array([c.volume for c in candles], dtype=np.float64),
            "vwap": np.array(
                [(c.high + c.low + c.close) / 3.0 for c in candles],
                dtype=np.float64,
            ),
        }

        observations: List[FactorObservation] = []
        weighted_scores: List[float] = []
        total_weight = 0.0

        for alpha_id, loaded in self._alphas.items():
            spec = loaded.spec

            # Evaluate expression
            raw_values = self._evaluator.evaluate(loaded.ast, data)
            current_raw = raw_values[-1] if len(raw_values) > 0 else np.nan

            if not np.isfinite(current_raw):
                observations.append(FactorObservation(
                    name=alpha_id, symbol=symbol, bias=Bias.NEUTRAL,
                    strength=0.0, timestamp=now,
                    thesis=f"{alpha_id}: no signal (NaN)",
                ))
                continue

            # Track history for normalization
            self._history[alpha_id][symbol].append(current_raw)
            hist = np.array(self._history[alpha_id][symbol])

            # Normalize
            lookback = spec.normalization.lookback
            if spec.normalization.method == NormalizationMethod.ROLLING_ZSCORE:
                key = f"{alpha_id}_{lookback}"
                if key not in self._rolling_normalizers:
                    self._rolling_normalizers[key] = RollingZScoreNormalizer(lookback)
                z = self._rolling_normalizers[key].update(symbol, current_raw, hist)
            else:
                # Cross-sectional handled at portfolio level; use raw for now
                z = current_raw

            # Signal extraction (AlphaGen logic)
            if abs(z) < self._noise_threshold:
                bias = Bias.NEUTRAL
            elif z > 0:
                bias = Bias.BULLISH
            else:
                bias = Bias.BEARISH

            strength = min(abs(z) / self._max_strength_z, 1.0)

            obs = FactorObservation(
                name=alpha_id,
                symbol=symbol,
                bias=bias,
                strength=strength,
                timestamp=now,
                thesis=f"{alpha_id}: z={z:.2f}",
            )
            observations.append(obs)

            # Weighted score for aggregation
            signed_strength = strength if bias == Bias.BULLISH else (-strength if bias == Bias.BEARISH else 0.0)
            weighted_scores.append(signed_strength * spec.weight_hint)
            total_weight += spec.weight_hint

        # Aggregate: weighted average
        if total_weight > 0 and weighted_scores:
            entry_score = sum(weighted_scores) / total_weight
        else:
            entry_score = 0.0

        return FactorSnapshot(
            symbol=symbol,
            timestamp=now,
            observations=observations,
            entry_score=entry_score,
            exit_score=entry_score,  # symmetric for MVP
            confidence=min(abs(entry_score), 1.0),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add alpha/registry.py features/extractor.py tests/test_registry.py
git commit -m "feat: add alpha registry — loads JSONs, evaluates expressions, produces FactorSnapshot"
```

---

## Task 7: Portfolio Optimizer (MVP)

**Files:**
- Create: `strategy/optimizer.py`, `strategy/sizing.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_optimizer.py`:

```python
import pytest
from datetime import datetime
from core.models import FactorSnapshot, FactorObservation, Bias
from strategy.optimizer import PortfolioOptimizer


def _snap(symbol: str, entry_score: float) -> FactorSnapshot:
    return FactorSnapshot(
        symbol=symbol,
        timestamp=datetime(2026, 1, 1),
        entry_score=entry_score,
        exit_score=entry_score,
        confidence=abs(entry_score),
    )


class TestEqualWeight:
    def test_two_symbols(self):
        opt = PortfolioOptimizer(mode="equal_weight", max_positions=2)
        snapshots = [_snap("BTC/USDT", 0.8), _snap("ETH/USDT", 0.6)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] == pytest.approx(0.5)
        assert weights["ETH/USDT"] == pytest.approx(0.5)

    def test_respects_max_positions(self):
        opt = PortfolioOptimizer(mode="equal_weight", max_positions=1)
        snapshots = [_snap("BTC/USDT", 0.8), _snap("ETH/USDT", 0.6)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] == pytest.approx(1.0)
        assert "ETH/USDT" not in weights or weights["ETH/USDT"] == 0.0

    def test_filters_negative_scores(self):
        opt = PortfolioOptimizer(mode="equal_weight", max_positions=2)
        snapshots = [_snap("BTC/USDT", 0.8), _snap("ETH/USDT", -0.3)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] == pytest.approx(1.0)


class TestScoreTilted:
    def test_higher_score_gets_more_weight(self):
        opt = PortfolioOptimizer(mode="score_tilted", max_positions=2, temperature=0.8)
        snapshots = [_snap("BTC/USDT", 0.9), _snap("ETH/USDT", 0.3)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] > weights["ETH/USDT"]

    def test_max_single_weight_cap(self):
        opt = PortfolioOptimizer(
            mode="score_tilted", max_positions=2,
            temperature=0.8, max_single_weight=0.6,
        )
        snapshots = [_snap("BTC/USDT", 0.99), _snap("ETH/USDT", 0.01)]
        weights = opt.allocate(snapshots)
        assert weights["BTC/USDT"] <= 0.6 + 1e-9

    def test_weights_sum_to_one(self):
        opt = PortfolioOptimizer(mode="score_tilted", max_positions=3, temperature=0.8)
        snapshots = [_snap("BTC", 0.8), _snap("ETH", 0.6), _snap("SOL", 0.4)]
        weights = opt.allocate(snapshots)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestEmptyInput:
    def test_no_snapshots(self):
        opt = PortfolioOptimizer(mode="equal_weight", max_positions=2)
        weights = opt.allocate([])
        assert weights == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement optimizer**

Create `strategy/optimizer.py`:

```python
"""Portfolio-level allocation: FactorSnapshots → target weights per symbol.

MVP modes:
  - equal_weight: uniform 1/N for top-N positive-score symbols
  - score_tilted: softmax on entry scores with temperature and per-symbol cap

Future work: mean-variance, risk-parity.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from core.models import FactorSnapshot

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Allocate portfolio weights across symbols based on factor snapshots."""

    def __init__(
        self,
        mode: str = "equal_weight",
        max_positions: int = 2,
        temperature: float = 0.8,
        max_single_weight: float = 1.0,
    ):
        self._mode = mode
        self._max_positions = max_positions
        self._temperature = temperature
        self._max_single_weight = max_single_weight

    def allocate(self, snapshots: List[FactorSnapshot]) -> Dict[str, float]:
        """Return target weight per symbol. Weights sum to 1.0 (or 0.0 if empty)."""
        # Filter to positive entry scores only (long-only)
        candidates = [(s.symbol, s.entry_score) for s in snapshots if s.entry_score > 0]
        if not candidates:
            return {}

        # Sort by score descending, take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:self._max_positions]

        if self._mode == "equal_weight":
            return self._equal_weight(selected)
        elif self._mode == "score_tilted":
            return self._score_tilted(selected)
        else:
            logger.warning("Unknown optimizer mode '%s', falling back to equal_weight", self._mode)
            return self._equal_weight(selected)

    def _equal_weight(self, selected: List[tuple]) -> Dict[str, float]:
        n = len(selected)
        if n == 0:
            return {}
        w = 1.0 / n
        return {sym: w for sym, _ in selected}

    def _score_tilted(self, selected: List[tuple]) -> Dict[str, float]:
        scores = np.array([s for _, s in selected], dtype=np.float64)

        # Softmax with temperature
        scaled = scores / self._temperature
        scaled -= scaled.max()  # numerical stability
        exp_scores = np.exp(scaled)
        weights = exp_scores / exp_scores.sum()

        # Apply per-symbol cap
        weights = np.minimum(weights, self._max_single_weight)

        # Re-normalize after capping
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return {sym: float(w) for (sym, _), w in zip(selected, weights)}
```

Create `strategy/sizing.py`:

```python
"""Per-position sizing: target weight → concrete quantity.

Converts portfolio weight + current prices into order quantities,
applying Kelly criterion bounds.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """Convert target weight to position quantity."""

    def __init__(
        self,
        base_size_pct: float = 0.05,
        max_size_pct: float = 0.15,
        kelly_fraction: float = 0.5,
    ):
        self._base = base_size_pct
        self._max = max_size_pct
        self._kelly = kelly_fraction

    def compute_quantity(
        self,
        target_weight: float,
        nav: float,
        price: float,
        confidence: float = 1.0,
    ) -> float:
        """Compute order quantity given target weight and portfolio state.

        Args:
            target_weight: Portfolio weight [0, 1] from optimizer.
            nav: Current net asset value.
            price: Current asset price.
            confidence: Signal confidence [0, 1] for Kelly scaling.

        Returns:
            Quantity to buy (in base asset units).
        """
        if price <= 0 or nav <= 0:
            return 0.0

        # Scale target weight by Kelly fraction and confidence
        effective_weight = target_weight * self._kelly * confidence

        # Clamp to [base, max] of portfolio
        effective_weight = max(self._base, min(self._max, effective_weight))

        notional = nav * effective_weight
        quantity = notional / price

        return max(0.0, quantity)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add strategy/optimizer.py strategy/sizing.py tests/test_optimizer.py
git commit -m "feat: add portfolio optimizer (equal_weight, score_tilted) and position sizer"
```

---

## Task 8: Built-in Alpha JSON Files

**Files:**
- Create: `alphas/builtin/momentum_impulse.json`, `alphas/builtin/trend_alignment.json`, `alphas/builtin/volume_confirmation.json`, `alphas/builtin/mean_reversion.json`
- Create: `alphas/examples/model_checkpoint_example.json`

These convert the current hardcoded factors from the old `factor_engine.py` (on branch `codex/core-satellite-best-20260325`) into expression-tree JSON format.

- [ ] **Step 1: Create builtin alphas directory**

```bash
mkdir -p alphas/builtin alphas/examples
```

- [ ] **Step 2: Create momentum_impulse.json**

Create `alphas/builtin/momentum_impulse.json`:

```json
{
  "alpha_id": "momentum_impulse",
  "version": "1.0.0",
  "type": "expression_tree",
  "description": "Short-term price momentum weighted by volume spike. Captures momentum bursts confirmed by unusual trading activity.",
  "expression": "Mul(Delta($close, 10), Div($volume, Mean($volume, 20)))",
  "normalization": {
    "method": "rolling_zscore",
    "lookback": 20
  },
  "validation": {
    "ic": 0.038,
    "icir": 0.55
  },
  "weight_hint": 0.30,
  "horizon": "intraday",
  "meta": {
    "author": "manual",
    "source_repo": "systematic_crypto_trading_bot"
  }
}
```

- [ ] **Step 3: Create trend_alignment.json**

Create `alphas/builtin/trend_alignment.json`:

```json
{
  "alpha_id": "trend_alignment",
  "version": "1.0.0",
  "type": "expression_tree",
  "description": "EMA crossover: fast EMA above slow EMA indicates uptrend alignment. Normalized by price level.",
  "expression": "Div(Sub(EMA($close, 12), EMA($close, 26)), EMA($close, 26))",
  "normalization": {
    "method": "rolling_zscore",
    "lookback": 30
  },
  "validation": {
    "ic": 0.042,
    "icir": 0.60
  },
  "weight_hint": 0.30,
  "horizon": "intraday",
  "meta": {
    "author": "manual",
    "source_repo": "systematic_crypto_trading_bot"
  }
}
```

- [ ] **Step 4: Create volume_confirmation.json**

Create `alphas/builtin/volume_confirmation.json`:

```json
{
  "alpha_id": "volume_confirmation",
  "version": "1.0.0",
  "type": "expression_tree",
  "description": "Volume ratio vs 20-bar average. High ratio confirms conviction behind price moves.",
  "expression": "Sub(Div($volume, Mean($volume, 20)), 1.0)",
  "normalization": {
    "method": "rolling_zscore",
    "lookback": 20
  },
  "validation": {
    "ic": 0.025,
    "icir": 0.40
  },
  "weight_hint": 0.20,
  "horizon": "intraday",
  "meta": {
    "author": "manual",
    "source_repo": "systematic_crypto_trading_bot"
  }
}
```

- [ ] **Step 5: Create mean_reversion.json**

Create `alphas/builtin/mean_reversion.json`:

```json
{
  "alpha_id": "mean_reversion",
  "version": "1.0.0",
  "type": "expression_tree",
  "description": "Deviation from 20-bar mean, inverted. Captures short-term mean reversion when price overshoots.",
  "expression": "Mul(Sub(Mean($close, 20), $close), Div(1.0, Std($close, 20)))",
  "normalization": {
    "method": "rolling_zscore",
    "lookback": 30
  },
  "validation": {
    "ic": 0.030,
    "icir": 0.45
  },
  "weight_hint": 0.20,
  "horizon": "intraday",
  "meta": {
    "author": "manual",
    "source_repo": "systematic_crypto_trading_bot"
  }
}
```

- [ ] **Step 6: Create model example**

Create `alphas/examples/model_checkpoint_example.json`:

```json
{
  "alpha_id": "lstm_btc_v1",
  "version": "1.0.0",
  "type": "model",
  "description": "Example: LSTM model alpha (requires model_inference plugin). Not loaded by default.",
  "compute": {
    "checkpoint": "artifacts/models/lstm_btc_v1.onnx",
    "input_features": ["rsi", "ema_fast", "ema_slow", "atr", "momentum", "volatility", "order_book_imbalance", "volume_ratio", "funding_rate", "taker_ratio"],
    "sequence_length": 240
  },
  "normalization": {
    "method": "rolling_zscore",
    "lookback": 20
  },
  "weight_hint": 0.20,
  "horizon": "intraday",
  "meta": {
    "author": "lstm_trainer",
    "source_repo": "trading_competition"
  }
}
```

- [ ] **Step 7: Test that registry loads builtins**

Run: `uv run python -c "from alpha.registry import AlphaRegistry; from pathlib import Path; r = AlphaRegistry(alpha_dir=Path('alphas/builtin')); print(f'Loaded {len(r.alphas)} alphas:', list(r.alphas.keys()))"`
Expected: `Loaded 4 alphas: ['mean_reversion', 'momentum_impulse', 'trend_alignment', 'volume_confirmation']`

- [ ] **Step 8: Commit**

```bash
git add alphas/
git commit -m "feat: add 4 built-in expression-tree alphas and model example"
```

---

## Task 9: Wire Alpha Registry into Strategy Monitor + main.py

**Files:**
- Modify: `strategy/monitor.py`, `main.py`, `config/default.yaml`

This is the integration task — replace `AlphaEngine` with `AlphaRegistry` in the main pipeline.

- [ ] **Step 1: Update config/default.yaml**

Replace the current config with the simplified version from the spec. Keep the existing structure for fields that are still used, replace the `alpha` section:

In `config/default.yaml`, replace the `alpha:` section with:

```yaml
alphas:
  directory: alphas/builtin
  normalization_default: rolling_zscore
  normalization_lookback: 20
  signal:
    noise_threshold: 0.3
    max_strength_z: 3.0

optimizer:
  mode: score_tilted
  max_positions: 2
  temperature: 0.8
  max_single_weight: 0.35
```

Keep all other sections (symbols, features, strategy, risk, execution, paper) as they are.

- [ ] **Step 2: Update main.py imports and initialization**

In `main.py`, add the new imports:

```python
from alpha.registry import AlphaRegistry
from strategy.optimizer import PortfolioOptimizer
```

In the main startup function (where `AlphaEngine` is currently constructed), replace with:

```python
# Alpha registry
alphas_cfg = config.get("alphas", {})
alpha_dir = Path(args.alphas if hasattr(args, "alphas") and args.alphas else alphas_cfg.get("directory", "alphas/builtin"))
alpha_registry = AlphaRegistry(alpha_dir=alpha_dir, config=config)
logger.info("Loaded %d alphas from %s", len(alpha_registry.alphas), alpha_dir)

# Portfolio optimizer
opt_cfg = config.get("optimizer", {})
optimizer = PortfolioOptimizer(
    mode=opt_cfg.get("mode", "score_tilted"),
    max_positions=opt_cfg.get("max_positions", 2),
    temperature=opt_cfg.get("temperature", 0.8),
    max_single_weight=opt_cfg.get("max_single_weight", 0.35),
)
```

Add `--alphas` argument to argparse:

```python
parser.add_argument("--alphas", type=str, default=None, help="Path to alpha JSON directory")
```

Pass `alpha_registry` and `optimizer` to `StrategyMonitor` instead of `alpha_engine`.

- [ ] **Step 3: Update strategy/monitor.py**

Replace `AlphaEngine` dependency with `AlphaRegistry`. In `__init__`:

```python
def __init__(
    self,
    config: dict,
    buffer: LiveBuffer,
    extractor: FeatureExtractor,
    alpha_registry: AlphaRegistry,  # was: alpha_engine: AlphaEngine
    risk_shield: RiskShield,
    tracker: PortfolioTracker,
    order_manager: OrderManager,
    optimizer: PortfolioOptimizer,  # NEW
    resampler=None,
    multi_resampler=None,
    executor=None,
):
```

In the main loop where `self._alpha_engine.score()` is called, replace with:

```python
# Evaluate all alphas for this symbol
snapshot = self._alpha_registry.evaluate(symbol, candles)

# Strategy logic consumes FactorSnapshot
strategy = self._strategies[symbol]
# ... pass snapshot to strategy logic
```

- [ ] **Step 4: Run existing tests to check for breakage**

Run: `uv run pytest -x -q`
Expected: Some tests may fail if they mock `AlphaEngine`. Fix these in the next task.

- [ ] **Step 5: Commit**

```bash
git add strategy/monitor.py main.py config/default.yaml
git commit -m "feat: wire AlphaRegistry and PortfolioOptimizer into main pipeline"
```

---

## Task 10: Extract Plugins

**Files:**
- Create: `plugins/__init__.py`, `plugins/roostoo/__init__.py`, `plugins/roostoo/executor.py`, `plugins/roostoo/auth.py`, `plugins/roostoo/README.md`
- Create: `plugins/model_inference/__init__.py`, `plugins/model_inference/evaluator.py`, `plugins/model_inference/model_wrapper.py`, `plugins/model_inference/README.md`
- Modify: `main.py` (conditional plugin imports)
- Delete: `execution/roostoo_executor.py`, `data/roostoo_auth.py`

- [ ] **Step 1: Create plugin directories**

```bash
mkdir -p plugins/roostoo plugins/model_inference
```

- [ ] **Step 2: Move Roostoo files**

```bash
git mv execution/roostoo_executor.py plugins/roostoo/executor.py
git mv data/roostoo_auth.py plugins/roostoo/auth.py
```

Create `plugins/__init__.py` and `plugins/roostoo/__init__.py` as empty files.

- [ ] **Step 3: Update Roostoo imports**

In `plugins/roostoo/executor.py`, update the import path for `roostoo_auth`:

```python
from plugins.roostoo.auth import RoostooAuth  # was: from data.roostoo_auth import ...
```

- [ ] **Step 4: Move model files**

```bash
git mv models/inference.py plugins/model_inference/evaluator.py
git mv models/model_wrapper.py plugins/model_inference/model_wrapper.py
```

Create `plugins/model_inference/__init__.py` as empty file.

- [ ] **Step 5: Update main.py for conditional plugin loading**

In `main.py`, replace unconditional Roostoo import with:

```python
# Plugin loading
mode = config.get("mode", "paper")
plugins_cfg = config.get("plugins", {})

executor = None
if mode == "roostoo" or plugins_cfg.get("roostoo", {}).get("enabled"):
    from plugins.roostoo.executor import RoostooExecutor
    executor = RoostooExecutor(config)
elif mode == "live":
    executor = LiveExecutor(config.get("exchange", {}))
else:
    executor = SimExecutor(config)
```

- [ ] **Step 6: Write plugin READMEs**

Create `plugins/roostoo/README.md`:

```markdown
# Roostoo Exchange Plugin

Exchange integration for the Roostoo trading competition.

## Setup

Set environment variables:
- `ROOSTOO_API_KEY`
- `ROOSTOO_API_SECRET`
- `ROOSTOO_COMP_ID` (competition ID)

## Usage

```bash
python main.py --mode roostoo
```

Or enable in config:
```yaml
plugins:
  roostoo:
    enabled: true
```
```

Create `plugins/model_inference/README.md`:

```markdown
# Model Inference Plugin

ONNX/PyTorch model inference for alpha generation.

## When to use

Use when your alpha requires sequential pattern recognition that
expression trees cannot capture (e.g., LSTM on 240-candle sequences).

Expression trees are preferred for most use cases because they are
self-contained, auditable, and require no binary checkpoints.

## Setup

Place model checkpoints in `artifacts/models/`. Create an alpha JSON
with `"type": "model"` pointing to the checkpoint path.

## Usage

Enable in config:
```yaml
plugins:
  model_inference:
    enabled: true
```
```

- [ ] **Step 7: Commit**

```bash
git add plugins/ main.py
git rm models/lstm_model.py models/transformer_model.py models/train.py models/icir_tracker.py models/__init__.py
git commit -m "refactor: extract Roostoo and model inference into plugins/"
```

---

## Task 11: Delete Dead Code

**Files:**
- Delete: `strategy/trade_tracker.py`, `scripts/upload_model_to_hf.py`, and other files per spec removal list

- [ ] **Step 1: Remove files**

```bash
git rm strategy/trade_tracker.py
git rm scripts/upload_model_to_hf.py
```

- [ ] **Step 2: Remove unused test files that reference deleted modules**

Check which tests import deleted modules:

```bash
grep -l "from models.train\|from models.lstm\|from models.transformer\|from models.icir\|trade_tracker" tests/*.py
```

Remove or update these test files as needed.

- [ ] **Step 3: Run tests**

Run: `uv run pytest -x -q`
Fix any import errors from deleted modules.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove competition-specific and training code"
```

---

## Task 12: Update Tests for New Architecture

**Files:**
- Modify: various `tests/test_*.py`
- Create: any missing tests

- [ ] **Step 1: Fix broken imports in existing tests**

Run `uv run pytest --collect-only 2>&1 | grep "Error"` to find all import errors.

For each broken test, either:
- Update imports to point to new locations
- Remove tests for deleted functionality
- Add new tests for new architecture

- [ ] **Step 2: Ensure all tests pass**

Run: `uv run pytest -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: update test suite for new alpha registry architecture"
```

---

## Task 13: README Rewrite

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Write the architecture-first README**

Rewrite `README.md` following the structure from the spec (Section 8). The README should include:

1. Title + one-line tagline
2. Architecture pipeline diagram (mermaid)
3. Decision chain with typed contracts
4. Alpha contract section with JSON example
5. Quick start (3 commands)
6. Module map table
7. Plugin system description
8. Design decisions section
9. Future work (optimizer, signal pipeline, hot-reload)
10. Acknowledgments (teammate + competition)

- [ ] **Step 2: Verify quick start commands work**

Run the quick start commands from the README:

```bash
uv sync
uv run python main.py
```

Expected: Paper trading starts without errors, processes synthetic candles.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: architecture-first README for portfolio showcase"
```

---

## Task 14: Final Cleanup and Push

**Files:**
- Modify: `pyproject.toml` (clean up unused dependencies)
- Delete: `ARCHITECTURE.md`, `REPOSPEC.md`, `notes/` (if still present)
- Update: `.gitignore`

- [ ] **Step 1: Remove unused dependencies from pyproject.toml**

Remove `wandb`, `torch` (unless needed by plugins — make optional), `onnxruntime` (plugin-only).

Move plugin dependencies to optional groups:

```toml
[project.optional-dependencies]
model = ["torch>=2.0", "onnxruntime>=1.19"]
roostoo = ["aiohttp>=3.9"]
```

- [ ] **Step 2: Clean up stale files**

```bash
git rm -f ARCHITECTURE.md REPOSPEC.md 2>/dev/null || true
rm -rf notes/ 2>/dev/null || true
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: All PASS

- [ ] **Step 4: Final commit and push**

```bash
git add -A
git commit -m "chore: final cleanup — remove stale docs, optional plugin deps"
git push origin main
```

---

## Execution Order Summary

| Task | What | Depends on |
|---|---|---|
| 1 | Fork & setup | — |
| 2 | Domain models | 1 |
| 3 | Expression evaluator | 1 |
| 4 | Alpha contract | 3 |
| 5 | Normalizer | 1 |
| 6 | Alpha registry | 2, 3, 4, 5 |
| 7 | Portfolio optimizer | 2 |
| 8 | Built-in alpha JSONs | 3, 4 |
| 9 | Wire into pipeline | 6, 7, 8 |
| 10 | Extract plugins | 9 |
| 11 | Delete dead code | 10 |
| 12 | Update tests | 11 |
| 13 | README | 12 |
| 14 | Final cleanup | 13 |

**Parallelizable:** Tasks 2, 3, 5, 7 can run in parallel (no dependencies on each other). Tasks 4, 8 can run in parallel after 3.
