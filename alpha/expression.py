"""
Expression-tree parser and numpy evaluator.

Supports parsing strings like:
    Mul(Delta($close, 5), Div($volume, Mean($volume, 20)))

into an AST and evaluating them over numpy arrays of OHLCV data.

Operators supported:
    Unary:        Abs, Sign, Log, CSRank
    Binary:       Add, Sub, Mul, Div, Pow, Greater, Less
    Rolling:      Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min, Med, Mad,
                  Rank, Delta, WMA, EMA
    Pair rolling: Cov, Corr
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_FEATURES: Set[str] = {"open", "close", "high", "low", "volume", "vwap"}

# Operator registry: name -> (category, n_expr_args)
#   category:     "unary" | "binary" | "rolling" | "pair_rolling"
#   n_expr_args:  number of sub-expression arguments (window comes after them)
_OPERATORS: Dict[str, tuple] = {
    # Unary
    "Abs":      ("unary",       1),
    "Sign":     ("unary",       1),
    "Log":      ("unary",       1),
    "CSRank":   ("unary",       1),
    # Binary
    "Add":      ("binary",      2),
    "Sub":      ("binary",      2),
    "Mul":      ("binary",      2),
    "Div":      ("binary",      2),
    "Pow":      ("binary",      2),
    "Greater":  ("binary",      2),
    "Less":     ("binary",      2),
    # Rolling (1 expr arg + window)
    "Ref":      ("rolling",     1),
    "Mean":     ("rolling",     1),
    "Sum":      ("rolling",     1),
    "Std":      ("rolling",     1),
    "Var":      ("rolling",     1),
    "Skew":     ("rolling",     1),
    "Kurt":     ("rolling",     1),
    "Max":      ("rolling",     1),
    "Min":      ("rolling",     1),
    "Med":      ("rolling",     1),
    "Mad":      ("rolling",     1),
    "Rank":     ("rolling",     1),
    "Delta":    ("rolling",     1),
    "WMA":      ("rolling",     1),
    "EMA":      ("rolling",     1),
    # Pair rolling (2 expr args + window)
    "Cov":      ("pair_rolling", 2),
    "Corr":     ("pair_rolling", 2),
}

# Tokenizer regex
_TOKEN_RE = re.compile(r"([+-]?\d+\.?\d*|\$\w+|\w+|[(),])")


# ---------------------------------------------------------------------------
# AST node
# ---------------------------------------------------------------------------

@dataclass
class ExprNode:
    """Node in the expression AST."""
    kind: str                           # "feature" | "constant" | "operator"
    value: Any = None                   # feature name (str) or constant (float)
    name: Optional[str] = None          # operator name
    children: List["ExprNode"] = field(default_factory=list)
    window: Optional[int] = None        # rolling window size

    def referenced_features(self) -> Set[str]:
        """Return set of feature names used by this node and its descendants."""
        if self.kind == "feature":
            return {self.value}
        result: Set[str] = set()
        for child in self.children:
            result |= child.referenced_features()
        return result


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ExpressionParser:
    """Recursive-descent parser for alpha expression strings."""

    def parse(self, text: str) -> ExprNode:
        tokens = _TOKEN_RE.findall(text.strip())
        node, pos = self._parse_expr(tokens, 0)
        if pos != len(tokens):
            raise ValueError(f"Unexpected tokens after expression: {tokens[pos:]}")
        return node

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_expr(self, tokens: List[str], pos: int) -> tuple[ExprNode, int]:
        """Parse one expression starting at *pos*, return (node, new_pos)."""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")

        tok = tokens[pos]

        # Feature: starts with $
        if tok.startswith("$"):
            fname = tok[1:]
            if fname not in VALID_FEATURES:
                raise ValueError(f"Unknown feature: ${fname}")
            return ExprNode(kind="feature", value=fname), pos + 1

        # Numeric constant
        try:
            val = float(tok)
            return ExprNode(kind="constant", value=val), pos + 1
        except ValueError:
            pass

        # Operator call: Name(...)
        if tok[0].isupper():
            if tok not in _OPERATORS:
                raise ValueError(f"Unknown operator: {tok}")
            return self._parse_operator(tokens, pos)

        raise ValueError(f"Cannot parse token: {tok!r}")

    def _parse_operator(self, tokens: List[str], pos: int) -> tuple[ExprNode, int]:
        name = tokens[pos]
        pos += 1
        category, n_expr_args = _OPERATORS[name]

        # Expect '('
        if pos >= len(tokens) or tokens[pos] != "(":
            raise ValueError(f"Expected '(' after operator {name}")
        pos += 1

        children: List[ExprNode] = []
        window: Optional[int] = None

        # Parse expression arguments
        for i in range(n_expr_args):
            child, pos = self._parse_expr(tokens, pos)
            children.append(child)
            if i < n_expr_args - 1:
                if pos >= len(tokens) or tokens[pos] != ",":
                    raise ValueError(f"Expected ',' between arguments of {name}")
                pos += 1

        # Parse window argument for rolling / pair_rolling
        if category in ("rolling", "pair_rolling"):
            if pos >= len(tokens) or tokens[pos] != ",":
                raise ValueError(f"Expected ',' before window in {name}")
            pos += 1
            if pos >= len(tokens):
                raise ValueError(f"Expected window integer in {name}")
            try:
                window = int(tokens[pos])
            except ValueError:
                raise ValueError(f"Window must be an integer in {name}, got {tokens[pos]!r}")
            pos += 1

        # Expect ')'
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError(f"Expected ')' to close {name}, got {tokens[pos] if pos < len(tokens) else 'EOF'!r}")
        pos += 1

        return ExprNode(kind="operator", name=name, children=children, window=window), pos


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ExpressionEvaluator:
    """Evaluates an ExprNode AST over a dict of 1-D numpy arrays."""

    def evaluate(self, node: ExprNode, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a 1-D float64 array with NaN for bars with insufficient history."""
        n = self._infer_length(data)
        return self._eval(node, data, n)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _infer_length(self, data: Dict[str, np.ndarray]) -> int:
        lengths = {len(v) for v in data.values()}
        if len(lengths) != 1:
            raise ValueError(f"All feature arrays must have the same length, got: {lengths}")
        return lengths.pop()

    def _eval(self, node: ExprNode, data: Dict[str, np.ndarray], n: int) -> np.ndarray:
        if node.kind == "feature":
            return data[node.value].astype(np.float64)

        if node.kind == "constant":
            return np.full(n, node.value, dtype=np.float64)

        # Operator
        category, _ = _OPERATORS[node.name]
        if category == "unary":
            return self._eval_unary(node, data, n)
        if category == "binary":
            return self._eval_binary(node, data, n)
        if category == "rolling":
            return self._eval_rolling(node, data, n)
        if category == "pair_rolling":
            return self._eval_pair_rolling(node, data, n)
        raise ValueError(f"Unknown category for operator {node.name}")

    # ------------------------------------------------------------------
    # Unary
    # ------------------------------------------------------------------

    def _eval_unary(self, node: ExprNode, data: Dict[str, np.ndarray], n: int) -> np.ndarray:
        x = self._eval(node.children[0], data, n)
        name = node.name

        if name == "Abs":
            return np.abs(x)
        if name == "Sign":
            return np.sign(x)
        if name == "Log":
            out = np.where(x > 0, np.log(x), np.nan)
            return out.astype(np.float64)
        if name == "CSRank":
            # Single-symbol: always return 0.5
            return np.full(n, 0.5, dtype=np.float64)
        raise ValueError(f"Unknown unary operator: {node.name}")

    # ------------------------------------------------------------------
    # Binary
    # ------------------------------------------------------------------

    def _eval_binary(self, node: ExprNode, data: Dict[str, np.ndarray], n: int) -> np.ndarray:
        x = self._eval(node.children[0], data, n)
        y = self._eval(node.children[1], data, n)
        name = node.name

        if name == "Add":
            return x + y
        if name == "Sub":
            return x - y
        if name == "Mul":
            return x * y
        if name == "Div":
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(y != 0, x / y, np.nan)
            return out.astype(np.float64)
        if name == "Pow":
            return np.power(x, y)
        if name == "Greater":
            return (x > y).astype(np.float64)
        if name == "Less":
            return (x < y).astype(np.float64)
        raise ValueError(f"Unknown binary operator: {node.name}")

    # ------------------------------------------------------------------
    # Rolling (single series)
    # ------------------------------------------------------------------

    def _eval_rolling(self, node: ExprNode, data: Dict[str, np.ndarray], n: int) -> np.ndarray:
        x = self._eval(node.children[0], data, n)
        w = node.window
        name = node.name

        # Special cases that don't use a sliding window the same way
        if name == "Ref":
            return self._rolling_ref(x, w)
        if name == "Delta":
            return self._rolling_delta(x, w)
        if name == "EMA":
            return self._rolling_ema(x, w)
        if name == "WMA":
            return self._rolling_wma(x, w)
        if name == "Rank":
            return self._rolling_rank(x, w)

        # Generic sliding-window aggregation
        return self._rolling_agg(x, w, name)

    def _rolling_ref(self, x: np.ndarray, w: int) -> np.ndarray:
        """Return x shifted back by w bars."""
        out = np.full_like(x, np.nan)
        if w < len(x):
            out[w:] = x[:-w] if w > 0 else x
        return out

    def _rolling_delta(self, x: np.ndarray, w: int) -> np.ndarray:
        """x[t] - x[t-w]."""
        out = np.full_like(x, np.nan)
        if w < len(x):
            out[w:] = x[w:] - x[:-w]
        return out

    def _rolling_ema(self, x: np.ndarray, w: int) -> np.ndarray:
        """Exponential moving average with alpha = 2/(w+1).
        First w-1 values are NaN; index w-1 is seeded with the first valid value.
        """
        alpha = 2.0 / (w + 1)
        out = np.full_like(x, np.nan, dtype=np.float64)
        n = len(x)
        # Find the first non-NaN index
        start = 0
        while start < n and np.isnan(x[start]):
            start += 1
        if start >= n:
            return out
        # Seed at start; mark first w-1 bars as NaN (warmup)
        ema = x[start]
        for i in range(start, n):
            if np.isnan(x[i]):
                ema = np.nan
            else:
                ema = alpha * x[i] + (1 - alpha) * ema
            if i >= start + w - 1:
                out[i] = ema
        return out

    def _rolling_wma(self, x: np.ndarray, w: int) -> np.ndarray:
        """Weighted moving average with linearly increasing weights 1..w."""
        weights = np.arange(1, w + 1, dtype=np.float64)
        wsum = weights.sum()
        out = np.full_like(x, np.nan, dtype=np.float64)
        n = len(x)
        for i in range(w - 1, n):
            window = x[i - w + 1: i + 1]
            if np.any(np.isnan(window)):
                continue
            out[i] = np.dot(weights, window) / wsum
        return out

    def _rolling_rank(self, x: np.ndarray, w: int) -> np.ndarray:
        """Rank of the last value within the rolling window, normalised to [0, 1]."""
        out = np.full_like(x, np.nan, dtype=np.float64)
        n = len(x)
        for i in range(w - 1, n):
            window = x[i - w + 1: i + 1]
            if np.any(np.isnan(window)):
                continue
            rank = np.sum(window < window[-1]) / (w - 1) if w > 1 else 0.5
            out[i] = rank
        return out

    def _rolling_agg(self, x: np.ndarray, w: int, name: str) -> np.ndarray:
        """Generic sliding-window aggregation."""
        out = np.full_like(x, np.nan, dtype=np.float64)
        n = len(x)
        for i in range(w - 1, n):
            window = x[i - w + 1: i + 1]
            if np.any(np.isnan(window)):
                continue
            out[i] = self._agg(window, name, w)
        return out

    def _agg(self, window: np.ndarray, name: str, w: int) -> float:
        if name == "Mean":
            return float(np.mean(window))
        if name == "Sum":
            return float(np.sum(window))
        if name == "Std":
            return float(np.std(window, ddof=1)) if w > 1 else 0.0
        if name == "Var":
            return float(np.var(window, ddof=1)) if w > 1 else 0.0
        if name == "Max":
            return float(np.max(window))
        if name == "Min":
            return float(np.min(window))
        if name == "Med":
            return float(np.median(window))
        if name == "Mad":
            return float(np.mean(np.abs(window - np.mean(window))))
        if name == "Skew":
            from scipy.stats import skew  # lazy import
            return float(skew(window))
        if name == "Kurt":
            from scipy.stats import kurtosis  # lazy import
            return float(kurtosis(window))
        raise ValueError(f"Unknown rolling aggregation: {name}")

    # ------------------------------------------------------------------
    # Pair rolling (two series)
    # ------------------------------------------------------------------

    def _eval_pair_rolling(self, node: ExprNode, data: Dict[str, np.ndarray], n: int) -> np.ndarray:
        x = self._eval(node.children[0], data, n)
        y = self._eval(node.children[1], data, n)
        w = node.window
        name = node.name
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(w - 1, n):
            wx = x[i - w + 1: i + 1]
            wy = y[i - w + 1: i + 1]
            if np.any(np.isnan(wx)) or np.any(np.isnan(wy)):
                continue
            if name == "Cov":
                out[i] = float(np.cov(wx, wy, ddof=1)[0, 1])
            elif name == "Corr":
                c = np.corrcoef(wx, wy)
                out[i] = float(c[0, 1])
        return out
