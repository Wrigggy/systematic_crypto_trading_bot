"""Signal normalization: raw expression output -> z-score.

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

        Returns 0.0 if insufficient history or zero std.
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

        Returns 0.0 for all if <= 1 symbol or zero std.
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
