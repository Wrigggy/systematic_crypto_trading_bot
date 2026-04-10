"""Portfolio-level allocation: FactorSnapshots -> target weights per symbol.

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
        candidates = [(s.symbol, s.entry_score) for s in snapshots if s.entry_score > 0]
        if not candidates:
            return {}

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
        scaled = scores / self._temperature
        scaled -= scaled.max()
        exp_scores = np.exp(scaled)
        weights = exp_scores / exp_scores.sum()

        # Iteratively cap and redistribute excess to uncapped positions
        cap = self._max_single_weight
        for _ in range(20):
            over = weights > cap
            if not np.any(over):
                break
            excess = (weights[over] - cap).sum()
            weights[over] = cap
            under = ~over
            under_sum = weights[under].sum()
            if under_sum > 0:
                weights[under] += excess * (weights[under] / under_sum)
            else:
                break

        # Final normalize to ensure sum = 1
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return {sym: float(w) for (sym, _), w in zip(selected, weights)}
