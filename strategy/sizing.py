"""Per-position sizing: target weight -> concrete quantity."""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
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
        if price <= 0 or nav <= 0:
            return 0.0
        effective_weight = target_weight * self._kelly * confidence
        effective_weight = max(self._base, min(self._max, effective_weight))
        notional = nav * effective_weight
        quantity = notional / price
        return max(0.0, quantity)
