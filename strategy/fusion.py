"""Interpretable fusion of price dislocation and multi-horizon model signals."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FusedSignal:
    price_z: float
    rule_strength: float
    model_strength: float | None
    joint_strength: float | None
    eligible: bool
    reason: str

    def as_dict(self):
        return asdict(self)


class SignalFusion:
    """Fixed weights are research defaults, not estimated optimal weights.

    Price-only and model-filter policies are explicit ablations. Missing model
    data never silently converts a combined policy into the price-only control.
    """

    def __init__(self, config: dict):
        self.mode = config.get("mode", "combined")
        self.rule_weight = float(config.get("rule_weight", 0.5))
        self.model_weight = float(config.get("model_weight", 0.5))
        self.model_scale = float(config.get("model_score_scale", 3.0))
        self.threshold = float(config.get("entry_threshold", 1.0))
        self.clip = float(config.get("strength_clip", 3.0))
        values = (
            self.rule_weight,
            self.model_weight,
            self.model_scale,
            self.threshold,
            self.clip,
        )
        if self.mode not in {"combined", "model_filter", "price_only"}:
            raise ValueError("Unknown signal fusion mode")
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(
                "Fusion scales, weights and threshold must be positive and finite"
            )

    def evaluate(
        self,
        price: float,
        middle: float,
        std: float,
        entry_sigma: float,
        model: dict | None,
    ) -> FusedSignal:
        if (
            not all(math.isfinite(v) for v in (price, middle, std, entry_sigma))
            or min(price, std, entry_sigma) <= 0
        ):
            raise ValueError("Invalid price signal inputs")
        price_z = (price - middle) / std
        rule_strength = max(-self.clip, min(self.clip, -price_z / entry_sigma))
        candidate = price_z < -entry_sigma
        if self.mode == "price_only":
            return FusedSignal(
                price_z,
                rule_strength,
                None,
                rule_strength,
                candidate,
                "qualified" if candidate else "not_oversold",
            )
        if model is None:
            return FusedSignal(
                price_z, rule_strength, None, None, False, "model_unavailable"
            )
        if not all(math.isfinite(model[k]) for k in ("composite", "long_support")):
            raise ValueError("Invalid model signal inputs")
        model_strength = max(
            -self.clip, min(self.clip, model["composite"] / self.model_scale)
        )
        joint = (
            self.rule_weight * rule_strength + self.model_weight * model_strength
        ) / (self.rule_weight + self.model_weight)
        reason = "qualified"
        if not candidate:
            reason = "not_oversold"
        elif model_strength <= 0 or model["long_support"] < 0:
            reason = "model_disagreement"
        elif self.mode == "model_filter" and model_strength < 1:
            reason = "model_below_filter"
        elif self.mode == "combined" and joint < self.threshold:
            reason = "joint_below_threshold"
        return FusedSignal(
            price_z, rule_strength, model_strength, joint, reason == "qualified", reason
        )
