"""Alpha registry: loads alpha JSONs, evaluates per tick, produces FactorSnapshot."""

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

NOISE_THRESHOLD = 0.3
MAX_STRENGTH_Z = 3.0


class _LoadedAlpha:
    def __init__(self, spec: AlphaSpec, ast: ExprNode):
        self.spec = spec
        self.ast = ast


class AlphaRegistry:
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
        self._history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._alphas: Dict[str, _LoadedAlpha] = {}

        if alpha_dir is not None:
            self._load_directory(alpha_dir)

    @property
    def alphas(self) -> Dict[str, AlphaSpec]:
        return {k: v.spec for k, v in self._alphas.items()}

    def _load_directory(self, alpha_dir: Path) -> None:
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
        now = candles[-1].timestamp if candles else datetime.utcnow()

        data = {
            "open":   np.array([c.open for c in candles],   dtype=np.float64),
            "high":   np.array([c.high for c in candles],   dtype=np.float64),
            "low":    np.array([c.low for c in candles],    dtype=np.float64),
            "close":  np.array([c.close for c in candles],  dtype=np.float64),
            "volume": np.array([c.volume for c in candles], dtype=np.float64),
            "vwap":   np.array(
                [(c.high + c.low + c.close) / 3.0 for c in candles],
                dtype=np.float64,
            ),
        }

        observations: List[FactorObservation] = []
        weighted_scores: List[float] = []
        total_weight = 0.0

        for alpha_id, loaded in self._alphas.items():
            spec = loaded.spec
            raw_values = self._evaluator.evaluate(loaded.ast, data)
            current_raw = raw_values[-1] if len(raw_values) > 0 else np.nan

            if not np.isfinite(current_raw):
                observations.append(FactorObservation(
                    name=alpha_id,
                    symbol=symbol,
                    bias=Bias.NEUTRAL,
                    strength=0.0,
                    timestamp=now,
                    thesis=f"{alpha_id}: no signal (NaN)",
                ))
                continue

            self._history[alpha_id][symbol].append(current_raw)
            hist = np.array(self._history[alpha_id][symbol])

            lookback = spec.normalization.lookback
            if spec.normalization.method == NormalizationMethod.ROLLING_ZSCORE:
                key = f"{alpha_id}_{lookback}"
                if key not in self._rolling_normalizers:
                    self._rolling_normalizers[key] = RollingZScoreNormalizer(lookback)
                z = self._rolling_normalizers[key].update(symbol, current_raw, hist)
            else:
                z = current_raw

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

            signed_strength = (
                strength if bias == Bias.BULLISH
                else (-strength if bias == Bias.BEARISH else 0.0)
            )
            weighted_scores.append(signed_strength * spec.weight_hint)
            total_weight += spec.weight_hint

        if total_weight > 0 and weighted_scores:
            entry_score = sum(weighted_scores) / total_weight
        else:
            entry_score = 0.0

        return FactorSnapshot(
            symbol=symbol,
            timestamp=now,
            observations=observations,
            entry_score=entry_score,
            exit_score=entry_score,
            confidence=min(abs(entry_score), 1.0),
        )
