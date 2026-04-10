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
    expression: Optional[str] = None
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
    """Check that all features referenced in an expression are available."""
    parser = ExpressionParser()
    try:
        node = parser.parse(expression)
    except ValueError as e:
        return [f"Parse error: {e}"]
    referenced = node.referenced_features()
    missing = referenced - available_features
    return [f"Missing feature: {feat}" for feat in sorted(missing)]
