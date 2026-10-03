"""Versioned, causal model-output boundary independent of the model runtime."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from statistics import mean, pstdev

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastHead(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1)
    window_start_seconds: int = Field(ge=0)
    window_end_seconds: int = Field(gt=0)
    target: str = "twap_log_return"
    prediction: float

    @model_validator(mode="after")
    def check_window(self):
        if self.window_start_seconds >= self.window_end_seconds:
            raise ValueError("Label window must have positive length")
        if self.target not in {"twap_log_return", "point_log_return"}:
            raise ValueError("Unsupported prediction units")
        return self


class ForecastPacket(BaseModel):
    """All times are UTC epoch seconds; returns are inverse-scaled log returns."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: int = 1
    symbol: str
    as_of: int = Field(ge=0)
    data_cutoff: int = Field(ge=0)
    available_at: int = Field(ge=0)
    valid_until: int = Field(ge=0)
    eligible_after: int = Field(ge=0)
    model_version: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    heads: list[ForecastHead] = Field(min_length=2)

    @model_validator(mode="after")
    def check_times(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported forecast schema")
        if (
            not self.eligible_after <= self.as_of
            or not self.data_cutoff
            <= self.as_of
            <= self.available_at
            < self.valid_until
        ):
            raise ValueError("Non-causal or expired forecast timestamps")
        if len({h.name for h in self.heads}) != len(self.heads):
            raise ValueError("Duplicate forecast horizon")
        return self


class CausalScores:
    """Time-window moments exclude the current forecast and reject duplicate time."""

    def __init__(
        self,
        window_seconds=86400,
        min_samples=60,
        min_span_seconds=82800,
        max_gap_seconds=600,
    ):
        if (
            not 0 < min_span_seconds <= window_seconds
            or min_samples < 2
            or max_gap_seconds <= 0
        ):
            raise ValueError("Invalid score-history requirements")
        self.window = window_seconds
        self.minimum = min_samples
        self.span = min_span_seconds
        self.gap = max_gap_seconds
        self.history = {}
        self.last = {}
        self.versions = {}

    def update(self, packet: ForecastPacket) -> dict[str, float] | None:
        version = (
            packet.model_version,
            packet.preprocessing_version,
            tuple(
                sorted(
                    (h.name, h.window_start_seconds, h.window_end_seconds, h.target)
                    for h in packet.heads
                )
            ),
        )
        symbol = packet.symbol
        if packet.as_of <= self.last.get(symbol, -1):
            raise ValueError("Forecasts must advance strictly for each asset")
        if (
            self.versions.get(symbol) != version
            or packet.as_of - self.last.get(symbol, packet.as_of) > self.gap
        ):
            self.history[symbol] = {h.name: deque() for h in packet.heads}
        self.versions[symbol] = version
        self.last[symbol] = packet.as_of
        result = {}
        for head in packet.heads:
            history = self.history[symbol][head.name]
            while history and history[0][0] < packet.as_of - self.window:
                history.popleft()
            if (
                len(history) >= self.minimum
                and packet.as_of - history[0][0] >= self.span
            ):
                values = [value for _, value in history]
                std = pstdev(values)
                if std > 1e-12:
                    result[head.name] = (head.prediction - mean(values)) / std
            history.append((packet.as_of, head.prediction))
        return result if len(result) == len(packet.heads) else None


class ForecastBook:
    """Causal admission, per-head normalization and fixed-weight aggregation.

    No fallback to rule-based/random scores when the model output is absent.
    Version and full label definitions must match a predeclared manifest.
    """

    def __init__(self, config: dict):
        self.model_version = config["model_version"]
        self.preprocessing_version = config["preprocessing_version"]
        self.definitions = {h["name"]: h for h in config["heads"]}
        self.weights = {h["name"]: float(h["weight"]) for h in config["heads"]}
        if len(self.definitions) != len(config["heads"]) or len(self.definitions) < 2:
            raise ValueError("At least two unique horizons required")
        if (
            any(not math.isfinite(w) or w < 0 for w in self.weights.values())
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError("Invalid horizon weights")
        self.long_heads = config["long_heads"]
        if not self.long_heads or not set(self.long_heads) <= self.definitions.keys():
            raise ValueError("Long-horizon support heads required")
        self.history = CausalScores(**config.get("normalization", {}))
        self.latest = {}
        self.scores = {}
        self.max_data_age = config.get("max_data_age_seconds", 120)
        if (
            not self.model_version
            or not self.preprocessing_version
            or self.max_data_age <= 0
        ):
            raise ValueError(
                "Artifact versions and positive data freshness are required"
            )
        for spec in self.definitions.values():
            ForecastHead(
                **{
                    key: spec[key]
                    for key in (
                        "name",
                        "window_start_seconds",
                        "window_end_seconds",
                        "target",
                    )
                },
                prediction=0,
            )

    def ingest(self, packet: ForecastPacket, now: int):
        if (
            packet.available_at > now
            or packet.valid_until <= now
            or now - packet.data_cutoff > self.max_data_age
        ):
            raise ValueError("Forecast is unavailable or stale")
        if (
            packet.model_version != self.model_version
            or packet.preprocessing_version != self.preprocessing_version
        ):
            raise ValueError("Unexpected forecast artifact version")
        if {h.name for h in packet.heads} != self.definitions.keys():
            raise ValueError("Missing or unexpected forecast heads")
        for head in packet.heads:
            spec = self.definitions[head.name]
            if any(
                getattr(head, key) != spec[key]
                for key in ("window_start_seconds", "window_end_seconds", "target")
            ):
                raise ValueError("Horizon label definition mismatch")
        scores = self.history.update(packet)
        self.latest[packet.symbol] = packet
        self.scores[packet.symbol] = scores

    def signal(self, symbol: str, now: int):
        packet, scores = self.latest.get(symbol), self.scores.get(symbol)
        if (
            packet is None
            or scores is None
            or packet.available_at > now
            or packet.valid_until <= now
            or now - packet.data_cutoff > self.max_data_age
        ):
            return None
        composite = sum(scores[k] * w for k, w in self.weights.items()) / sum(
            self.weights.values()
        )
        long_support = mean(scores[k] for k in self.long_heads)
        return {
            "composite": composite,
            "long_support": long_support,
            "as_of": packet.as_of,
            "valid_until": packet.valid_until,
            "scores": dict(scores),
        }


def read_forecasts(path: str | Path):
    """Read versioned packets, not pickled model objects or executable artifacts."""
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield ForecastPacket.model_validate(json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid forecast line {line_number}") from exc
