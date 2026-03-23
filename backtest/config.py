from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Sequence

from main import _apply_strategy_profile, _validate_config, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def build_runtime_config(
    config_path: Path,
    profile: str | None = None,
    symbols: Sequence[str] | None = None,
    initial_capital: float | None = None,
) -> dict:
    config = copy.deepcopy(load_config(str(config_path)))
    config["mode"] = "paper"

    if profile:
        config.setdefault("strategy", {})["profile"] = profile

    _apply_strategy_profile(config, config.get("strategy", {}).get("profile"))

    if symbols:
        config["symbols"] = list(symbols)

    if initial_capital is not None:
        config.setdefault("paper", {})["initial_capital"] = float(initial_capital)

    # Historical replays should age orders on simulated candle time.
    config.setdefault("execution", {})["order_timeout_seconds"] = 0

    _validate_config(config)
    return config
