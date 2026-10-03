# Systematic Crypto Trading Bot

A factor-first cryptocurrency trading system with a pluggable expression-tree alpha contract. Research discovers alphas, exports them as JSON, and this system consumes them — no code changes needed.

## Architecture

```
data → features → alpha evaluation → signal normalization → strategy → optimizer → risk → execution
```

### Decision Chain (Typed Contracts)

Every step produces a typed Pydantic object. No raw dicts, no implicit state.

```
OHLCV → FeatureVector → AlphaSpec (JSON) → FactorObservation → FactorSnapshot → StrategyIntent → TradeInstruction → Order
```

| Contract | Responsibility |
|---|---|
| `OHLCV` | Raw market bar (symbol, OHLCV, timestamp) |
| `FeatureVector` | 10 technical features (RSI, EMA, ATR, momentum, ...) |
| `AlphaSpec` | JSON alpha definition — the research ↔ production boundary |
| `FactorObservation` | Single alpha output: bias (BULLISH/BEARISH/NEUTRAL), strength [0,1] |
| `FactorSnapshot` | Aggregated signal across all alphas for one symbol |
| `StrategyIntent` | What to trade and why, before sizing |
| `TradeInstruction` | Sized order ready for execution |

### Alpha Contract

The boundary between research and production. Drop a JSON file into `alphas/`, restart, trade.

```json
{
  "alpha_id": "momentum_impulse",
  "type": "expression_tree",
  "expression": "Mul(Delta($close, 10), Div($volume, Mean($volume, 20)))",
  "normalization": { "method": "rolling_zscore", "lookback": 20 },
  "weight_hint": 0.30
}
```

Expressions use 22 operators ported from [AlphaGen](https://github.com/RL-MLDM/alphagen): rolling (Mean, EMA, Delta, Std, ...), binary (Mul, Div, Add, ...), and unary (Abs, Sign, Log, CSRank). Signal extraction follows AlphaGen's normalize-by-day pattern adapted for crypto.

## Quick Start

```bash
# Install
uv sync

# Paper trade with built-in alphas (no API keys needed)
uv run python main.py

# Paper trade with custom alphas
uv run python main.py --alphas path/to/your/alphas/

# Live trade
uv run python main.py --mode live
```

### Demo (Interview)

`config/demo.yaml` is a trimmed variant of `default.yaml` tuned for live walkthroughs:
3 majors (BTC/ETH/SOL) instead of 60+ symbols, and `paper.speed_multiplier` lowered
from 60× to 20× so the console pace tracks human reading speed. Everything else
(features, alphas, risk, fees) is identical, so the methodology you describe matches
what runs.

```bash
# Recommended launch (auto-activates .venv, tees to logs/trading_paper_<ts>.log)
CONFIG=config/demo.yaml ./scripts/paper_trade.sh

# Or the uv-native form
uv run python main.py --config config/demo.yaml
```

Paper mode uses synthetic GBM candles — no API keys, no network needed.

Approximate on-stage timings:

| Stage | Wall-clock | What to narrate |
|-------|------------|-----------------|
| Startup (load 4 builtin alphas, init monitor/risk/executor) | ~5–10 s | Typed pipeline: `OHLCV → FeatureVector → AlphaSpec → FactorObservation → StrategyIntent → Order` |
| First feature/alpha emissions (rolling indicators warm up) | ~30–60 s | Expression-tree evaluation + IC-weighted composite |
| First `StrategyIntent` / sim order / fill | ~1–3 min | Half-Kelly sizing + 3-layer `RiskShield` (pre-trade, trailing stop, ATR stop) |
| Trailing/ATR stop trigger, P&L drift visible | ~5 min | Circuit breaker (5% daily DD halt + liquidate) |

A 2–3 min run shows the full alpha → intent → order → fill loop; budget ~5 min if
you want to demo risk-side exits as well.

## Module Map

| Module | Responsibility | Key File |
|---|---|---|
| `core/` | Pydantic domain contracts | `models.py` |
| `data/` | WebSocket feed, buffer, resampler | `connector.py`, `buffer.py` |
| `features/` | Stateless numpy feature extraction | `extractor.py` |
| `alpha/` | Expression parser, evaluator, registry | `registry.py`, `expression.py` |
| `strategy/` | State machine, optimizer, sizing | `monitor.py`, `logic.py`, `optimizer.py` |
| `risk/` | Pre-trade validation, stops, circuit breaker | `risk_shield.py`, `tracker.py` |
| `execution/` | Executor abstraction, order lifecycle | `executor.py`, `order_manager.py` |
| `plugins/` | Optional: Roostoo exchange, model inference | `roostoo/`, `model_inference/` |

## Plugin System

Plugins extend the system without polluting the core:

- **Roostoo** (`plugins/roostoo/`): Exchange integration for the Roostoo trading competition
- **Model Inference** (`plugins/model_inference/`): ONNX/PyTorch model-based alpha generation

## Design Decisions

**Why expression-tree over model as the default alpha type?**
Expression-tree alphas are self-contained data — a JSON file IS the alpha. No binary checkpoints, no framework dependency, fully auditable. An RL pipeline (AlphaGen/AlphaQCM) discovers expressions, exports them as JSON, and this system consumes them without code changes.

**Why plugin pattern for exchange integration?**
The core system is exchange-agnostic. Roostoo is a competition-specific executor; isolating it prevents competition code from polluting the trading logic.

**Why restart-to-reload over hot-reload?**
Simplicity and correctness. Alpha JSONs are loaded and validated at startup. No risk of partial updates or inconsistent state mid-trading.

## Hybrid mean reversion and multi horizon forecasts

The research-only hybrid combines the teammate's price mean-reversion strength
with causal multi-horizon model scores into one entry/ranking signal. It preserves
price-only and binary model-filter controls for attribution. Desired holding time
is three to four hours, with earlier risk/recovery exits; decision cadence is an
explicit required parameter, not inferred from the model horizon.

See [the implementation plan](docs/MULTIHORIZON_HYBRID_PLAN.md) and
[the configuration template](config/hybrid.yaml). Offline replay is available via
`python -m scripts.replay_hybrid --config CONFIG --output NEW_JSON_PATH` after
supplying timestamped events, eligible forecasts and required configuration.
No new multi-horizon weights or live inference adapter are included. `main.py`
rejects hybrid configuration instead of falling back to the legacy alpha engine.

Ordinary orders prefer passive limits; urgent risk exits use taker orders only
after conflicting orders are reconciled. Roostoo requests share a rolling budget,
actual fills/commissions drive accounting, and ambiguous orders block replacement.
These code changes have not been deployed or validated against a live account.

## Future Work

- **Portfolio optimizer**: Mean-variance, risk-parity allocation (currently score-tilted softmax)
- **Signal pipeline**: Advanced position optimization beyond Kelly sizing
- **Cross-sectional signals**: Multi-symbol relative value alphas
- **Hot-reload**: Alpha rotation without restart
- **Additional operators**: Custom domain-specific expression operators

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=. --cov-report=term-missing
```

## Acknowledgments

This project is built upon [trading_competition](https://github.com/qunzhongwang/trading_competition) — a system originally developed with my teammate [@qunzhongwang](https://github.com/qunzhongwang) and [@William147WU](https://github.com/William147WU) for the Roostoo trading competition. This fork restructures the architecture around a pluggable alpha contract and expression-tree evaluation pipeline.
