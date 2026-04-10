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
