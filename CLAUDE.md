# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run paper trading (synthetic data, no API keys needed)
source .venv/bin/activate && python main.py

# Run with custom config or mode override
python main.py --config config/my_config.yaml
python main.py --mode live

# Production runner with auto-restart
./scripts/start.sh

# Run all tests
pytest

# Run a single test file
pytest tests/test_strategy.py

# Run a single test class or method
pytest tests/test_alpha_contract.py::TestAlphaContract
pytest tests/test_expression.py::TestExpressionEvaluator::test_momentum

# Run with coverage
pytest --cov=. --cov-report=term-missing
```

## Architecture

Async event-driven trading bot. All modules connect through a producer-consumer pattern orchestrated by `main.py` via dependency injection.

**Data flow per candle:**
```
Feed → LiveBuffer → [event.set()] → StrategyMonitor → FeatureExtractor → AlphaRegistry → StrategyLogic → RiskShield → OrderManager → Executor
```

**Key async mechanism:** `LiveBuffer` uses `asyncio.Event` — the feed calls `push_candle()` which sets the event, and `StrategyMonitor` blocks on `wait_for_update()`. This is event-driven, not polling.

### Module dependency graph

- **`core/models.py`** — Pydantic schemas imported by every other module. All data flows through these types: `OHLCV`, `Signal` (with `decayed_alpha()` for time decay), `Order`, `Position`, `PortfolioSnapshot`. Enums: `StrategyState` (FLAT/LONG_PENDING/HOLDING).
- **`data/`** — `LiveBuffer` is the central data store (asyncio.Lock-guarded deques) for candles, supplementary data (depth, funding, taker ratio), and resampled higher-TF candles. `WSConnector` (live) and `SimulatedFeed` (paper) are interchangeable candle producers. `BinanceSupplementaryFeed` collects free public Binance data (order book depth WS, mark price WS, taker ratio REST) in parallel. `MultiResampler` resamples 1m candles into multiple timeframes (5m, 15m, 1h).
- **`features/extractor.py`** — Stateless. Pure numpy, no pandas. `extract()` for single snapshot (rule-based), `extract_sequence()` for model input `(seq_len, 10)` array with z-score normalization. 10 features: RSI, EMA fast/slow, ATR, momentum, volatility, order book imbalance, volume ratio, funding rate, taker ratio. Accepts optional `supplementary` dict for live data features.
- **`alpha/`** — The alpha pipeline. `contract.py` defines `AlphaContract` (JSON-validated schema). `expression.py` evaluates declarative expression trees against feature snapshots. `registry.py` loads built-in and user alphas from `alphas/`. `normalizer.py` cross-sectionally normalizes scores. Alpha JSONs live in `alphas/builtin/` and `alphas/examples/`.
- **`strategy/logic.py`** — One `StrategyLogic` instance per symbol. Signal confirmation (N consecutive bars above threshold). State machine transitions driven by alpha signals and order fills. Half-Kelly position sizing via `sizing.py`. Graduated exit tiers. Alpha-based order type: >0.85 → market, otherwise → limit. Alpha decay for stale signals.
- **`strategy/sizing.py`** — Kelly sizing: `base_size_pct: 0.05` (floor) to `max_size_pct: 0.15` (ceiling), `kelly_fraction: 0.5`.
- **`strategy/monitor.py`** — The orchestrator. Runs the full pipeline each time a closed candle arrives. Manages `MultiResampler`, feeds alpha registry with feature snapshots, records trades. Also runs `RiskShield.check_stops()` and circuit breaker checks every iteration.
- **`strategy/optimizer.py`** — Alpha weight optimizer (optional). Uses walk-forward cross-validation to tune per-alpha weights offline.
- **`risk/`** — `PortfolioTracker` is the single source of truth for cash, positions, NAV. `RiskShield` does pre-trade validation and post-trade stop monitoring.
- **`execution/`** — `SimExecutor` and `LiveExecutor` implement the same `BaseExecutor` ABC. `OrderManager` handles lifecycle (including order timeout cancellation) and dispatches fill callbacks to both `PortfolioTracker` and `StrategyLogic`.
- **`plugins/model_inference/`** — Optional plugin. `ModelWrapper` runs ONNX model inference. `evaluator.py` wraps model output as an alpha evaluator compatible with the alpha contract interface. Requires `pip install -e ".[model]"`.
- **`plugins/roostoo/`** — Optional plugin for Roostoo exchange. Requires `pip install -e ".[roostoo]"`.

### Paper vs Live

The system is mode-agnostic after initialization. `main.py` selects the data feed and executor based on `config["mode"]`:
- Paper: `SimulatedFeed` + `SimExecutor` (GBM candles, instant fills with slippage)
- Live: `WSConnector` + `LiveExecutor` (Binance WebSocket, ccxt orders)

### Alpha registry pipeline

Alphas are declared as JSON files conforming to `alphas/schema.json`. The registry loads all JSONs from `alphas/builtin/` at startup, plus any user paths specified in config. Each alpha defines:
- `id`, `name`, `description`
- `expression` — a nested operator tree (arithmetic, comparators, conditionals) evaluated against the feature snapshot
- `weight` — contribution weight in the composite score
- `filters` — optional pre-conditions that gate evaluation

The `ExpressionEvaluator` walks the tree and returns a scalar score in `[-1, 1]`. The `AlphaRegistry` aggregates weighted scores across all active alphas.

### Risk layers (checked every iteration)

1. **Pre-trade** (`RiskShield.validate`): circuit breaker, long-only enforcement, rate limit (60/min), exposure caps (50% portfolio, 15% per symbol), cash check
2. **Trailing stop**: sells if price drops 3% from peak
3. **ATR stop**: sells if price drops 2x ATR below entry
4. **Circuit breaker**: 5% daily drawdown halts all trading and liquidates

### Logging

`main.py` logs to console + rotating files in `logs/`:
- Trading: `logs/trading_{mode}_{timestamp}.log`

### Directory layout

```
config/default.yaml    — all tunable parameters (thresholds, risk limits, fees)
alphas/builtin/        — built-in alpha JSON definitions
alphas/examples/       — example alpha definitions
alphas/schema.json     — JSON schema for alpha contracts
artifacts/             — model checkpoints (.pt, .onnx) — gitignored
logs/                  — rotating log files — gitignored
scripts/               — shell scripts (paper_trade, test, start)
plugins/               — optional plugins (model_inference, roostoo)
data/historical/       — cached CSV data — gitignored
```

## Configuration

All tunable parameters are in `config/default.yaml`. When modifying thresholds or risk limits, change this file — no code changes needed. The config dict is passed to every component constructor.

Key: `alpha.alphas_dir` points to `alphas/builtin/`. Add custom alpha JSONs there or configure additional paths.

## Optional Dependencies

- **Model inference plugin**: `uv pip install -e ".[model]"` (torch, onnxruntime, onnxscript)
- **Roostoo exchange plugin**: `uv pip install -e ".[roostoo]"` (aiohttp)
- **Experiment tracking**: `uv pip install -e ".[dev]"` (wandb)
