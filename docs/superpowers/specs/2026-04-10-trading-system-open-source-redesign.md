# Trading System Open-Source Redesign

**Date:** 2026-04-10
**Approach:** Surgical Refactor (Approach B)
**Audience:** Quant job seekers / portfolio showcase
**Source:** Fork of [trading_competition](https://github.com/qunzhongwang/trading_competition)

---

## 1. Goal

Transform `trading_competition` from a Roostoo competition entry into a portfolio-showcase crypto trading system for quant recruiters. The repo should demonstrate:

- Clean factor-first architecture with typed contracts
- A standardized alpha contract (JSON) that bridges research and production
- Plugin pattern for exchange integrations and model inference
- Rigorous risk management and audit logging

The fork preserves git history and credits the original collaboration.

**Repo name**: `systematic_crypto_trading_bot`. The directory structure, package imports, and README all use this name.

---

## 2. Core Design Decision: Expression-Tree Alpha Contract

### Why expression_tree as the primary alpha type

Expression-tree alphas are self-contained data. A JSON file IS the alpha — no binary checkpoints, no framework dependency, fully auditable. This aligns with the research-to-production handoff: an RL pipeline (AlphaGen/AlphaQCM) discovers expressions, exports them as JSON, and this system consumes them without code changes.

Model inference is available as a plugin for teams that need sequential pattern recognition (e.g., LSTM on 240-candle sequences), but is not the default path.

### Alpha JSON contract schema

```json
{
  "alpha_id": "momentum_impulse_v2",
  "version": "1.0.0",
  "type": "expression_tree",
  "description": "Momentum burst confirmed by volume spike",

  "expression": "Mul(Delta($close, 5), Div($volume, Mean($volume, 20)))",

  "normalization": {
    "method": "rolling_zscore",
    "lookback": 20
  },

  "validation": {
    "ic": 0.045,
    "icir": 0.62,
    "decay_halflife": 5,
    "backtest_sharpe": 0.41,
    "validated_on": "2026-03-15"
  },

  "weight_hint": 0.14,
  "horizon": "intraday",

  "meta": {
    "author": "alphagen_ppo",
    "source_repo": "alpha-harness"
  }
}
```

### Expression operators (ported from AlphaGen)

| Category | Operators |
|---|---|
| Unary | `Abs`, `Sign`, `Log`, `CSRank` |
| Binary | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Greater`, `Less` |
| Rolling | `Ref`, `Mean`, `Sum`, `Std`, `Var`, `Skew`, `Kurt`, `Max`, `Min`, `Med`, `Mad`, `Rank`, `Delta`, `WMA`, `EMA` |
| Pair Rolling | `Cov`, `Corr` |

Features: `$open`, `$close`, `$high`, `$low`, `$volume`, `$vwap`

---

## 3. Execution Chain: Alpha JSON to Trade

```
1. STARTUP
   Load alphas/*.json
   → Parse each expression string into AST
   → Validate all referenced features ($close, $volume, etc.) exist in extractor
   → Instantiate evaluator per alpha
   → If validation fails, abort with clear error message

2. RUNTIME TICK (per candle close)
   OHLCV candle arrives via WebSocket or SimFeed
        ↓
3. FEATURE EXTRACTION
   extractor.py produces feature arrays (OHLCV + supplementary)
        ↓
4. PER-ALPHA EVALUATION
   expression.evaluate(features) → raw float per symbol
        ↓
5. NORMALIZATION
   rolling_zscore: (value - rolling_mean) / rolling_std over lookback window
   cross_sectional: (value - mean_all_symbols) / std_all_symbols per tick
        ↓
6. SIGNAL EXTRACTION (follows AlphaGen logic)
   z > 0 → BULLISH
   z < 0 → BEARISH
   |z| < noise_threshold (0.3) → NEUTRAL
   strength = clamp(|z| / max_strength_z, 0, 1)
   → produces FactorObservation per alpha per symbol
        ↓
7. AGGREGATION
   Weighted sum across all alphas (weight_hint) → FactorSnapshot
   Same as AlphaGen ensemble: Σ weight_i × alpha_i
        ↓
8. STRATEGY
   FactorSnapshot → StrategyIntent (entry/exit decisions via state machine)
        ↓
9. OPTIMIZER
   StrategyIntent → portfolio allocation across target positions
   MVP: score_tilted (softmax) or equal_weight
   Future: mean-variance, risk-parity
        ↓
10. RISK VALIDATION
    RiskShield gates all orders (exposure, drawdown, rate limits)
         ↓
11. EXECUTION
    TradeInstruction → Order via executor (Binance ccxt / SimExecutor / plugin)
         ↓
12. AUDIT
    JSONL append-only log of every decision in the chain
```

### Normalization methods

| Method | Logic | When to use |
|---|---|---|
| `rolling_zscore` | `(value - rolling_mean) / rolling_std` over lookback window | Default. Per-symbol time-series signal. "Is BTC's momentum unusual vs its own history?" |
| `cross_sectional` | `(value - mean_all_symbols) / std_all_symbols` per tick | Comparing across symbols. "Which symbol has strongest momentum right now?" |

### System-level signal constants (not per-alpha)

| Constant | Value | Meaning |
|---|---|---|
| `noise_threshold` | 0.3 | Below 0.3 sigma = NEUTRAL |
| `max_strength_z` | 3.0 | z=3 maps to strength=1.0 |

---

## 4. Target Directory Structure

```
systematic_crypto_trading_bot/                        # renamed from trading_competition
├── README.md                          # Architecture-first showcase
├── pyproject.toml
├── main.py                            # Composition root, mode selection
│
├── config/
│   ├── default.yaml                   # Core system parameters
│   └── examples/
│       ├── paper_btc_only.yaml
│       └── multi_symbol_live.yaml
│
├── alphas/                            # Alpha JSON contract directory
│   ├── schema.json                    # JSON Schema definition
│   ├── builtin/                       # Pre-packaged expression-tree alphas
│   │   ├── momentum_impulse.json
│   │   ├── trend_alignment.json
│   │   ├── volume_confirmation.json
│   │   └── ...
│   └── examples/
│       └── model_checkpoint_example.json
│
├── src/
│   ├── core/
│   │   └── models.py                  # Pydantic domain contracts
│   ├── data/
│   │   ├── connector.py               # Binance WebSocket
│   │   ├── buffer.py                  # LiveBuffer
│   │   ├── resampler.py               # Multi-timeframe
│   │   └── sim_feed.py                # Paper trading feed
│   ├── features/
│   │   └── extractor.py               # Feature engineering
│   ├── alpha/                         # Alpha contract layer
│   │   ├── contract.py                # AlphaSpec Pydantic model + JSON loader
│   │   ├── registry.py                # Load, validate, manage alphas at startup
│   │   ├── expression.py              # Expression parser + AST evaluator (from AlphaGen)
│   │   └── normalizer.py             # rolling_zscore / cross_sectional
│   ├── strategy/
│   │   ├── monitor.py                 # Orchestrator (main event loop)
│   │   ├── logic.py                   # Per-symbol state machine
│   │   ├── sizing.py                  # Per-position sizing (Kelly, half-Kelly, fixed)
│   │   └── optimizer.py              # Portfolio-level allocation (MVP: score_tilted, equal_weight)
│   ├── risk/
│   │   ├── risk_shield.py             # Pre-trade + runtime validation
│   │   └── tracker.py                 # Portfolio state (NAV, PnL, drawdown)
│   ├── execution/
│   │   ├── base.py                    # BaseExecutor ABC
│   │   ├── executor.py                # Live executor (Binance via ccxt)
│   │   ├── sim_executor.py            # Paper executor
│   │   ├── order_manager.py           # Order lifecycle
│   │   └── trade_logger.py            # JSONL audit logs
│   └── plugins/
│       ├── roostoo/
│       │   ├── executor.py            # RoostooExecutor
│       │   ├── auth.py                # Roostoo API auth
│       │   └── README.md
│       └── model_inference/
│           ├── evaluator.py           # ONNX/PyTorch inference wrapper
│           ├── model_wrapper.py       # Model loading utilities
│           └── README.md
│
├── backtest/
│   ├── runner.py                      # Backtest orchestration
│   └── analysis.py                    # Trade analysis and metrics
│
├── scripts/
│   ├── start.sh                       # Unified launcher
│   └── paper_trade.sh
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
└── docs/
    └── architecture.md
```

---

## 5. What to Remove from Current Repo

| Current file/module | Action | Reason |
|---|---|---|
| `strategy/factor_engine.py` | **Delete** | Replaced by `alpha/` + JSON alphas |
| `strategy/factor_icir_manager.py` | **Delete** | Bayesian weight adaptation — future work |
| `strategy/bayesian_symbol_performance.py` | **Delete** | Over-engineered for showcase |
| `models/inference.py` | **Move** to `plugins/model_inference/` | Plugin |
| `models/lstm_model.py` | **Delete** | Training belongs in research repo |
| `models/transformer_model.py` | **Delete** | Training belongs in research repo |
| `models/train.py` | **Delete** | Training belongs in research repo |
| `models/model_wrapper.py` | **Move** to `plugins/model_inference/` | Plugin |
| `models/icir_tracker.py` | **Delete** | Bayesian tracking — future work |
| `onchain/` (entire directory) | **Delete** | Not core to trading system |
| `data_pipeline/` | **Delete** | On-chain fetching, not core |
| `data/roostoo_auth.py` | **Move** to `plugins/roostoo/` | Plugin isolation |
| `execution/roostoo_executor.py` | **Move** to `plugins/roostoo/` | Plugin isolation |
| `execution/trade_logger.py` | **Keep** | Audit logging is valuable |
| `risk/bayesian_volatility.py` | **Delete** | Basic volatility suffices |
| `backtest/onchain_loader.py` | **Delete** | On-chain not core |
| `backtest/simulated_*.py` | **Consolidate** into `backtest/runner.py` | Simplify |
| `scripts/sweep_*.sh` | **Delete** | Competition-specific |
| `scripts/upload_model_to_hf.py` | **Delete** | No models in core |
| `scripts/train.sh`, `export_model.sh` | **Delete** | Training not in scope |
| `scripts/start_competition.sh` | **Delete** | Use `--plugin roostoo` flag instead |
| `notes/` | **Delete** | Competition notes |
| `artifacts/icir_*.json` | **Delete** | Bayesian priors not needed |

Net effect: ~50 Python modules → ~25 core + 2 plugin modules.

---

## 6. Config

```yaml
# config/default.yaml
mode: paper                    # paper | live

symbols:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT

alphas:
  directory: alphas/builtin
  normalization_default: rolling_zscore
  normalization_lookback: 20
  signal:
    noise_threshold: 0.3
    max_strength_z: 3.0

strategy:
  max_active_positions: 2
  min_entry_score: 0.55
  exit_score_threshold: -0.3
  confirmation_bars: 1

optimizer:
  mode: score_tilted           # equal_weight | score_tilted
  temperature: 0.8
  max_single_weight: 0.35

risk:
  max_portfolio_exposure: 0.18
  max_per_symbol_exposure: 0.05
  trailing_stop_pct: 0.018
  daily_drawdown_limit: 0.025

execution:
  order_type: limit
  limit_offset_bps: 5
  timeout_seconds: 30

paper:
  initial_capital: 100000
  slippage_bps: 5

plugins:
  roostoo:
    enabled: false
  model_inference:
    enabled: false
```

---

## 7. Startup Commands

```bash
# Paper trade (zero config)
uv sync && python main.py

# Paper trade with custom alphas
python main.py --alphas alphas/my_research/

# Live trade
python main.py --mode live --config config/live.yaml

# With Roostoo plugin
python main.py --mode roostoo --plugin roostoo
```

---

## 8. README Structure

```
# [Repo Name]

One-line tagline.

## Architecture

Pipeline diagram (mermaid or text):
  data → features → alpha evaluation → signal normalization
    → strategy → optimizer → risk → execution

### Decision Chain (typed contracts)
OHLCV → FeatureVector → AlphaSpec (JSON) → FactorObservation
  → FactorSnapshot → StrategyIntent → TradeInstruction → Order

### Alpha Contract
The boundary between research and production.
Drop a JSON file into alphas/, restart, trade.
[Show one example JSON]

## Quick Start
Three commands: install, paper trade, see results.

## Module Map
One table: module → responsibility → key file.

## Plugin System
- Roostoo (exchange integration)
- Model Inference (ONNX/PyTorch)

## Design Decisions
- Why expression_tree over model as default
- Why plugin pattern for exchange integration
- Why restart-to-reload over hot-reload

## Future Work
- Signal → sizing pipeline (advanced position optimization)
- Portfolio optimizer (mean-variance, risk-parity)
- Cross-sectional signal refinement
- Hot-reload alpha rotation
- More expression operators

## Acknowledgments

This project is built upon
[trading_competition](https://github.com/qunzhongwang/trading_competition) —
a system originally developed with my teammate
[@qunzhongwang](https://github.com/qunzhongwang) for the Roostoo trading
competition. This fork restructures the architecture around a pluggable
alpha contract and expression-tree evaluation pipeline.
```

---

## 9. Refactor Execution Order

Surgical refactor in PR-sized passes:

1. **Fork & setup** — Fork repo, rename, update metadata
2. **Alpha contract layer** — Add `src/alpha/` (contract, registry, expression evaluator, normalizer)
3. **Plugin system** — Extract Roostoo to `plugins/roostoo/`, move model code to `plugins/model_inference/`
4. **Cleanup** — Delete removed files, consolidate backtest, restructure into `src/`
5. **Strategy refactor** — Replace `factor_engine.py` usage in monitor/logic with alpha registry, extract `sizing.py` and `optimizer.py`
6. **Config simplification** — New `default.yaml`, remove competition-specific params
7. **Tests** — Update tests for new structure, add alpha contract tests
8. **README** — Architecture-first rewrite
9. **Built-in alphas** — Convert current hardcoded factors to expression-tree JSON files

Each pass is a separate commit with a clear message showing engineering discipline.
