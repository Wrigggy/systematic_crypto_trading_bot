from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

import numpy as np

from core.models import FactorBias, OHLCV, Order, OrderStatus, Side, StrategyState
from data.buffer import LiveBuffer
from data.resampler import CandleResampler, MultiResampler
from execution.order_manager import OrderManager
from execution.trade_logger import TradeLogger
from features.extractor import FeatureExtractor
from models.inference import AlphaEngine
from risk.bayesian_volatility import BayesianVolatilityManager
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from strategy.bayesian_symbol_performance import BayesianSymbolPerformanceManager
from strategy.factor_engine import FactorEngine
from strategy.factor_icir_manager import BayesianFactorICIRManager
from strategy.logic import StrategyLogic

if TYPE_CHECKING:
    from execution.base import BaseExecutor

logger = logging.getLogger(__name__)


class StrategyMonitor:
    """Central event loop / orchestrator.

    Consumes candles from the buffer, runs the feature → alpha → strategy → risk → execution pipeline.
    Event-driven: blocks on buffer.wait_for_update(), triggered by new closed candles.
    """

    def __init__(
        self,
        config: dict,
        buffer: LiveBuffer,
        extractor: FeatureExtractor,
        alpha_engine: AlphaEngine,
        risk_shield: RiskShield,
        tracker: PortfolioTracker,
        order_manager: OrderManager,
        factor_engine: Optional[FactorEngine] = None,
        resampler: Optional[CandleResampler] = None,
        multi_resampler: Optional[MultiResampler] = None,
        trade_tracker=None,
        icir_tracker=None,
        executor: Optional["BaseExecutor"] = None,
        trade_logger: Optional[TradeLogger] = None,
    ):
        self._config = config
        self._buffer = buffer
        self._extractor = extractor
        self._factor_engine = factor_engine or FactorEngine(config)
        self._alpha_engine = alpha_engine
        self._risk_shield = risk_shield
        self._tracker = tracker
        self._order_manager = order_manager
        self._running = False
        self._resampler = resampler
        self._multi_resampler = multi_resampler
        self._trade_tracker = trade_tracker
        self._icir_tracker = icir_tracker
        self._executor = executor
        self._trade_logger = trade_logger
        self._volatility_manager = BayesianVolatilityManager(config)
        self._performance_manager = BayesianSymbolPerformanceManager(config)
        self._factor_icir_manager = BayesianFactorICIRManager(config)

        # Multi-TF timeframes to fetch for alpha filter
        alpha_cfg = config.get("alpha", {})
        self._multi_timeframes = alpha_cfg.get("multi_timeframes", [])
        self._use_model_overlay = config.get("strategy", {}).get(
            "use_model_overlay", False
        )
        strategy_cfg = config.get("strategy", {})
        risk_cfg = config.get("risk", {})
        default_slot_count = max(len(config.get("symbols", [])), 1)
        self._core_symbols: Set[str] = set(strategy_cfg.get("core_symbols", []))
        self._satellite_symbols: Set[str] = set(
            strategy_cfg.get("satellite_symbols", [])
        )
        self._allow_satellite_in_neutral: bool = strategy_cfg.get(
            "allow_satellite_in_neutral", True
        )
        self._min_entry_score: float = float(
            strategy_cfg.get("min_entry_score", 0.0)
        )
        self._top_n_entries_per_cycle: int = max(
            1, int(strategy_cfg.get("top_n_entries_per_cycle", default_slot_count))
        )
        self._max_active_positions: int = max(
            1, int(strategy_cfg.get("max_active_positions", default_slot_count))
        )
        self._satellite_max_active_positions: int = max(
            0,
            int(
                strategy_cfg.get(
                    "satellite_max_active_positions", len(self._satellite_symbols)
                )
            ),
        )
        self._satellite_min_entry_score_bonus: float = float(
            strategy_cfg.get("satellite_min_entry_score_bonus", 0.0)
        )
        self._core_priority_bonus: float = float(
            strategy_cfg.get("core_priority_bonus", 0.0)
        )
        self._portfolio_targeting_enabled: bool = bool(
            strategy_cfg.get("portfolio_targeting_enabled", True)
        )
        self._portfolio_rank_weight_power: float = float(
            strategy_cfg.get("portfolio_rank_weight_power", 1.0)
        )
        self._portfolio_max_rank_size_multiplier: float = float(
            strategy_cfg.get("portfolio_max_rank_size_multiplier", 1.0)
        )
        self._relative_strength_enabled: bool = bool(
            strategy_cfg.get("relative_strength_enabled", True)
        )
        self._relative_strength_lookback_bars: int = max(
            5, int(strategy_cfg.get("relative_strength_lookback_bars", 36))
        )
        self._relative_strength_vol_window: int = max(
            self._relative_strength_lookback_bars + 1,
            int(strategy_cfg.get("relative_strength_vol_window", 72)),
        )
        self._relative_strength_vol_percentile: float = float(
            strategy_cfg.get("relative_strength_vol_percentile", 0.80)
        )
        self._relative_strength_weight: float = float(
            strategy_cfg.get("relative_strength_weight", 0.10)
        )
        self._relative_strength_weak_penalty: float = float(
            strategy_cfg.get("relative_strength_weak_penalty", 0.08)
        )
        self._relative_strength_min_score: float = float(
            strategy_cfg.get("relative_strength_min_score", -0.25)
        )
        self._volume_weighted_momentum_enabled: bool = bool(
            strategy_cfg.get("volume_weighted_momentum_enabled", False)
        )
        self._volume_momentum_short_window: int = max(
            2, int(strategy_cfg.get("volume_momentum_short_window", 6))
        )
        self._volume_momentum_long_window: int = max(
            self._volume_momentum_short_window + 1,
            int(strategy_cfg.get("volume_momentum_long_window", 24)),
        )
        self._volume_ratio_cap: float = max(
            1.0, float(strategy_cfg.get("volume_ratio_cap", 2.5))
        )
        self._volume_weighted_momentum_weight: float = float(
            strategy_cfg.get("volume_weighted_momentum_weight", 0.0)
        )
        self._volume_weighted_momentum_min_score: float = float(
            strategy_cfg.get("volume_weighted_momentum_min_score", -999.0)
        )
        self._candidate_cohort_gate_enabled: bool = bool(
            strategy_cfg.get("candidate_cohort_gate_enabled", False)
        )
        self._no_trade_min_candidate_score: float = float(
            strategy_cfg.get("no_trade_min_candidate_score", 0.0)
        )
        self._no_trade_min_score_margin: float = float(
            strategy_cfg.get("no_trade_min_score_margin", 0.0)
        )
        self._symbol_performance_enabled: bool = bool(
            strategy_cfg.get("symbol_performance_enabled", True)
        )
        self._symbol_performance_weight: float = float(
            strategy_cfg.get("symbol_performance_weight", 0.18)
        )
        self._core_underperformance_penalty: float = float(
            strategy_cfg.get("core_underperformance_penalty", 0.03)
        )
        self._satellite_underperformance_penalty: float = float(
            strategy_cfg.get("satellite_underperformance_penalty", 0.05)
        )
        self._core_underperformance_threshold: float = float(
            strategy_cfg.get("core_underperformance_threshold", -0.01)
        )
        self._satellite_underperformance_threshold: float = float(
            strategy_cfg.get("satellite_underperformance_threshold", 0.00)
        )
        self._max_portfolio_exposure_ratio: float = float(
            risk_cfg.get("max_portfolio_exposure", 1.0)
        )
        self._max_single_exposure_ratio: float = float(
            risk_cfg.get("max_single_exposure", 1.0)
        )
        self._primary_minutes = 1
        if multi_resampler is not None:
            self._primary_minutes = multi_resampler.primary_minutes
        elif resampler is not None:
            self._primary_minutes = resampler.minutes

        # Per-symbol strategy state machines
        symbols = config.get("symbols", [])
        self._strategies: Dict[str, StrategyLogic] = {
            sym: StrategyLogic(sym, config) for sym in symbols
        }

        # Inject trade tracker into each strategy for adaptive Kelly
        if trade_tracker is not None:
            for strategy in self._strategies.values():
                strategy.set_trade_tracker(trade_tracker)
        for strategy in self._strategies.values():
            strategy.set_volatility_manager(self._volatility_manager)
            strategy.set_performance_manager(self._performance_manager)
        self._risk_shield.set_volatility_manager(self._volatility_manager)
        if hasattr(self._factor_engine, "set_performance_manager"):
            self._factor_engine.set_performance_manager(self._performance_manager)
        if hasattr(self._factor_engine, "set_factor_icir_manager"):
            self._factor_engine.set_factor_icir_manager(self._factor_icir_manager)

        # Register fill callback for strategy logic
        self._order_manager.register_fill_callback(self._on_order_event)

        # Min candles before we start trading (engine-aware)
        engine_type = config.get("alpha", {}).get("engine", "rule_based")
        seq_len = config.get("alpha", {}).get("seq_len", 30)
        if self._use_model_overlay and engine_type in ("lstm", "transformer", "ensemble"):
            self._warmup_candles = (extractor.min_candles + seq_len) * self._primary_minutes
        else:
            self._warmup_candles = extractor.min_candles * self._primary_minutes

        # Day boundary tracking for circuit breaker / tracker daily reset
        self._last_trading_date: Optional[str] = None

        # Time-based periodic logging (wall clock, not iteration count)
        self._last_status_log: float = 0.0
        self._status_log_interval: float = 60.0  # 1 minute

        # ICIR tracking: store previous factors per symbol for online learning
        self._prev_factors: Dict[str, list] = {}
        self._prev_prices: Dict[str, float] = {}
        self._prev_factor_snapshot_scores: Dict[str, Dict[str, float]] = {}
        self._latest_market_context: Optional[dict] = None

    async def run(self) -> None:
        """Main event loop."""
        self._running = True
        logger.info("StrategyMonitor started, warmup=%d candles", self._warmup_candles)

        iteration = 0
        while self._running:
            got_update = await self._buffer.wait_for_update(timeout=5.0)
            if not got_update:
                continue

            iteration += 1
            try:
                await self._process_iteration(iteration)
            except Exception:
                logger.exception("Error in iteration %d, continuing", iteration)

    async def stop(self) -> None:
        self._running = False
        logger.info("StrategyMonitor stopped")

    @property
    def strategies(self) -> Dict[str, StrategyLogic]:
        return self._strategies

    async def _process_iteration(self, iteration: int) -> None:
        """Process one iteration: for each symbol, run the full pipeline.

        Price updates and stop checks run every 1-min candle.
        Alpha scoring and trade decisions only run when the resampler emits
        a completed N-min bar (or every candle if no resampler).
        """
        # Day boundary detection — reset circuit breaker and daily PnL at midnight UTC
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_trading_date is not None and today != self._last_trading_date:
            logger.info("Day boundary crossed: %s → %s, resetting daily state", self._last_trading_date, today)
            self._risk_shield.reset_daily()
            self._tracker.reset_daily()
        self._last_trading_date = today

        # Check pending limit orders for fills
        await self._order_manager.check_pending()

        snapshot = self._tracker.snapshot()
        latest_candles: Dict[str, OHLCV] = {}
        atr_values: Dict[str, float] = {}
        symbol_state: Dict[str, Dict[str, Any]] = {}

        # Track which symbols have a completed resampled bar this iteration
        alpha_ready: Set[str] = set()

        for symbol, strategy in self._strategies.items():
            candles = await self._buffer.get_candles(symbol)
            if not candles:
                continue

            latest_candles[symbol] = candles[-1]

            # Update position prices (every 1-min candle)
            self._tracker.update_prices(symbol, candles[-1].close)

            # Gate alpha on resampled bar completion
            if self._multi_resampler is not None:
                resampled_bars = self._multi_resampler.push(candles[-1])
                # Store higher-TF completed bars in the buffer
                for period, bar in resampled_bars.items():
                    if bar is not None:
                        await self._buffer.push_resampled(period, bar)
                # Alpha gating on primary (smallest) period
                primary = self._multi_resampler.primary_minutes
                if resampled_bars.get(primary) is not None:
                    alpha_ready.add(symbol)
            elif self._resampler is not None:
                resampled = self._resampler.push(candles[-1])
                if resampled is not None:
                    await self._buffer.push_resampled(self._resampler.minutes, resampled)
                    alpha_ready.add(symbol)
            else:
                alpha_ready.add(symbol)

            # Check warmup
            if len(candles) < self._warmup_candles:
                if iteration % 50 == 1:
                    logger.info(
                        "[%s] warming up: %d/%d candles",
                        symbol,
                        len(candles),
                        self._warmup_candles,
                    )
                continue

            # ── Feature Extraction (always, for ATR stops) ──
            supplementary = await self._buffer.get_supplementary(symbol)
            strategy_candles = candles
            if self._primary_minutes > 1:
                resampled_candles = await self._buffer.get_resampled_candles(
                    symbol, self._primary_minutes, n=max(self._extractor.min_candles + 10, 60)
                )
                if resampled_candles:
                    strategy_candles = resampled_candles

            if len(strategy_candles) < self._extractor.min_candles:
                continue

            features = self._extractor.extract(
                strategy_candles, supplementary=supplementary
            )
            self._volatility_manager.update(symbol, features.volatility)
            atr_values[symbol] = features.atr

            # Fetch higher-TF candles once so the same context is available to
            # both the market-regime builder and the per-symbol factor pipeline.
            candles_15m = None
            candles_1h = None
            if self._multi_timeframes:
                if 15 in self._multi_timeframes:
                    candles_15m = await self._buffer.get_resampled_candles(
                        symbol, 15, n=50
                    )
                if 60 in self._multi_timeframes:
                    candles_1h = await self._buffer.get_resampled_candles(
                        symbol, 60, n=50
                    )

            symbol_state[symbol] = {
                "candles": candles,
                "strategy_candles": strategy_candles,
                "supplementary": supplementary,
                "features": features,
                "candles_15m": candles_15m,
                "candles_1h": candles_1h,
            }

            # ── Strategy Gating (only on completed resampled bars) ──
            if symbol not in alpha_ready:
                continue

        market_context = self._build_market_context(symbol_state)
        self._latest_market_context = market_context
        buy_candidates: list[dict[str, Any]] = []

        for symbol, strategy in self._strategies.items():
            state = symbol_state.get(symbol)
            if state is None or symbol not in alpha_ready:
                continue

            candles = state["candles"]
            strategy_candles = state["strategy_candles"]
            supplementary = state["supplementary"]
            features = state["features"]
            candles_15m = state["candles_15m"]
            candles_1h = state["candles_1h"]

            self._record_online_icir(symbol, candles[-1].close)
            self._store_rule_icir_inputs(symbol, features, candles[-1].close)

            factor_history_window = getattr(
                self._factor_engine, "supplementary_history_window", 2
            )
            supplementary_history_len = max(
                self._alpha_engine._seq_len,
                factor_history_window,
            )
            if self._use_model_overlay and self._primary_minutes > 1:
                supplementary_history_len *= self._primary_minutes
            supplementary_history = await self._buffer.get_supplementary_history(
                symbol, supplementary_history_len
            )
            model_supplementary_history = self._align_supplementary_history(
                supplementary_history,
                seq_len=self._alpha_engine._seq_len,
            )

            # Fetch higher-TF candles for multi-TF filter
            candles_15m = None
            candles_1h = None
            if self._multi_timeframes:
                if 15 in self._multi_timeframes:
                    candles_15m = await self._buffer.get_resampled_candles(
                        symbol, 15, n=50
                    )
                if 60 in self._multi_timeframes:
                    candles_1h = await self._buffer.get_resampled_candles(
                        symbol, 60, n=50
                    )

            factor_snapshot = self._factor_engine.evaluate(
                features,
                supplementary=supplementary,
                supplementary_history=supplementary_history,
                candles_15m=candles_15m,
                candles_1h=candles_1h,
                market_context=market_context,
            )
            self._store_factor_snapshot_history(symbol, factor_snapshot)
            model_signal = None
            if self._use_model_overlay:
                model_signal = self._alpha_engine.score(
                    strategy_candles,
                    supplementary=supplementary,
                    supplementary_history=model_supplementary_history,
                    candles_15m=candles_15m,
                    candles_1h=candles_1h,
                )

            if self._trade_logger is not None:
                await self._trade_logger.log_factor_snapshot(
                    symbol=factor_snapshot.symbol,
                    regime=factor_snapshot.regime,
                    entry_score=factor_snapshot.entry_score,
                    blocker_score=factor_snapshot.blocker_score,
                    confidence=factor_snapshot.confidence,
                    observations=[
                        obs.model_dump(mode="json")
                        for obs in factor_snapshot.observations
                    ],
                    summary=factor_snapshot.summary,
                )

            # ── Strategy Intent → Instruction ──
            snapshot = self._tracker.snapshot()
            intent = strategy.on_factors(
                factor_snapshot,
                snapshot,
                current_price=candles[-1].close,
                model_signal=model_signal,
            )

            if intent is not None:
                if self._trade_logger is not None:
                    await self._trade_logger.log_strategy_intent(
                        intent.model_dump(mode="json")
                    )

                if intent.direction == Side.BUY:
                    buy_candidates.append(
                        {
                            "symbol": symbol,
                            "strategy": strategy,
                            "intent": intent,
                            "current_price": candles[-1].close,
                            "factor_snapshot": factor_snapshot,
                            "strategy_candles": strategy_candles,
                        }
                    )
                    continue

                instruction = strategy.build_instruction(
                    intent, current_price=candles[-1].close
                )
                if self._trade_logger is not None:
                    await self._trade_logger.log_trade_instruction(
                        instruction.model_dump(mode="json")
                    )

                order = instruction.to_order()
                # ── Risk Validation ──
                validated = self._risk_shield.validate(
                    order,
                    self._tracker,
                    market_price=candles[-1].close,
                )
                if validated is not None:
                    if self._has_pending_side_order(validated.symbol, Side.SELL):
                        logger.info(
                            "[%s] skipping duplicate sell submission because an exit order is already pending",
                            validated.symbol,
                        )
                    else:
                        await self._order_manager.submit(validated)
                else:
                    # Risk rejected the order — reset strategy state
                    strategy.on_cancel(order)

        # ── Post-trade Risk Checks ──
        ranked_buy_candidates = self._rank_buy_candidates(buy_candidates)
        selected_buy_candidates = ranked_buy_candidates[: self._top_n_entries_per_cycle]
        skipped_buy_candidates = ranked_buy_candidates[self._top_n_entries_per_cycle :]
        for candidate in skipped_buy_candidates:
            self._cancel_buy_candidate(
                candidate,
                "not in top entry cohort for this cycle",
            )

        for idx, candidate in enumerate(selected_buy_candidates):
            remaining_candidates = selected_buy_candidates[idx:]
            unsubmitted_symbols = {item["symbol"] for item in remaining_candidates}
            if (
                self._active_symbol_count(exclude_symbols=unsubmitted_symbols)
                >= self._max_active_positions
            ):
                self._cancel_buy_candidate(
                    candidate,
                    "max active position cap reached",
                )
                continue

            if (
                candidate["symbol"] in self._satellite_symbols
                and self._active_satellite_count(exclude_symbols=unsubmitted_symbols)
                >= self._satellite_max_active_positions
            ):
                self._cancel_buy_candidate(
                    candidate,
                    "satellite position cap reached",
                )
                continue

            strategy = candidate["strategy"]
            intent = candidate["intent"]
            current_price = candidate["current_price"]
            instruction = strategy.build_instruction(
                intent,
                current_price=current_price,
            )
            instruction = self._apply_portfolio_entry_targeting(
                candidate,
                instruction,
                remaining_candidates,
            )
            if instruction.quantity <= 0:
                self._cancel_buy_candidate(
                    candidate,
                    "portfolio target sizing reduced the entry below minimum tradable size",
                )
                continue
            if self._trade_logger is not None:
                await self._trade_logger.log_trade_instruction(
                    instruction.model_dump(mode="json")
                )

            order = instruction.to_order()
            validated = self._risk_shield.validate(
                order,
                self._tracker,
                market_price=current_price,
            )
            if validated is not None:
                if self._has_pending_side_order(validated.symbol, Side.SELL):
                    self._cancel_buy_candidate(
                        candidate,
                        "exit order already pending for this symbol",
                    )
                    continue
                await self._order_manager.submit(validated)
            else:
                strategy.on_cancel(order)

        # Trailing stops and ATR stops
        stop_orders = self._risk_shield.check_stops(
            self._tracker, latest_candles, atr_values
        )
        for stop_order in stop_orders:
            if self._has_pending_side_order(stop_order.symbol, Side.SELL):
                self._mark_strategy_exit_pending(stop_order.symbol)
                logger.info(
                    "[%s] stop already armed via pending sell order, skipping duplicate stop submission",
                    stop_order.symbol,
                )
                continue
            validated = self._risk_shield.validate(
                stop_order,
                self._tracker,
                market_price=latest_candles.get(stop_order.symbol).close
                if latest_candles.get(stop_order.symbol) is not None
                else 0.0,
                is_stop=True,
            )
            if validated is not None:
                await self._order_manager.submit(validated)
                self._mark_strategy_exit_pending(stop_order.symbol)

        # Circuit breaker check
        if self._risk_shield.check_circuit_breaker(self._tracker):
            await self._liquidate_all()

        # Periodic logging (time-based, not iteration-based)
        now = time.monotonic()
        if now - self._last_status_log >= self._status_log_interval:
            self._last_status_log = now
            snap = self._tracker.snapshot()
            holdings = [
                f"{p.symbol}:{self._format_quantity(p.quantity)}@{p.current_price:.2f}"
                for p in snap.positions
                if p.state == StrategyState.HOLDING
            ]
            logger.info(
                "Iter %d | NAV=%.2f | Cash=%.2f | PnL=%.2f | DD=%.2f%% | %s | Holdings=%s",
                iteration,
                snap.nav,
                snap.cash,
                snap.daily_pnl,
                snap.drawdown * 100,
                self._format_regime_status(self._latest_market_context),
                holdings or "none",
            )

            # Roostoo mode: fetch live balance to verify API connectivity
            if self._executor is not None and hasattr(self._executor, "get_balance"):
                try:
                    roostoo_bal = await self._executor.get_balance()
                    if roostoo_bal:
                        bal_parts = [
                            self._format_balance_item(asset, value)
                            for asset, value in roostoo_bal.items()
                        ]
                        logger.info("Roostoo balance | %s", " | ".join(bal_parts))
                    else:
                        logger.warning("Roostoo balance fetch returned empty")
                except Exception:
                    logger.warning("Roostoo balance fetch failed", exc_info=True)

    def _align_supplementary_history(
        self,
        history: dict,
        seq_len: int,
    ) -> dict:
        """Align raw per-minute supplementary history to the strategy timeframe."""
        if self._primary_minutes <= 1:
            return {
                key: list(values)[-seq_len:]
                for key, values in history.items()
            }

        aligned = {}
        for key, values in history.items():
            series = list(values)
            if not series:
                aligned[key] = []
                continue
            sampled = []
            idx = len(series) - 1
            while idx >= 0 and len(sampled) < seq_len:
                sampled.append(series[idx])
                idx -= self._primary_minutes
            sampled.reverse()
            aligned[key] = sampled
        return aligned

    def _rank_buy_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self._annotate_buy_candidates(candidates)
        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            reason = self._buy_candidate_rejection_reason(candidate)
            if reason is not None:
                self._cancel_buy_candidate(candidate, reason)
                continue
            ranked.append(candidate)
        ranked.sort(key=self._candidate_priority_score, reverse=True)
        ranked = self._apply_candidate_cohort_gate(ranked)
        return ranked

    def _buy_candidate_rejection_reason(
        self, candidate: dict[str, Any]
    ) -> Optional[str]:
        symbol = candidate["symbol"]
        snapshot = candidate["factor_snapshot"]
        regime = snapshot.regime
        if regime == "risk_off":
            return "market regime is risk_off"
        if symbol in self._satellite_symbols and regime == "neutral":
            if not self._allow_satellite_in_neutral:
                return "satellite entries disabled outside risk_on"
        if (
            symbol in self._satellite_symbols
            and snapshot.entry_score
            < (self._min_entry_score + self._satellite_min_entry_score_bonus)
        ):
            return "satellite entry score below stricter threshold"
        metrics = candidate.get("selection_metrics", {})
        if self._relative_strength_enabled:
            if not metrics.get("passes_vol_filter", True):
                return "vol-adjusted momentum filter blocked entry"
            if metrics.get("vol_adjusted_momentum", 0.0) < self._relative_strength_min_score:
                return "relative strength score too weak"
        if self._volume_weighted_momentum_enabled and (
            metrics.get("volume_weighted_momentum", 0.0)
            < self._volume_weighted_momentum_min_score
        ):
            return "volume-weighted momentum score too weak"
        performance_penalty = self._dynamic_entry_penalty(symbol)
        if performance_penalty > 0 and snapshot.entry_score < (
            self._min_entry_score + performance_penalty
        ):
            return "symbol underperformance raised the entry threshold"
        return None

    def _candidate_priority_score(self, candidate: dict[str, Any]) -> float:
        snapshot = candidate["factor_snapshot"]
        symbol = candidate["symbol"]
        score = snapshot.entry_score - 0.5 * snapshot.blocker_score
        if symbol in self._core_symbols:
            score += self._core_priority_bonus
        metrics = candidate.get("selection_metrics", {})
        if self._relative_strength_enabled:
            score += self._relative_strength_weight * metrics.get(
                "relative_strength_zscore", 0.0
            )
            if metrics.get("vol_adjusted_momentum", 0.0) < 0:
                score -= self._relative_strength_weak_penalty
        if self._volume_weighted_momentum_enabled:
            score += self._volume_weighted_momentum_weight * metrics.get(
                "volume_weighted_momentum_signal", 0.0
            )
        if self._symbol_performance_enabled:
            score += self._symbol_performance_weight * self._symbol_performance_score(symbol)
        return score

    def _annotate_buy_candidates(self, candidates: list[dict[str, Any]]) -> None:
        if not candidates:
            return

        momentum_values = []
        for candidate in candidates:
            metrics = self._compute_selection_metrics(
                candidate.get("strategy_candles", [])
            )
            candidate["selection_metrics"] = metrics
            if self._relative_strength_enabled:
                momentum_values.append(metrics["vol_adjusted_momentum"])

        if not self._relative_strength_enabled or not momentum_values:
            return

        mean = float(np.mean(momentum_values))
        std = float(np.std(momentum_values))
        for candidate in candidates:
            momentum = candidate["selection_metrics"]["vol_adjusted_momentum"]
            zscore = (momentum - mean) / std if std > 1e-10 else 0.0
            candidate["selection_metrics"]["relative_strength_zscore"] = zscore

    def _compute_selection_metrics(self, strategy_candles: list[OHLCV]) -> dict[str, float | bool]:
        if not (
            self._relative_strength_enabled or self._volume_weighted_momentum_enabled
        ):
            return {
                "vol_adjusted_momentum": 0.0,
                "current_volatility": 0.0,
                "volatility_threshold": 0.0,
                "passes_vol_filter": True,
                "volume_ratio": 1.0,
                "volume_weighted_momentum": 0.0,
                "volume_weighted_momentum_signal": 0.0,
                "relative_strength_zscore": 0.0,
            }

        closes = np.array([c.close for c in strategy_candles], dtype=np.float64)
        volumes = np.array([max(c.volume, 0.0) for c in strategy_candles], dtype=np.float64)
        required = max(
            self._relative_strength_lookback_bars + 1,
            self._relative_strength_vol_window + 1,
            self._volume_momentum_long_window
            if self._volume_weighted_momentum_enabled
            else 0,
        )
        if len(closes) < required:
            return {
                "vol_adjusted_momentum": 0.0,
                "current_volatility": 0.0,
                "volatility_threshold": 0.0,
                "passes_vol_filter": True,
                "volume_ratio": 1.0,
                "volume_weighted_momentum": 0.0,
                "volume_weighted_momentum_signal": 0.0,
                "relative_strength_zscore": 0.0,
            }

        log_returns = np.diff(np.log(np.clip(closes, 1e-12, None)))
        momentum = float(np.sum(log_returns[-self._relative_strength_lookback_bars :]))
        current_vol = float(np.std(log_returns[-self._relative_strength_vol_window :]))

        rolling_vols = []
        for idx in range(self._relative_strength_vol_window, len(log_returns) + 1):
            window = log_returns[idx - self._relative_strength_vol_window : idx]
            rolling_vols.append(float(np.std(window)))
        vol_threshold = float(
            np.quantile(rolling_vols, self._relative_strength_vol_percentile)
        ) if rolling_vols else current_vol
        passes_vol_filter = current_vol <= max(vol_threshold, 1e-12)
        vol_adjusted_momentum = momentum / max(current_vol, 1e-6)

        short_volume = float(np.mean(volumes[-self._volume_momentum_short_window :]))
        long_volume = float(np.mean(volumes[-self._volume_momentum_long_window :]))
        volume_ratio = short_volume / max(long_volume, 1e-6)
        capped_volume_ratio = float(
            np.clip(
                volume_ratio,
                1.0 / self._volume_ratio_cap,
                self._volume_ratio_cap,
            )
        )
        volume_weighted_momentum = vol_adjusted_momentum * capped_volume_ratio

        return {
            "vol_adjusted_momentum": float(vol_adjusted_momentum),
            "current_volatility": current_vol,
            "volatility_threshold": vol_threshold,
            "passes_vol_filter": passes_vol_filter,
            "volume_ratio": float(volume_ratio),
            "volume_weighted_momentum": float(volume_weighted_momentum),
            "volume_weighted_momentum_signal": float(
                np.tanh(volume_weighted_momentum / 3.0)
            ),
            "relative_strength_zscore": 0.0,
        }

    def _apply_candidate_cohort_gate(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not candidates or not self._candidate_cohort_gate_enabled:
            return candidates

        top_score = self._candidate_priority_score(candidates[0])
        if top_score < self._no_trade_min_candidate_score:
            self._cancel_candidate_batch(
                candidates,
                "top candidate composite score below no-trade floor",
            )
            return []

        if len(candidates) > 1:
            second_score = self._candidate_priority_score(candidates[1])
            if (top_score - second_score) < self._no_trade_min_score_margin:
                self._cancel_candidate_batch(
                    candidates,
                    "top candidate edge versus next alternative too small",
                )
                return []

        return candidates

    def _candidate_rank_weights(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, float]:
        if not candidates:
            return {}

        adjusted_scores: dict[str, float] = {}
        for candidate in candidates:
            score = max(self._candidate_priority_score(candidate), 1e-6)
            adjusted_scores[candidate["symbol"]] = score ** max(
                self._portfolio_rank_weight_power, 1e-6
            )

        total = sum(adjusted_scores.values())
        if total <= 1e-12:
            equal_weight = 1.0 / len(candidates)
            return {candidate["symbol"]: equal_weight for candidate in candidates}
        return {
            symbol: score / total for symbol, score in adjusted_scores.items()
        }

    def _apply_portfolio_entry_targeting(
        self,
        candidate: dict[str, Any],
        instruction,
        remaining_candidates: list[dict[str, Any]],
    ):
        if (
            not self._portfolio_targeting_enabled
            or instruction.direction != Side.BUY
            or instruction.quantity <= 0
        ):
            return instruction

        current_price = candidate["current_price"]
        if current_price <= 0:
            return instruction

        symbol = candidate["symbol"]
        snapshot = self._tracker.snapshot()
        current_total_exposure = self._tracker.get_total_exposure()
        remaining_portfolio_ratio = max(
            0.0, self._max_portfolio_exposure_ratio - current_total_exposure
        )
        current_symbol_exposure = self._tracker.get_exposure(symbol)
        single_headroom_ratio = max(
            0.0, self._max_single_exposure_ratio - current_symbol_exposure
        )
        if remaining_portfolio_ratio <= 0 or single_headroom_ratio <= 0:
            instruction.quantity = 0.0
            instruction.size_notional = 0.0
            instruction.size_pct = 0.0
            return instruction

        rank_weights = self._candidate_rank_weights(remaining_candidates)
        target_notional = snapshot.nav * remaining_portfolio_ratio * rank_weights.get(
            symbol, 1.0
        )
        single_headroom_notional = snapshot.nav * single_headroom_ratio
        boosted_baseline_notional = (
            instruction.size_notional * self._portfolio_max_rank_size_multiplier
        )
        final_notional = min(
            snapshot.cash * 0.99,
            single_headroom_notional,
            target_notional,
            boosted_baseline_notional,
        )
        final_notional = max(0.0, final_notional)
        final_quantity = final_notional / current_price if current_price > 0 else 0.0

        if abs(final_quantity - instruction.quantity) > 1e-9:
            logger.info(
                "[%s] portfolio targeting adjusted buy size: qty %.6f -> %.6f, notional %.2f -> %.2f",
                symbol,
                instruction.quantity,
                final_quantity,
                instruction.size_notional,
                final_notional,
            )

        instruction.quantity = final_quantity
        instruction.size_notional = final_notional
        instruction.size_pct = final_notional / snapshot.nav if snapshot.nav > 0 else 0.0
        return instruction

    def _symbol_performance_score(self, symbol: str) -> float:
        if not self._symbol_performance_enabled:
            return 0.0
        if self._performance_manager.enabled:
            return float(self._performance_manager.score(symbol))
        strategy = self._strategies.get(symbol)
        if strategy is None:
            return 0.0
        return float(strategy.symbol_performance_score())

    def _record_online_icir(self, symbol: str, current_price: float) -> None:
        if current_price <= 0:
            return
        prev_price = self._prev_prices.get(symbol, 0.0)
        if prev_price <= 0:
            return
        realized_return = (current_price - prev_price) / prev_price
        if self._icir_tracker is not None and symbol in self._prev_factors:
            self._icir_tracker.record(symbol, self._prev_factors[symbol], realized_return)
        if (
            self._factor_icir_manager.enabled
            and symbol in self._prev_factor_snapshot_scores
        ):
            self._factor_icir_manager.record(
                symbol,
                self._prev_factor_snapshot_scores[symbol],
                realized_return,
            )

    def _store_rule_icir_inputs(self, symbol: str, features, current_price: float) -> None:
        if self._icir_tracker is not None:
            self._prev_factors[symbol] = [
                (50.0 - features.rsi) / 50.0,
                max(-1.0, min(1.0, features.momentum * 20.0)),
                max(
                    -1.0,
                    min(
                        1.0,
                        (features.ema_fast - features.ema_slow)
                        / features.ema_slow
                        * 100.0,
                    ),
                )
                if features.ema_slow > 0
                else 0.0,
                min(1.0, features.volatility * 50.0),
            ]
        self._prev_prices[symbol] = current_price

    def _store_factor_snapshot_history(self, symbol: str, factor_snapshot) -> None:
        if not self._factor_icir_manager.enabled:
            return
        scores: Dict[str, float] = {}
        for obs in factor_snapshot.observations:
            if obs.name not in self._factor_icir_manager.targets:
                continue
            signed_strength = obs.strength
            if obs.bias == FactorBias.BEARISH:
                signed_strength = -obs.strength
            elif obs.bias == FactorBias.NEUTRAL:
                signed_strength = 0.0
            scores[obs.name] = float(signed_strength)
        self._prev_factor_snapshot_scores[symbol] = scores

    def _dynamic_entry_penalty(self, symbol: str) -> float:
        score = self._symbol_performance_score(symbol)
        if symbol in self._satellite_symbols:
            return (
                self._satellite_underperformance_penalty
                if score < self._satellite_underperformance_threshold
                else 0.0
            )
        if symbol in self._core_symbols:
            return (
                self._core_underperformance_penalty
                if score < self._core_underperformance_threshold
                else 0.0
            )
        return 0.0

    def _cancel_buy_candidate(self, candidate: dict[str, Any], reason: str) -> None:
        symbol = candidate["symbol"]
        logger.info("[%s] skipped buy candidate: %s", symbol, reason)
        strategy = candidate.get("strategy")
        current_price = candidate.get("current_price")
        intent = candidate.get("intent")
        if strategy is None or current_price is None or intent is None:
            return
        order = strategy.build_instruction(
            intent,
            current_price=current_price,
        ).to_order()
        strategy.on_cancel(order)

    def _cancel_candidate_batch(
        self, candidates: list[dict[str, Any]], reason: str
    ) -> None:
        for candidate in candidates:
            self._cancel_buy_candidate(candidate, reason)

    def _active_symbol_count(self, exclude_symbols: Optional[Set[str]] = None) -> int:
        excluded = exclude_symbols or set()
        active_symbols = {
            pos.symbol
            for pos in self._tracker.snapshot().positions
            if pos.quantity > 0 and pos.symbol not in excluded
        }
        active_states = {
            StrategyState.LONG_PENDING,
            StrategyState.HOLDING,
            StrategyState.EXIT_PENDING,
        }
        active_symbols.update(
            symbol
            for symbol, strategy in self._strategies.items()
            if symbol not in excluded and strategy._state in active_states
        )
        return len(active_symbols)

    def _active_satellite_count(
        self, exclude_symbols: Optional[Set[str]] = None
    ) -> int:
        excluded = exclude_symbols or set()
        return len(
            self._satellite_symbols.intersection(
                {
                    pos.symbol
                    for pos in self._tracker.snapshot().positions
                    if pos.quantity > 0 and pos.symbol not in excluded
                }
            ).union(
                {
                    symbol
                    for symbol, strategy in self._strategies.items()
                    if (
                        symbol in self._satellite_symbols
                        and symbol not in excluded
                        and strategy._state
                        in {
                            StrategyState.LONG_PENDING,
                            StrategyState.HOLDING,
                            StrategyState.EXIT_PENDING,
                        }
                    )
                }
            )
        )

    async def _liquidate_all(self) -> None:
        """Emergency liquidation: sell all positions."""
        logger.critical("LIQUIDATING ALL POSITIONS")
        snapshot = self._tracker.snapshot()

        for pos in snapshot.positions:
            if pos.quantity > 0:
                from core.models import OrderType, Side

                if self._has_pending_side_order(pos.symbol, Side.SELL):
                    self._mark_strategy_exit_pending(pos.symbol)
                    continue
                order = Order(
                    symbol=pos.symbol,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                )
                await self._order_manager.submit(order)
                self._mark_strategy_exit_pending(pos.symbol)

    def _on_order_event(self, order: Order) -> None:
        """Callback for order fill/cancel events."""
        if order.symbol in self._strategies:
            if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                self._strategies[order.symbol].on_fill(order)
            elif order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                self._strategies[order.symbol].on_cancel(order)

    def _has_pending_side_order(self, symbol: str, side: Side) -> bool:
        pending_statuses = {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        }
        active_orders = getattr(self._order_manager, "active_orders", {})
        if not isinstance(active_orders, dict):
            return False
        return any(
            order.symbol == symbol
            and order.side == side
            and order.status in pending_statuses
            for order in active_orders.values()
        )

    def _mark_strategy_exit_pending(self, symbol: str) -> None:
        strategy = self._strategies.get(symbol)
        if strategy is None:
            return
        strategy.mark_exit_pending()

    def _build_market_context(
        self, symbol_state: Dict[str, Dict[str, Any]]
    ) -> Optional[dict]:
        regime_cfg = self._config.get("regime", {})
        if not regime_cfg.get("enabled", False):
            return None
        if not symbol_state:
            return None

        benchmark_symbols = regime_cfg.get(
            "benchmark_symbols", ["BTC/USDT", "ETH/USDT"]
        )
        benchmark_scores: Dict[str, float] = {}
        benchmark_funding: list[float] = []
        benchmark_volatility: list[float] = []

        for symbol in benchmark_symbols:
            state = symbol_state.get(symbol)
            if state is None:
                continue
            features = state["features"]
            benchmark_scores[symbol] = self._benchmark_score(
                features,
                candles_1h=state.get("candles_1h"),
            )
            benchmark_funding.append(
                state["supplementary"].get("funding_rate", features.funding_rate)
            )
            benchmark_volatility.append(features.volatility)

        if not benchmark_scores:
            return None

        feature_list = [state["features"] for state in symbol_state.values()]
        positive_count = sum(
            1
            for feat in feature_list
            if feat.momentum > 0 and feat.ema_fast >= feat.ema_slow
        )
        breadth = positive_count / len(feature_list)
        breadth_score = max(-1.0, min(1.0, 2.0 * breadth - 1.0))
        volume_expansion = sum(
            max(0.0, min(1.0, (feat.volume_ratio - 1.0) / 1.5))
            for feat in feature_list
        ) / max(len(feature_list), 1)
        benchmark_avg = sum(benchmark_scores.values()) / len(benchmark_scores)
        volatility_ceiling = regime_cfg.get("volatility_ceiling", 0.020)
        avg_benchmark_vol = sum(benchmark_volatility) / max(len(benchmark_volatility), 1)
        vol_stress = max(
            0.0,
            min(1.0, avg_benchmark_vol / max(volatility_ceiling, 1e-6) - 1.0),
        )

        score = (
            0.50 * benchmark_avg
            + 0.30 * breadth_score
            + 0.10 * volume_expansion
            - 0.10 * vol_stress
        )
        risk_on_threshold = regime_cfg.get("risk_on_threshold", 0.25)
        neutral_threshold = regime_cfg.get("neutral_threshold", 0.05)
        breadth_min_symbols = max(0, int(regime_cfg.get("breadth_min_symbols", 0)))
        breadth_ok = (
            positive_count >= min(breadth_min_symbols, len(feature_list))
            if breadth_min_symbols > 0
            else True
        )

        if score >= risk_on_threshold and breadth_ok:
            regime = "risk_on"
        elif score >= neutral_threshold:
            regime = "neutral"
        else:
            regime = "risk_off"

        return {
            "regime": regime,
            "score": max(-1.0, min(1.0, score)),
            "breadth": breadth,
            "positive_symbols": positive_count,
            "breadth_ok": breadth_ok,
            "volume_expansion": volume_expansion,
            "avg_funding": sum(benchmark_funding) / max(len(benchmark_funding), 1),
            "avg_benchmark_volatility": avg_benchmark_vol,
            "vol_stress": vol_stress,
            "benchmarks": benchmark_scores,
        }

    @staticmethod
    def _benchmark_score(features: Any, candles_1h: Optional[list[OHLCV]]) -> float:
        ema_spread = 0.0
        if features.ema_slow > 0:
            ema_spread = (features.ema_fast - features.ema_slow) / features.ema_slow

        hourly_momentum = 0.0
        if candles_1h and len(candles_1h) >= 6 and candles_1h[-6].close > 0:
            hourly_momentum = (
                candles_1h[-1].close - candles_1h[-6].close
            ) / candles_1h[-6].close

        score = ema_spread * 120.0 + features.momentum * 8.0 + hourly_momentum * 10.0
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _format_quantity(quantity: float) -> str:
        if quantity >= 100:
            return f"{quantity:.2f}"
        if quantity >= 1:
            return f"{quantity:.4f}"
        if quantity >= 0.01:
            return f"{quantity:.6f}"
        return f"{quantity:.8f}"

    @classmethod
    def _format_balance_item(cls, asset: str, value: float) -> str:
        if asset == "USD":
            return f"{asset}:{value:.2f}"
        return f"{asset}:{cls._format_quantity(value)}"

    @classmethod
    def _format_regime_status(cls, market_context: Optional[dict]) -> str:
        if not market_context:
            return "Regime=n/a"
        regime = market_context.get("regime", "n/a")
        score = float(market_context.get("score", 0.0))
        breadth = float(market_context.get("breadth", 0.0))
        return f"Regime={regime}(score={score:.3f}, breadth={breadth:.2f})"
