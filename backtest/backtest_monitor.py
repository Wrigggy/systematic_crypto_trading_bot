from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Set

from core.models import OHLCV, Order, OrderStatus, Side
from strategy.monitor import StrategyMonitor


class BacktestMonitor(StrategyMonitor):
    """StrategyMonitor variant that uses simulated timestamps for backtests."""

    async def process_backtest_iteration(
        self, iteration: int, current_time: datetime
    ) -> None:
        today = current_time.strftime("%Y-%m-%d")
        if self._last_trading_date is not None and today != self._last_trading_date:
            self._risk_shield.reset_daily()
            self._tracker.reset_daily()
        self._last_trading_date = today

        if hasattr(self._order_manager, "check_pending_at"):
            await self._order_manager.check_pending_at(current_time)
        else:
            await self._order_manager.check_pending()

        latest_candles: Dict[str, OHLCV] = {}
        atr_values: Dict[str, float] = {}
        symbol_state: Dict[str, Dict[str, Any]] = {}
        alpha_ready: Set[str] = set()

        for symbol, _strategy in self._strategies.items():
            candles = await self._buffer.get_candles(symbol)
            if not candles:
                continue

            latest_candles[symbol] = candles[-1]
            self._tracker.update_prices(symbol, candles[-1].close)

            if self._multi_resampler is not None:
                resampled_bars = self._multi_resampler.push(candles[-1])
                for period, bar in resampled_bars.items():
                    if bar is not None:
                        await self._buffer.push_resampled(period, bar)
                if resampled_bars.get(self._multi_resampler.primary_minutes) is not None:
                    alpha_ready.add(symbol)
            elif self._resampler is not None:
                resampled = self._resampler.push(candles[-1])
                if resampled is not None:
                    await self._buffer.push_resampled(self._resampler.minutes, resampled)
                    alpha_ready.add(symbol)
            else:
                alpha_ready.add(symbol)

            if len(candles) < self._warmup_candles:
                continue

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

            features = self._extractor.extract(strategy_candles, supplementary=supplementary)
            atr_values[symbol] = features.atr

            candles_15m = None
            candles_1h = None
            if self._multi_timeframes:
                if 15 in self._multi_timeframes:
                    candles_15m = await self._buffer.get_resampled_candles(symbol, 15, n=50)
                if 60 in self._multi_timeframes:
                    candles_1h = await self._buffer.get_resampled_candles(symbol, 60, n=50)

            symbol_state[symbol] = {
                "candles": candles,
                "strategy_candles": strategy_candles,
                "supplementary": supplementary,
                "features": features,
                "candles_15m": candles_15m,
                "candles_1h": candles_1h,
            }

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

            factor_history_window = getattr(self._factor_engine, "supplementary_history_window", 2)
            supplementary_history_len = max(getattr(self._alpha_engine, "_seq_len", 30), factor_history_window)
            if self._use_model_overlay and self._primary_minutes > 1:
                supplementary_history_len *= self._primary_minutes

            supplementary_history = await self._buffer.get_supplementary_history(
                symbol, supplementary_history_len
            )
            model_supplementary_history = self._align_supplementary_history(
                supplementary_history,
                seq_len=getattr(self._alpha_engine, "_seq_len", 30),
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

            snapshot = self._tracker.snapshot()
            intent = strategy.on_factors(
                factor_snapshot,
                snapshot,
                current_price=candles[-1].close,
                model_signal=model_signal,
            )

            if intent is not None:
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

                instruction = strategy.build_instruction(intent, current_price=candles[-1].close)
                order = instruction.to_order()
                order.created_at = current_time
                if hasattr(self._risk_shield, "validate_at"):
                    validated = self._risk_shield.validate_at(
                        order,
                        self._tracker,
                        current_time=current_time,
                        market_price=candles[-1].close,
                    )
                else:
                    validated = self._risk_shield.validate(
                        order,
                        self._tracker,
                        market_price=candles[-1].close,
                    )
                if validated is not None:
                    if hasattr(self._order_manager, "submit_at"):
                        await self._order_manager.submit_at(validated, current_time)
                    else:
                        await self._order_manager.submit(validated)
                else:
                    strategy.on_cancel(order)

        ranked_buy_candidates = self._rank_buy_candidates(buy_candidates)
        selected_buy_candidates = ranked_buy_candidates[: self._top_n_entries_per_cycle]
        skipped_buy_candidates = ranked_buy_candidates[self._top_n_entries_per_cycle :]
        for candidate in skipped_buy_candidates:
            self._cancel_buy_candidate(candidate, "not in top entry cohort for this cycle")

        for idx, candidate in enumerate(selected_buy_candidates):
            remaining_candidates = selected_buy_candidates[idx:]
            unsubmitted_symbols = {item["symbol"] for item in remaining_candidates}
            if self._active_symbol_count(exclude_symbols=unsubmitted_symbols) >= self._max_active_positions:
                self._cancel_buy_candidate(candidate, "max active position cap reached")
                continue

            if (
                candidate["symbol"] in self._satellite_symbols
                and self._active_satellite_count(exclude_symbols=unsubmitted_symbols)
                >= self._satellite_max_active_positions
            ):
                self._cancel_buy_candidate(candidate, "satellite position cap reached")
                continue

            strategy = candidate["strategy"]
            intent = candidate["intent"]
            current_price = candidate["current_price"]
            instruction = strategy.build_instruction(intent, current_price=current_price)
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

            order = instruction.to_order()
            order.created_at = current_time
            if hasattr(self._risk_shield, "validate_at"):
                validated = self._risk_shield.validate_at(
                    order,
                    self._tracker,
                    current_time=current_time,
                    market_price=current_price,
                )
            else:
                validated = self._risk_shield.validate(
                    order,
                    self._tracker,
                    market_price=current_price,
                )
            if validated is not None:
                if hasattr(self._order_manager, "submit_at"):
                    await self._order_manager.submit_at(validated, current_time)
                else:
                    await self._order_manager.submit(validated)
            else:
                strategy.on_cancel(order)

        stop_orders = self._risk_shield.check_stops(self._tracker, latest_candles, atr_values)
        for stop_order in stop_orders:
            stop_order.created_at = current_time
            market_price = latest_candles.get(stop_order.symbol).close if latest_candles.get(stop_order.symbol) else 0.0
            if hasattr(self._risk_shield, "validate_at"):
                validated = self._risk_shield.validate_at(
                    stop_order,
                    self._tracker,
                    current_time=current_time,
                    market_price=market_price,
                    is_stop=True,
                )
            else:
                validated = self._risk_shield.validate(
                    stop_order,
                    self._tracker,
                    market_price=market_price,
                    is_stop=True,
                )
            if validated is not None:
                if hasattr(self._order_manager, "submit_at"):
                    await self._order_manager.submit_at(validated, current_time)
                else:
                    await self._order_manager.submit(validated)
                if stop_order.symbol in self._strategies:
                    self._strategies[stop_order.symbol].force_flat()

        if self._risk_shield.check_circuit_breaker(self._tracker):
            await self._liquidate_all()

    def _on_order_event(self, order: Order) -> None:
        if order.symbol in self._strategies:
            if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                self._strategies[order.symbol].on_fill(order)
            elif order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                self._strategies[order.symbol].on_cancel(order)
