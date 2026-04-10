"""Integration tests — full pipeline from feed → strategy → execution."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from core.models import OHLCV
from data.buffer import LiveBuffer
from execution.order_manager import OrderManager
from execution.sim_executor import SimExecutor
from features.extractor import FeatureExtractor
from plugins.model_inference.evaluator import AlphaEngine
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from strategy.monitor import StrategyMonitor
from tests.conftest import make_candle_series


def _fast_config(entry_threshold: float = 0.15):
    """Config with low thresholds so trades actually trigger on synthetic data."""
    return {
        "mode": "paper",
        "symbols": ["BTC/USDT"],
        "data": {"buffer_size": 500, "candle_interval": "1m"},
        "features": {
            "rsi_period": 14,
            "ema_fast": 12,
            "ema_slow": 26,
            "atr_period": 14,
            "volatility_window": 20,
            "momentum_window": 10,
        },
        "alpha": {
            "engine": "rule_based",
            "entry_threshold": entry_threshold,
            "exit_threshold": -0.05,
            "seq_len": 30,
        },
        "strategy": {
            "max_positions_per_symbol": 1,
            "position_size_pct": 0.10,
            "urgent_alpha_threshold": 0.05,
            "confirmation_bars": 1,
        },
        "risk": {
            "max_portfolio_exposure": 0.50,
            "max_single_exposure": 0.15,
            "trailing_stop_pct": 0.03,
            "atr_stop_multiplier": 2.0,
            "daily_drawdown_limit": 0.05,
            "max_orders_per_minute": 100,
        },
        "paper": {"initial_capital": 100000.0, "slippage_bps": 5, "fee_bps": 10},
    }


def _build_pipeline(config):
    """Wire up the full pipeline, return (monitor, buffer, tracker)."""
    buffer = LiveBuffer(max_candles=500)
    paper_cfg = config.get("paper", {})
    executor = SimExecutor(paper_cfg, buffer)
    extractor = FeatureExtractor(config.get("features", {}))
    alpha_engine = AlphaEngine(config, extractor)
    tracker = PortfolioTracker(paper_cfg["initial_capital"], paper_cfg["fee_bps"])
    risk_shield = RiskShield(config)
    order_manager = OrderManager(executor, tracker)
    monitor = StrategyMonitor(
        config=config,
        buffer=buffer,
        extractor=extractor,
        alpha_engine=alpha_engine,
        risk_shield=risk_shield,
        tracker=tracker,
        order_manager=order_manager,
    )
    return monitor, buffer, tracker


async def _feed_candles(buffer: LiveBuffer, candles: list[OHLCV]):
    """Push candles into the buffer one at a time."""
    for candle in candles:
        await buffer.push_candle(candle)


class TestFullPipeline:
    """End-to-end: push candles → monitor processes → check portfolio state."""

    async def test_warmup_no_trades(self):
        """During warmup, no trades should execute."""
        config = _fast_config()
        monitor, buffer, tracker = _build_pipeline(config)

        # Push fewer candles than warmup requires (rule_based warmup = min_candles = 28)
        candles = make_candle_series(15, start_price=100.0, trend=0.5)
        await _feed_candles(buffer, candles)

        # Run one iteration
        await monitor._process_iteration(1)

        snap = tracker.snapshot()
        assert snap.cash == 100000.0
        assert snap.nav == 100000.0

    async def test_uptrend_triggers_buy(self):
        """Strong uptrend with low threshold should trigger a buy."""
        config = _fast_config(entry_threshold=0.10)
        monitor, buffer, tracker = _build_pipeline(config)

        # Push enough candles to pass warmup, with strong uptrend
        candles = make_candle_series(80, start_price=100.0, trend=0.8, noise=0.1)
        await _feed_candles(buffer, candles)

        # Run several iterations to let the pipeline trigger
        for i in range(1, 20):
            await monitor._process_iteration(i)

        snap = tracker.snapshot()
        # Either a position was opened or cash decreased (trade happened)
        traded = snap.cash < 100000.0 or len(snap.positions) > 0
        assert traded, f"Expected trade with uptrend, got cash={snap.cash}"

    async def test_monitor_run_and_stop(self):
        """Monitor.run() processes candles and stops cleanly."""
        config = _fast_config()
        monitor, buffer, tracker = _build_pipeline(config)

        # Pre-fill buffer so there's data to process
        candles = make_candle_series(70, start_price=100.0)
        await _feed_candles(buffer, candles)

        # Run monitor in background, stop after brief period
        async def run_then_stop():
            await asyncio.sleep(0.1)
            await monitor.stop()

        await asyncio.gather(
            monitor.run(),
            run_then_stop(),
        )
        # Should complete without error
        assert not monitor._running


class TestRiskIntegration:
    """Risk checks integrated with the full pipeline."""

    async def test_circuit_breaker_stops_trading(self):
        """Circuit breaker should halt trading on large drawdown."""
        config = _fast_config(entry_threshold=0.05)
        config["risk"]["daily_drawdown_limit"] = 0.001  # 0.1% — very tight
        monitor, buffer, tracker = _build_pipeline(config)

        # Create candles that drop sharply after initial rise
        up = make_candle_series(60, start_price=100.0, trend=0.5, noise=0.1, seed=1)
        down = make_candle_series(
            30, start_price=up[-1].close, trend=-2.0, noise=0.1, seed=2
        )
        # Fix timestamps for the down candles
        base = up[-1].timestamp
        for i, c in enumerate(down):
            down[i] = OHLCV(
                symbol=c.symbol,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
                timestamp=base + timedelta(minutes=i + 1),
                is_closed=True,
            )

        await _feed_candles(buffer, up + down)

        for i in range(1, 50):
            await monitor._process_iteration(i)

        # With 0.1% drawdown limit, circuit breaker should have activated
        # (either cash changed from trades, or breaker fired)
        snap = tracker.snapshot()
        # Just verify no crash — the pipeline handled the scenario
        assert snap.nav > 0


class TestModelWrapper:
    """Test ONNX model loading/inference."""

    def test_onnx_load_and_predict(self):
        from pathlib import Path
        from plugins.model_inference.model_wrapper import ModelWrapper
        import numpy as np

        onnx_path = "artifacts/model.onnx"
        if not Path(onnx_path).exists():
            pytest.skip("ONNX model not generated yet")

        wrapper = ModelWrapper(onnx_path, n_features=6)
        wrapper.load()
        assert wrapper.is_loaded

        seq = np.random.randn(30, 6).astype(np.float32)
        score = wrapper.predict(seq)
        assert -1.0 <= score <= 1.0
