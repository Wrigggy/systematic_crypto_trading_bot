from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from data.buffer import LiveBuffer
from data.resampler import CandleResampler, MultiResampler
from execution.trade_logger import TradeLogger
from features.extractor import FeatureExtractor
from main import _build_feature_config
from models.inference import AlphaEngine
from models.model_wrapper import ModelWrapper
from risk.tracker import PortfolioTracker
from strategy.factor_engine import FactorEngine

from backtest.analysis import BacktestResult, TradeAnalyzer
from backtest.backtest_monitor import BacktestMonitor
from backtest.data_loader import SymbolDataset, iter_candle_batches
from backtest.simulated_execution import BacktestOrderManager, BacktestSimExecutor
from backtest.simulated_risk import BacktestRiskShield

logger = logging.getLogger(__name__)


class BacktestRunner:
    def __init__(self, config: dict, report_root: Path):
        self._config = config
        self._report_root = report_root

    async def run(self, datasets: Dict[str, SymbolDataset], run_id: str) -> BacktestResult:
        if not datasets:
            raise ValueError("No datasets supplied")

        data_cfg = self._config.get("data", {})
        buffer = LiveBuffer(
            max_candles=data_cfg.get("buffer_size", 1500),
            max_ticks=data_cfg.get("tick_buffer_size", 5000),
        )
        paper_cfg = self._config.get("paper", {})
        executor = BacktestSimExecutor(paper_cfg, buffer)
        extractor = FeatureExtractor(_build_feature_config(self._config))
        factor_engine = FactorEngine(self._config)
        tracker = PortfolioTracker(
            paper_cfg.get("initial_capital", 100000.0),
            paper_cfg.get("fee_bps", 10.0),
        )
        risk_shield = BacktestRiskShield(self._config)
        order_manager = BacktestOrderManager(
            executor,
            tracker,
            timeout_seconds=self._config.get("execution", {}).get("order_timeout_seconds", 0),
        )

        model = self._load_optional_model(extractor)
        alpha_engine = AlphaEngine(self._config, extractor, model=model)

        resample_minutes = self._config.get("alpha", {}).get("resample_minutes", 1)
        resampler = CandleResampler(resample_minutes) if resample_minutes > 1 else None

        multi_timeframes = self._config.get("alpha", {}).get("multi_timeframes", [])
        multi_resampler = None
        if multi_timeframes:
            multi_resampler = MultiResampler(sorted(set([resample_minutes] + multi_timeframes)))
            resampler = None

        run_dir = self._report_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        trade_logger = TradeLogger(log_dir=str(run_dir))
        monitor = BacktestMonitor(
            config=self._config,
            buffer=buffer,
            extractor=extractor,
            alpha_engine=alpha_engine,
            risk_shield=risk_shield,
            tracker=tracker,
            order_manager=order_manager,
            factor_engine=factor_engine,
            resampler=resampler,
            multi_resampler=multi_resampler,
            trade_logger=trade_logger,
        )

        analyzer = TradeAnalyzer(fee_rate=paper_cfg.get("fee_bps", 10.0) / 10000.0)
        order_manager.register_fill_callback(analyzer.on_fill)

        start = None
        end = None
        last_timestamp = None

        for iteration, (timestamp, candles) in enumerate(iter_candle_batches(datasets), start=1):
            if start is None:
                start = timestamp
            end = timestamp
            last_timestamp = timestamp

            for candle in candles:
                await buffer.push_candle(candle)

            await monitor.process_backtest_iteration(iteration, timestamp)
            snapshot = tracker.snapshot()
            analyzer.record_equity(timestamp, snapshot, tracker.get_total_exposure())

        if last_timestamp is None or start is None or end is None:
            raise ValueError("No historical timestamps were processed")

        await order_manager.cancel_all()
        final_snapshot = tracker.snapshot()
        analyzer.record_equity(last_timestamp, final_snapshot, tracker.get_total_exposure())

        open_positions = sum(1 for pos in final_snapshot.positions if pos.quantity > 0)
        result = analyzer.build_result(
            run_id=run_id,
            strategy_profile=self._config.get("strategy", {}).get("profile", "custom"),
            symbols=list(datasets),
            start=start,
            end=end,
            initial_capital=paper_cfg.get("initial_capital", 100000.0),
            open_positions=open_positions,
        )
        analyzer.write_artifacts(run_dir, result)
        return result

    def _load_optional_model(self, extractor: FeatureExtractor) -> ModelWrapper | None:
        alpha_cfg = self._config.get("alpha", {})
        engine_type = alpha_cfg.get("engine", "rule_based")
        if engine_type not in ("lstm", "transformer", "ensemble"):
            return None

        model_path = alpha_cfg.get("model_path", "")
        model_type = alpha_cfg.get("model_type", "lstm")
        path = Path(model_path)
        if not model_path or not path.exists():
            logger.warning(
                "Model path %s not found; backtest will fall back to rule-based alpha",
                model_path,
            )
            self._config.setdefault("alpha", {})["engine"] = "rule_based"
            return None

        model = ModelWrapper(str(path), n_features=extractor.N_FEATURES, model_type=model_type)
        model.load()
        return model
