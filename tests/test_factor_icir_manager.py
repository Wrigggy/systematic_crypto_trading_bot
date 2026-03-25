from strategy.factor_icir_manager import BayesianFactorICIRManager


def test_factor_icir_manager_overweights_more_predictive_factor():
    config = {
        "strategy": {
            "factor_icir_enabled": True,
            "factor_icir_targets": [
                "onchain_mvrv_state",
                "onchain_activity_state",
                "onchain_nvt_proxy",
            ],
            "factor_icir_window": 50,
            "factor_icir_min_samples": 10,
            "factor_icir_min_lambda": 0.35,
            "factor_icir_tau": 20.0,
            "factor_icir_weight_floor": 0.70,
            "factor_icir_weight_ceiling": 1.35,
            "factor_weights": {
                "onchain_mvrv_state": 0.10,
                "onchain_activity_state": 0.10,
                "onchain_nvt_proxy": 0.08,
            },
        }
    }
    manager = BayesianFactorICIRManager(config)

    for idx in range(20):
        realized_return = 0.01 if idx % 2 == 0 else -0.01
        manager.record(
            "BTC/USDT",
            {
                "onchain_mvrv_state": realized_return * 50.0,
                "onchain_activity_state": 0.02,
                "onchain_nvt_proxy": -realized_return * 10.0,
            },
            realized_return,
        )

    assert manager.multiplier("BTC/USDT", "onchain_mvrv_state") > 1.0
    assert manager.multiplier("BTC/USDT", "onchain_activity_state") <= 1.0
