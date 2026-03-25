from strategy.bayesian_symbol_performance import BayesianSymbolPerformanceManager


def test_bayesian_symbol_performance_shrinks_small_samples_toward_prior():
    config = {
        "strategy": {
            "bayesian_symbol_performance_enabled": True,
            "symbol_performance_window": 8,
            "symbol_performance_min_trades": 1,
            "bayesian_symbol_prior_mean": 0.0,
            "bayesian_symbol_prior_strength": 8.0,
            "bayesian_symbol_observation_strength": 1.0,
            "bayesian_symbol_uncertainty_penalty": 0.0,
        }
    }
    manager = BayesianSymbolPerformanceManager(config)

    manager.record_trade("BTC/USDT", -0.10)

    score = manager.score("BTC/USDT")
    assert score < 0.0
    assert score > -0.10


def test_bayesian_symbol_performance_penalizes_consistent_losses_more_than_one_loss():
    config = {
        "strategy": {
            "bayesian_symbol_performance_enabled": True,
            "symbol_performance_window": 8,
            "symbol_performance_min_trades": 1,
            "bayesian_symbol_prior_mean": 0.0,
            "bayesian_symbol_prior_strength": 4.0,
            "bayesian_symbol_observation_strength": 1.5,
            "bayesian_symbol_uncertainty_penalty": 0.5,
        }
    }
    manager = BayesianSymbolPerformanceManager(config)

    manager.record_trade("SOL/USDT", -0.02)
    first_profile = manager.profile("SOL/USDT")
    manager.record_trade("SOL/USDT", -0.05)
    manager.record_trade("SOL/USDT", -0.04)
    later_profile = manager.profile("SOL/USDT")

    assert later_profile["posterior_mean"] < first_profile["posterior_mean"]
    assert later_profile["sample_count"] > first_profile["sample_count"]
