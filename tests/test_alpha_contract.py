import json
import pytest
from pathlib import Path
from alpha.contract import AlphaSpec, load_alpha, validate_alpha_features


class TestAlphaSpec:
    def test_load_valid_expression_tree(self, tmp_path):
        alpha_json = {
            "alpha_id": "test_momentum",
            "version": "1.0.0",
            "type": "expression_tree",
            "description": "Test alpha",
            "expression": "Delta($close, 5)",
            "normalization": {"method": "rolling_zscore", "lookback": 20},
            "weight_hint": 0.5,
            "horizon": "intraday",
            "meta": {"author": "test"},
        }
        f = tmp_path / "test.json"
        f.write_text(json.dumps(alpha_json))

        spec = load_alpha(f)
        assert spec.alpha_id == "test_momentum"
        assert spec.type == "expression_tree"
        assert spec.expression == "Delta($close, 5)"
        assert spec.normalization.method == "rolling_zscore"
        assert spec.weight_hint == 0.5

    def test_missing_expression_fails(self, tmp_path):
        alpha_json = {
            "alpha_id": "bad",
            "version": "1.0.0",
            "type": "expression_tree",
            "normalization": {"method": "rolling_zscore", "lookback": 20},
        }
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(alpha_json))

        with pytest.raises(Exception):
            load_alpha(f)

    def test_invalid_normalization_method_fails(self, tmp_path):
        alpha_json = {
            "alpha_id": "bad",
            "version": "1.0.0",
            "type": "expression_tree",
            "expression": "$close",
            "normalization": {"method": "invalid_method", "lookback": 20},
        }
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(alpha_json))

        with pytest.raises(Exception):
            load_alpha(f)

    def test_model_type_accepted(self, tmp_path):
        alpha_json = {
            "alpha_id": "lstm_v1",
            "version": "1.0.0",
            "type": "model",
            "description": "LSTM alpha",
            "compute": {
                "checkpoint": "artifacts/model.onnx",
                "input_features": ["rsi", "momentum"],
                "sequence_length": 240,
            },
            "normalization": {"method": "rolling_zscore", "lookback": 20},
            "weight_hint": 0.2,
            "meta": {"author": "test"},
        }
        f = tmp_path / "model.json"
        f.write_text(json.dumps(alpha_json))

        spec = load_alpha(f)
        assert spec.type == "model"
        assert spec.compute.checkpoint == "artifacts/model.onnx"


class TestFeatureValidation:
    def test_valid_features_pass(self):
        available = {"close", "volume", "open", "high", "low", "vwap"}
        errors = validate_alpha_features("Delta($close, 5)", available)
        assert errors == []

    def test_missing_feature_detected(self):
        available = {"close"}
        errors = validate_alpha_features("Mul($close, $volume)", available)
        assert len(errors) == 1
        assert "volume" in errors[0]
