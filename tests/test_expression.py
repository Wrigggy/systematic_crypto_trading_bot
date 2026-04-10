import numpy as np
import pytest
from alpha.expression import ExpressionParser, ExpressionEvaluator


class TestExpressionParser:
    def setup_method(self):
        self.parser = ExpressionParser()

    def test_parse_feature(self):
        expr = self.parser.parse("$close")
        assert expr.kind == "feature"
        assert expr.value == "close"

    def test_parse_constant(self):
        expr = self.parser.parse("1.5")
        assert expr.kind == "constant"
        assert expr.value == 1.5

    def test_parse_unary(self):
        expr = self.parser.parse("Abs($close)")
        assert expr.kind == "operator"
        assert expr.name == "Abs"
        assert len(expr.children) == 1

    def test_parse_binary(self):
        expr = self.parser.parse("Mul($close, $volume)")
        assert expr.kind == "operator"
        assert expr.name == "Mul"
        assert len(expr.children) == 2

    def test_parse_rolling(self):
        expr = self.parser.parse("Mean($close, 20)")
        assert expr.kind == "operator"
        assert expr.name == "Mean"
        assert expr.window == 20

    def test_parse_nested(self):
        expr = self.parser.parse("Mul(Delta($close, 5), Div($volume, Mean($volume, 20)))")
        assert expr.kind == "operator"
        assert expr.name == "Mul"

    def test_parse_invalid_feature(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            self.parser.parse("$nonexistent")

    def test_parse_invalid_operator(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            self.parser.parse("FakeOp($close)")

    def test_referenced_features(self):
        expr = self.parser.parse("Mul(Delta($close, 5), $volume)")
        assert expr.referenced_features() == {"close", "volume"}


class TestExpressionEvaluator:
    def setup_method(self):
        self.parser = ExpressionParser()
        np.random.seed(42)
        n = 100
        base = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        self.data = {
            "open": base + np.random.randn(n) * 0.1,
            "high": base + abs(np.random.randn(n)) * 0.5,
            "low": base - abs(np.random.randn(n)) * 0.5,
            "close": base.copy(),
            "volume": np.random.exponential(1000, n),
            "vwap": base + np.random.randn(n) * 0.05,
        }
        self.evaluator = ExpressionEvaluator()

    def test_evaluate_feature(self):
        expr = self.parser.parse("$close")
        result = self.evaluator.evaluate(expr, self.data)
        np.testing.assert_array_equal(result, self.data["close"])

    def test_evaluate_constant(self):
        expr = self.parser.parse("1.5")
        result = self.evaluator.evaluate(expr, self.data)
        assert np.all(result == 1.5)
        assert len(result) == len(self.data["close"])

    def test_evaluate_abs(self):
        expr = self.parser.parse("Abs(Sub($close, $open))")
        result = self.evaluator.evaluate(expr, self.data)
        expected = np.abs(self.data["close"] - self.data["open"])
        np.testing.assert_allclose(result, expected)

    def test_evaluate_mean_rolling(self):
        expr = self.parser.parse("Mean($close, 20)")
        result = self.evaluator.evaluate(expr, self.data)
        assert np.all(np.isnan(result[:19]))
        expected = np.mean(self.data["close"][:20])
        np.testing.assert_allclose(result[19], expected, rtol=1e-10)

    def test_evaluate_delta(self):
        expr = self.parser.parse("Delta($close, 5)")
        result = self.evaluator.evaluate(expr, self.data)
        assert np.isnan(result[4])
        expected = self.data["close"][5] - self.data["close"][0]
        np.testing.assert_allclose(result[5], expected)

    def test_evaluate_nested(self):
        expr = self.parser.parse("Mul(Delta($close, 5), Div($volume, Mean($volume, 20)))")
        result = self.evaluator.evaluate(expr, self.data)
        assert len(result) == 100
        assert np.all(np.isnan(result[:19]))
        valid = result[19:]
        assert np.all(np.isfinite(valid))

    def test_evaluate_csrank(self):
        expr = self.parser.parse("CSRank($close)")
        result = self.evaluator.evaluate(expr, self.data)
        np.testing.assert_allclose(result, np.full(100, 0.5))

    def test_nan_propagation(self):
        self.data["close"][50] = np.nan
        expr = self.parser.parse("Abs($close)")
        result = self.evaluator.evaluate(expr, self.data)
        assert np.isnan(result[50])
