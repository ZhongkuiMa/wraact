"""Tests for activation function implementations (NumPy and derivatives).

Tests verify that the wrapped NumPy implementations match expected behavior.
"""

__docformat__ = "restructuredtext"

import numpy as np


class TestReLUFunction:
    """Tests for ReLU activation function."""

    def test_relu_basic_property(self):
        """ReLU should return max(0, x)."""
        from wraact._functions import relu_np

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = relu_np(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])

        np.testing.assert_array_almost_equal(y, expected)

    def test_relu_scalar(self):
        """ReLU should work on scalars."""
        from wraact._functions import relu_np

        assert relu_np(-1.0) == 0.0
        assert relu_np(0.0) == 0.0
        assert relu_np(1.0) == 1.0

    def test_relu_preserves_positive(self):
        """ReLU should preserve positive values."""
        from wraact._functions import relu_np

        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        y = relu_np(x)

        np.testing.assert_array_almost_equal(y, x)

    def test_relu_zeros_negative(self):
        """ReLU should zero out negative values."""
        from wraact._functions import relu_np

        x = np.array([-10.0, -2.0, -0.1])
        y = relu_np(x)

        np.testing.assert_array_almost_equal(y, np.zeros_like(x))


class TestLeakyReLUFunction:
    """Tests for LeakyReLU activation function."""

    def test_leakyrelu_default_slope(self):
        """LeakyReLU with default slope should be y = max(x, 0.01*x)."""
        from wraact._functions import leakyrelu_np

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = leakyrelu_np(x)
        expected = np.array([-0.02, -0.01, 0.0, 1.0, 2.0])

        np.testing.assert_array_almost_equal(y, expected, decimal=8)

    def test_leakyrelu_custom_slope(self):
        """LeakyReLU with custom slope."""
        from wraact._functions import leakyrelu_np

        x = np.array([-2.0, -1.0, 1.0, 2.0])
        y = leakyrelu_np(x, negative_slope=0.1)
        expected = np.array([-0.2, -0.1, 1.0, 2.0])

        np.testing.assert_array_almost_equal(y, expected)

    def test_leakyrelu_positive_pass_through(self):
        """LeakyReLU should pass through positive values unchanged."""
        from wraact._functions import leakyrelu_np

        x = np.array([0.0, 0.1, 1.0, 10.0])
        y = leakyrelu_np(x, negative_slope=0.01)

        np.testing.assert_array_almost_equal(y, x)


class TestELUFunction:
    """Tests for ELU activation function."""

    def test_elu_positive_identity(self):
        """ELU should be identity for positive inputs."""
        from wraact._functions import elu_np

        x = np.array([0.0, 0.1, 1.0, 2.0])
        y = elu_np(x)

        np.testing.assert_array_almost_equal(y, x)

    def test_elu_negative_exponential(self):
        """ELU should be exp(x)-1 for negative inputs."""
        from wraact._functions import elu_np

        x = np.array([-1.0, -0.5, -0.1])
        y = elu_np(x)
        expected = np.exp(x) - 1.0

        np.testing.assert_array_almost_equal(y, expected)

    def test_elu_continuous_at_zero(self):
        """ELU should be continuous at x=0."""
        from wraact._functions import elu_np

        y_at_zero = elu_np(0.0)
        assert y_at_zero == 0.0

    def test_elu_bounds_negative_region(self):
        """ELU output for negative inputs should be in [-1, 0)."""
        from wraact._functions import elu_np

        x = np.linspace(-10.0, -0.001, 100)
        y = elu_np(x)

        assert np.all(y >= -1.0)
        assert np.all(y < 0.0)


class TestSigmoidFunction:
    """Tests for Sigmoid activation function."""

    def test_sigmoid_output_range(self):
        """Sigmoid should output values in [0, 1]."""
        from wraact._functions import sigmoid_np

        x = np.linspace(-10, 10, 100)
        y = sigmoid_np(x)

        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_sigmoid_at_zero(self):
        """sigmoid(0) should be 0.5."""
        from wraact._functions import sigmoid_np

        y = sigmoid_np(0.0)
        assert np.isclose(y, 0.5)

    def test_sigmoid_symmetry(self):
        """sigmoid(-x) + sigmoid(x) should equal 1."""
        from wraact._functions import sigmoid_np

        x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        y = sigmoid_np(x)
        y_neg = sigmoid_np(-x)

        np.testing.assert_array_almost_equal(y + y_neg, np.ones_like(x))

    def test_sigmoid_limits(self):
        """Sigmoid should approach 0 as x→-∞ and 1 as x→+∞."""
        from wraact._functions import sigmoid_np

        y_very_neg = sigmoid_np(-100.0)
        y_very_pos = sigmoid_np(100.0)

        assert y_very_neg < 1e-10
        assert y_very_pos > 1.0 - 1e-10


class TestTanhFunction:
    """Tests for Tanh activation function."""

    def test_tanh_output_range(self):
        """Tanh should output values in [-1, 1]."""
        from wraact._functions import tanh_np

        x = np.linspace(-10, 10, 100)
        y = tanh_np(x)

        assert np.all(y >= -1.0)
        assert np.all(y <= 1.0)

    def test_tanh_at_zero(self):
        """tanh(0) should be 0."""
        from wraact._functions import tanh_np

        y = tanh_np(0.0)
        assert y == 0.0

    def test_tanh_odd_function(self):
        """Tanh should be odd: tanh(-x) = -tanh(x)."""
        from wraact._functions import tanh_np

        x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        y = tanh_np(x)
        y_neg = tanh_np(-x)

        np.testing.assert_array_almost_equal(y_neg, -y)

    def test_tanh_limits(self):
        """Tanh should approach -1 as x→-∞ and 1 as x→+∞."""
        from wraact._functions import tanh_np

        y_very_neg = tanh_np(-100.0)
        y_very_pos = tanh_np(100.0)

        assert y_very_neg < -1.0 + 1e-10
        assert y_very_pos > 1.0 - 1e-10


class TestDerivatives:
    """Tests for activation function derivatives."""

    def test_relu_derivative_properties(self):
        """ReLU derivative should be 0 for x<0 and 1 for x>0."""
        from wraact._functions import drelu_np

        x_neg = np.array([-2.0, -1.0, -0.1])
        x_pos = np.array([0.1, 1.0, 2.0])

        dy_neg = drelu_np(x_neg)
        dy_pos = drelu_np(x_pos)

        np.testing.assert_array_almost_equal(dy_neg, np.zeros_like(x_neg))
        np.testing.assert_array_almost_equal(dy_pos, np.ones_like(x_pos))

    def test_sigmoid_derivative_range(self):
        """Sigmoid derivative should be in (0, 0.25] (except at extremes)."""
        from wraact._functions import dsigmoid_np

        x = np.linspace(-10, 10, 1000)  # Use finite range to avoid underflow
        dy = dsigmoid_np(x)

        assert np.all(dy > 0.0)
        assert np.all(dy <= 0.25 + 1e-10)

    def test_sigmoid_derivative_max_at_zero(self):
        """Sigmoid derivative should be maximum at x=0."""
        from wraact._functions import dsigmoid_np

        y_at_zero = dsigmoid_np(0.0)
        x_test = np.array([-1.0, -0.5, 0.5, 1.0])
        y_test = dsigmoid_np(x_test)

        assert y_at_zero > np.max(y_test)
        assert np.isclose(y_at_zero, 0.25)

    def test_tanh_derivative_range(self):
        """Tanh derivative should be in (0, 1] (except at extremes)."""
        from wraact._functions import dtanh_np

        x = np.linspace(-10, 10, 1000)  # Use finite range to avoid underflow
        dy = dtanh_np(x)

        assert np.all(dy > 0.0)
        assert np.all(dy <= 1.0 + 1e-10)

    def test_tanh_derivative_max_at_zero(self):
        """Tanh derivative should be maximum at x=0."""
        from wraact._functions import dtanh_np

        y_at_zero = dtanh_np(0.0)
        x_test = np.array([-1.0, -0.5, 0.5, 1.0])
        y_test = dtanh_np(x_test)

        assert y_at_zero > np.max(y_test)
        assert np.isclose(y_at_zero, 1.0)


class TestFunctionMonotonicity:
    """Test monotonicity properties of activation functions."""

    def test_relu_monotonic_increasing(self):
        """ReLU should be monotonically increasing."""
        from wraact._functions import relu_np

        x = np.linspace(-10, 10, 100)
        y = relu_np(x)

        dy = np.diff(y)
        assert np.all(dy >= 0.0)

    def test_sigmoid_monotonic_increasing(self):
        """Sigmoid should be monotonically increasing."""
        from wraact._functions import sigmoid_np

        x = np.linspace(-10, 10, 100)
        y = sigmoid_np(x)

        dy = np.diff(y)
        assert np.all(dy > 0.0)

    def test_tanh_monotonic_increasing(self):
        """Tanh should be monotonically increasing."""
        from wraact._functions import tanh_np

        x = np.linspace(-10, 10, 100)
        y = tanh_np(x)

        dy = np.diff(y)
        assert np.all(dy > 0.0)

    def test_leakyrelu_monotonic_increasing(self):
        """LeakyReLU should be monotonically increasing."""
        from wraact._functions import leakyrelu_np

        x = np.linspace(-10, 10, 100)
        y = leakyrelu_np(x, negative_slope=0.01)

        dy = np.diff(y)
        assert np.all(dy > 0.0)


class TestFunctionCompositionProperties:
    """Test mathematical properties and relationships between functions."""

    def test_relu_vs_leakyrelu(self):
        """LeakyReLU should be <= ReLU in magnitude for negative inputs."""
        from wraact._functions import leakyrelu_np, relu_np

        x = np.array([-2.0, -1.0, -0.5])
        y_relu = relu_np(x)
        y_leakyrelu = leakyrelu_np(x, negative_slope=0.01)

        # For negative inputs: relu is always 0, leakyrelu is negative
        # So leakyrelu < relu
        assert np.all(y_leakyrelu < y_relu)

    def test_tanh_sigmoid_relationship(self):
        """tanh(x) = 2*sigmoid(2x) - 1."""
        from wraact._functions import sigmoid_np, tanh_np

        x = np.linspace(-2, 2, 50)
        y_tanh = tanh_np(x)
        y_from_sigmoid = 2.0 * sigmoid_np(2.0 * x) - 1.0

        np.testing.assert_array_almost_equal(y_tanh, y_from_sigmoid, decimal=10)

    def test_elu_vs_relu(self):
        """ELU should allow negative outputs (unlike ReLU)."""
        from wraact._functions import elu_np, relu_np

        x = np.array([-2.0, -1.0, -0.1])
        y_elu = elu_np(x)
        y_relu = relu_np(x)

        # ELU outputs should be negative for negative inputs
        assert np.all(y_elu < 0.0)
        # ReLU outputs should be zero
        assert np.all(y_relu == 0.0)
        # ELU allows negative (in range [-1, 0)), ReLU only gives 0
        assert np.all(y_elu <= y_relu)
