"""Tests for activation function implementations (NumPy and derivatives).

Tests verify that the wrapped NumPy implementations match expected behavior.
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact._functions import (
    ddsigmoid_np,
    ddtanh_np,
    delu_np,
    dleakyrelu_np,
    drelu_np,
    dsigmoid_np,
    dtanh_np,
    elu_np,
    leakyrelu_np,
    relu_np,
    sigmoid_np,
    tanh_np,
)


class TestReLUFunction:
    """Tests for ReLU activation function."""

    def test_relu_basic_property(self):
        """ReLU should return max(0, x)."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = relu_np(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])

        np.testing.assert_array_almost_equal(y, expected)

    def test_relu_scalar(self):
        """ReLU should work on scalars."""
        assert relu_np(-1.0) == 0.0
        assert relu_np(0.0) == 0.0
        assert relu_np(1.0) == 1.0

    def test_relu_preserves_positive(self):
        """ReLU should preserve positive values."""
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        y = relu_np(x)

        np.testing.assert_array_almost_equal(y, x)

    def test_relu_zeros_negative(self):
        """ReLU should zero out negative values."""
        x = np.array([-10.0, -2.0, -0.1])
        y = relu_np(x)

        np.testing.assert_array_almost_equal(y, np.zeros_like(x))


class TestLeakyReLUFunction:
    """Tests for LeakyReLU activation function."""

    def test_leakyrelu_default_slope(self):
        """LeakyReLU with default slope should be y = max(x, 0.01*x)."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = leakyrelu_np(x)
        expected = np.array([-0.02, -0.01, 0.0, 1.0, 2.0])

        np.testing.assert_array_almost_equal(y, expected, decimal=8)

    def test_leakyrelu_custom_slope(self):
        """LeakyReLU with custom slope."""
        x = np.array([-2.0, -1.0, 1.0, 2.0])
        y = leakyrelu_np(x, negative_slope=0.1)
        expected = np.array([-0.2, -0.1, 1.0, 2.0])

        np.testing.assert_array_almost_equal(y, expected)

    def test_leakyrelu_positive_pass_through(self):
        """LeakyReLU should pass through positive values unchanged."""
        x = np.array([0.0, 0.1, 1.0, 10.0])
        y = leakyrelu_np(x, negative_slope=0.01)

        np.testing.assert_array_almost_equal(y, x)


class TestELUFunction:
    """Tests for ELU activation function."""

    def test_elu_positive_identity(self):
        """ELU should be identity for positive inputs."""
        x = np.array([0.0, 0.1, 1.0, 2.0])
        y = elu_np(x)

        np.testing.assert_array_almost_equal(y, x)

    def test_elu_negative_exponential(self):
        """ELU should be exp(x)-1 for negative inputs."""
        x = np.array([-1.0, -0.5, -0.1])
        y = elu_np(x)
        expected = np.exp(x) - 1.0

        np.testing.assert_array_almost_equal(y, expected)

    def test_elu_continuous_at_zero(self):
        """ELU should be continuous at x=0."""
        y_at_zero = elu_np(0.0)
        assert y_at_zero == 0.0

    def test_elu_bounds_negative_region(self):
        """ELU output for negative inputs should be in [-1, 0)."""
        x = np.linspace(-10.0, -0.001, 100)
        y = elu_np(x)

        assert np.all(y >= -1.0)
        assert np.all(y < 0.0)


class TestSigmoidFunction:
    """Tests for Sigmoid activation function."""

    def test_sigmoid_output_range(self):
        """Sigmoid should output values in [0, 1]."""
        x = np.linspace(-10, 10, 100)
        y = sigmoid_np(x)

        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_sigmoid_at_zero(self):
        """sigmoid(0) should be 0.5."""
        y = sigmoid_np(0.0)
        assert np.isclose(y, 0.5)

    def test_sigmoid_symmetry(self):
        """sigmoid(-x) + sigmoid(x) should equal 1."""
        x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        y = sigmoid_np(x)
        y_neg = sigmoid_np(-x)

        np.testing.assert_array_almost_equal(y + y_neg, np.ones_like(x))

    def test_sigmoid_limits(self):
        """Sigmoid should approach 0 as x→-∞ and 1 as x→+∞."""
        y_very_neg = sigmoid_np(-100.0)
        y_very_pos = sigmoid_np(100.0)

        assert y_very_neg < 1e-10
        assert y_very_pos > 1.0 - 1e-10


class TestTanhFunction:
    """Tests for Tanh activation function."""

    def test_tanh_output_range(self):
        """Tanh should output values in [-1, 1]."""
        x = np.linspace(-10, 10, 100)
        y = tanh_np(x)

        assert np.all(y >= -1.0)
        assert np.all(y <= 1.0)

    def test_tanh_at_zero(self):
        """tanh(0) should be 0."""
        y = tanh_np(0.0)
        assert y == 0.0

    def test_tanh_odd_function(self):
        """Tanh should be odd: tanh(-x) = -tanh(x)."""
        x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        y = tanh_np(x)
        y_neg = tanh_np(-x)

        np.testing.assert_array_almost_equal(y_neg, -y)

    def test_tanh_limits(self):
        """Tanh should approach -1 as x→-∞ and 1 as x→+∞."""
        y_very_neg = tanh_np(-100.0)
        y_very_pos = tanh_np(100.0)

        assert y_very_neg < -1.0 + 1e-10
        assert y_very_pos > 1.0 - 1e-10


class TestDerivatives:
    """Tests for activation function derivatives."""

    def test_relu_derivative_properties(self):
        """ReLU derivative should be 0 for x<0 and 1 for x>0."""
        x_neg = np.array([-2.0, -1.0, -0.1])
        x_pos = np.array([0.1, 1.0, 2.0])

        dy_neg = drelu_np(x_neg)
        dy_pos = drelu_np(x_pos)

        np.testing.assert_array_almost_equal(dy_neg, np.zeros_like(x_neg))
        np.testing.assert_array_almost_equal(dy_pos, np.ones_like(x_pos))

    def test_sigmoid_derivative_range(self):
        """Sigmoid derivative should be in (0, 0.25] (except at extremes)."""
        x = np.linspace(-10, 10, 1000)  # Use finite range to avoid underflow
        dy = dsigmoid_np(x)

        assert np.all(dy > 0.0)
        assert np.all(dy <= 0.25 + 1e-10)

    def test_sigmoid_derivative_max_at_zero(self):
        """Sigmoid derivative should be maximum at x=0."""
        y_at_zero = dsigmoid_np(0.0)
        x_test = np.array([-1.0, -0.5, 0.5, 1.0])
        y_test = dsigmoid_np(x_test)

        assert y_at_zero > np.max(y_test)
        assert np.isclose(y_at_zero, 0.25)

    def test_tanh_derivative_range(self):
        """Tanh derivative should be in (0, 1] (except at extremes)."""
        x = np.linspace(-10, 10, 1000)  # Use finite range to avoid underflow
        dy = dtanh_np(x)

        assert np.all(dy > 0.0)
        assert np.all(dy <= 1.0 + 1e-10)

    def test_tanh_derivative_max_at_zero(self):
        """Tanh derivative should be maximum at x=0."""
        y_at_zero = dtanh_np(0.0)
        x_test = np.array([-1.0, -0.5, 0.5, 1.0])
        y_test = dtanh_np(x_test)

        assert y_at_zero > np.max(y_test)
        assert np.isclose(y_at_zero, 1.0)


class TestFunctionMonotonicity:
    """Test monotonicity properties of activation functions."""

    def test_relu_monotonic_increasing(self):
        """ReLU should be monotonically increasing."""
        x = np.linspace(-10, 10, 100)
        y = relu_np(x)

        dy = np.diff(y)
        assert np.all(dy >= 0.0)

    def test_sigmoid_monotonic_increasing(self):
        """Sigmoid should be monotonically increasing."""
        x = np.linspace(-10, 10, 100)
        y = sigmoid_np(x)

        dy = np.diff(y)
        assert np.all(dy > 0.0)

    def test_tanh_monotonic_increasing(self):
        """Tanh should be monotonically increasing."""
        x = np.linspace(-10, 10, 100)
        y = tanh_np(x)

        dy = np.diff(y)
        assert np.all(dy > 0.0)

    def test_leakyrelu_monotonic_increasing(self):
        """LeakyReLU should be monotonically increasing."""
        x = np.linspace(-10, 10, 100)
        y = leakyrelu_np(x, negative_slope=0.01)

        dy = np.diff(y)
        assert np.all(dy > 0.0)


class TestFunctionCompositionProperties:
    """Test mathematical properties and relationships between functions."""

    def test_relu_vs_leakyrelu(self):
        """LeakyReLU should be <= ReLU in magnitude for negative inputs."""
        x = np.array([-2.0, -1.0, -0.5])
        y_relu = relu_np(x)
        y_leakyrelu = leakyrelu_np(x, negative_slope=0.01)

        # For negative inputs: relu is always 0, leakyrelu is negative
        # So leakyrelu < relu
        assert np.all(y_leakyrelu < y_relu)

    def test_tanh_sigmoid_relationship(self):
        """tanh(x) = 2*sigmoid(2x) - 1."""
        x = np.linspace(-2, 2, 50)
        y_tanh = tanh_np(x)
        y_from_sigmoid = 2.0 * sigmoid_np(2.0 * x) - 1.0

        np.testing.assert_array_almost_equal(y_tanh, y_from_sigmoid, decimal=10)

    def test_elu_vs_relu(self):
        """ELU should allow negative outputs (unlike ReLU)."""
        x = np.array([-2.0, -1.0, -0.1])
        y_elu = elu_np(x)
        y_relu = relu_np(x)

        # ELU outputs should be negative for negative inputs
        assert np.all(y_elu < 0.0)
        # ReLU outputs should be zero
        assert np.all(y_relu == 0.0)
        # ELU allows negative (in range [-1, 0)), ReLU only gives 0
        assert np.all(y_elu <= y_relu)


class TestELUDerivative:
    """Tests for ELU first derivative (delu_np)."""

    def test_delu_positive_inputs_equal_one(self):
        """ELU derivative should be 1 for all positive inputs."""
        x = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        dy = delu_np(x)

        np.testing.assert_array_almost_equal(dy, np.ones_like(x))

    def test_delu_negative_inputs_equal_exp(self):
        """ELU derivative should be exp(x) for negative inputs."""
        x = np.array([-3.0, -1.0, -0.5, -0.1])
        dy = delu_np(x)
        expected = np.exp(x)

        np.testing.assert_array_almost_equal(dy, expected)

    def test_delu_scalar_positive(self):
        """delu_np should return 1.0 for a positive scalar."""
        assert delu_np(2.0) == 1.0

    def test_delu_scalar_negative(self):
        """delu_np should return exp(x) for a negative scalar."""
        x = -1.0
        result = delu_np(x)
        assert np.isclose(result, np.exp(x))

    def test_delu_positive_everywhere(self):
        """ELU derivative should be strictly positive for all inputs."""
        x = np.linspace(-5.0, 5.0, 100)
        dy = delu_np(x)

        assert np.all(dy > 0.0)

    def test_delu_continuous_at_zero(self):
        """ELU derivative should approach 1 from both sides at x=0."""
        dy_pos = delu_np(1e-6)
        dy_neg = delu_np(-1e-6)

        assert np.isclose(dy_pos, 1.0, atol=1e-5)
        assert np.isclose(dy_neg, np.exp(-1e-6), atol=1e-5)


class TestLeakyReLUDerivative:
    """Tests for Leaky ReLU first derivative (dleakyrelu_np)."""

    @pytest.mark.parametrize(
        ("x_arr", "negative_slope", "expected"),
        [
            (np.array([0.1, 1.0, 2.0]), 0.01, np.array([1.0, 1.0, 1.0])),
            (np.array([-2.0, -0.5, -0.1]), 0.01, np.array([0.01, 0.01, 0.01])),
            (np.array([-1.0, 0.5]), 0.1, np.array([0.1, 1.0])),
        ],
    )
    def test_dleakyrelu_output_values(
        self, x_arr: np.ndarray, negative_slope: float, expected: np.ndarray
    ):
        """dleakyrelu_np returns 1 for positive inputs and negative_slope for negative."""
        dy = dleakyrelu_np(x_arr, negative_slope=negative_slope)

        np.testing.assert_array_almost_equal(dy, expected)

    def test_dleakyrelu_default_slope(self):
        """dleakyrelu_np default slope is 0.01."""
        x = np.array([-1.0, 1.0])
        dy = dleakyrelu_np(x)

        np.testing.assert_array_almost_equal(dy, np.array([0.01, 1.0]))

    def test_dleakyrelu_positive_everywhere(self):
        """Leaky ReLU derivative should be strictly positive for any positive slope."""
        x = np.linspace(-5.0, 5.0, 100)
        dy = dleakyrelu_np(x, negative_slope=0.01)

        assert np.all(dy > 0.0)


class TestSecondDerivativeSigmoid:
    """Tests for sigmoid second derivative (ddsigmoid_np)."""

    def test_ddsigmoid_zero_at_origin(self):
        """Second derivative of sigmoid should be 0 at x=0."""
        result = ddsigmoid_np(0.0)
        assert np.isclose(result, 0.0)

    def test_ddsigmoid_positive_for_negative_x(self):
        """Second derivative of sigmoid should be positive for x < 0."""
        x = np.array([-3.0, -2.0, -1.0, -0.5])
        ddy = ddsigmoid_np(x)

        assert np.all(ddy > 0.0)

    def test_ddsigmoid_negative_for_positive_x(self):
        """Second derivative of sigmoid should be negative for x > 0."""
        x = np.array([0.5, 1.0, 2.0, 3.0])
        ddy = ddsigmoid_np(x)

        assert np.all(ddy < 0.0)

    def test_ddsigmoid_antisymmetric(self):
        """Second derivative of sigmoid is an odd function: ddsigmoid(-x) = -ddsigmoid(x)."""
        x = np.array([0.5, 1.0, 1.5, 2.0])
        ddy_pos = ddsigmoid_np(x)
        ddy_neg = ddsigmoid_np(-x)

        np.testing.assert_array_almost_equal(ddy_neg, -ddy_pos)

    def test_ddsigmoid_numerical_value_at_one(self):
        """Second derivative of sigmoid at x=1 matches analytical value."""
        x = 1.0
        s = 1.0 / (1.0 + np.exp(-x))
        expected = s * (1.0 - s) * (1.0 - 2.0 * s)
        result = ddsigmoid_np(x)

        assert np.isclose(result, expected)


class TestSecondDerivativeTanh:
    """Tests for tanh second derivative (ddtanh_np)."""

    def test_ddtanh_zero_at_origin(self):
        """Second derivative of tanh should be 0 at x=0."""
        result = ddtanh_np(0.0)
        assert np.isclose(result, 0.0)

    def test_ddtanh_negative_for_positive_x(self):
        """Second derivative of tanh should be negative for x > 0."""
        x = np.array([0.5, 1.0, 2.0, 3.0])
        ddy = ddtanh_np(x)

        assert np.all(ddy < 0.0)

    def test_ddtanh_positive_for_negative_x(self):
        """Second derivative of tanh should be positive for x < 0."""
        x = np.array([-3.0, -2.0, -1.0, -0.5])
        ddy = ddtanh_np(x)

        assert np.all(ddy > 0.0)

    def test_ddtanh_antisymmetric(self):
        """Second derivative of tanh is an odd function: ddtanh(-x) = -ddtanh(x)."""
        x = np.array([0.5, 1.0, 1.5, 2.0])
        ddy_pos = ddtanh_np(x)
        ddy_neg = ddtanh_np(-x)

        np.testing.assert_array_almost_equal(ddy_neg, -ddy_pos)

    def test_ddtanh_numerical_value_at_one(self):
        """Second derivative of tanh at x=1 matches analytical value."""
        x = 1.0
        expected = -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2)
        result = ddtanh_np(x)

        assert np.isclose(result, expected)
