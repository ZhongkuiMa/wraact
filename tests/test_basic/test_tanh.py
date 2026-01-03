"""Soundness tests for TanhHull using template pattern.

This demonstrates how to reuse the BaseSoundnessTest template for
a different activation function (Tanh - Hyperbolic Tangent).

Key Features of Tanh:
=====================
- S-shaped function: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- Output range: [-1, 1] (unlike Sigmoid which outputs [0, 1])
- Symmetric: tanh(-x) = -tanh(x) (odd function)
- Smooth derivatives: no kinks or discontinuities
- Often performs better than sigmoid in neural networks

Differences from Sigmoid:
- Output centered at 0 (symmetric bounds ideal)
- Output range [-1, 1] vs [0, 1] for sigmoid
- Zero-centered outputs often help with optimization
- Steeper gradients near 0 compared to sigmoid

Template Usage:
===============
This test class inherits from BaseSoundnessTest and only needs to:
1. Define the activation function
2. Specify which hull class to test
3. All soundness tests run automatically!
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from tests.test_templates import BaseSoundnessTest

# Import the actual hull class
from wraact.acthull import TanhHull


def tanh_np(x):
    """NumPy implementation of Tanh for testing.

    Args:
        x: Input value(s)

    Returns:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return np.tanh(x)


class TestTanhSoundness(BaseSoundnessTest):
    """Soundness tests for TanhHull.

    Reuses all tests from BaseSoundnessTest by implementing:
    1. activation_fn fixture: The actual function to test
    2. hull_class_to_test fixture: The hull class (TanhHull)

    All soundness tests are inherited and run automatically:
    - test_soundness_2d_box_monte_carlo
    - test_soundness_3d_box_monte_carlo
    - test_soundness_4d_box_monte_carlo
    - test_soundness_random_seeds (parametrized, 3 seeds)
    - test_hull_contains_actual_outputs
    - test_deterministic_computation
    - test_soundness_preserved_after_multiple_calls

    NOTE: DIAGNOSTIC FAILURES
    ========================
    Like Sigmoid, Tanh soundness tests may reveal constraint violations.
    These are expected and will be investigated/discussed with the team.
    """

    @pytest.fixture
    def activation_fn(self):
        """Return the Tanh function."""
        return tanh_np

    @pytest.fixture
    def hull_class_to_test(self):
        """Return the TanhHull class to test."""
        return TanhHull


class TestTanhBasicFunctionality:
    """Basic functionality tests for TanhHull."""

    def test_cal_hull_returns_ndarray(self):
        """Verify cal_hull() returns an ndarray."""
        from wraact.acthull import TanhHull

        hull = TanhHull()
        # Tanh is symmetric around 0, use symmetric bounds
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # 2D array

    def test_cal_hull_output_shape_2d(self):
        """Verify output shape follows formula: 2*dim + 1 = 5 for 2D."""
        from wraact.acthull import TanhHull

        hull = TanhHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 2D input: 2*2 + 1 = 5 columns
        assert result.shape[1] == 5

    def test_cal_hull_output_shape_3d(self):
        """Verify output shape follows formula: 2*dim + 1 = 7 for 3D."""
        from wraact.acthull import TanhHull

        hull = TanhHull()
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 3D input: 2*3 + 1 = 7 columns
        assert result.shape[1] == 7

    def test_cal_hull_output_finite(self):
        """Verify output contains no inf or nan values."""
        from wraact.acthull import TanhHull

        hull = TanhHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))

    def test_tanh_function_characteristics(self):
        """Verify Tanh function has expected characteristics."""
        # Test that tanh output is always in [-1, 1]
        x = np.linspace(-10, 10, 100)
        y = tanh_np(x)

        assert np.all(y >= -1.0)
        assert np.all(y <= 1.0)

    def test_tanh_symmetry(self):
        """Verify tanh odd function property: tanh(-x) = -tanh(x)."""
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

        y = tanh_np(x)
        y_neg = tanh_np(-x)

        # tanh(-x) should equal -tanh(x)
        np.testing.assert_array_almost_equal(y_neg, -y)

    def test_tanh_at_zero(self):
        """Verify tanh(0) = 0."""
        y_at_zero = tanh_np(0.0)
        assert np.isclose(y_at_zero, 0.0)

    def test_tanh_monotonicity(self):
        """Verify tanh is strictly increasing."""
        x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        y = tanh_np(x)

        # All differences should be positive (strictly increasing)
        dy = np.diff(y)
        assert np.all(dy > 0.0)

    def test_tanh_extreme_values(self):
        """Verify tanh behavior at extreme values."""
        # For very negative x, tanh should approach -1
        y_very_neg = tanh_np(-100.0)
        assert y_very_neg < -1.0 + 1e-10

        # For very positive x, tanh should approach 1
        y_very_pos = tanh_np(100.0)
        assert y_very_pos > 1.0 - 1e-10

    def test_tanh_vs_sigmoid_comparison(self):
        """Compare Tanh with Sigmoid to highlight differences."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_tanh = tanh_np(x)
        y_sigmoid = 1.0 / (1.0 + np.exp(-x))

        # Tanh output range is [-1, 1], Sigmoid is [0, 1]
        assert np.all(y_tanh >= -1.0)
        assert np.all(y_tanh <= 1.0)
        assert np.all(y_sigmoid >= 0.0)
        assert np.all(y_sigmoid <= 1.0)

        # Tanh is symmetric: tanh(0) = 0
        # Sigmoid is shifted: sigmoid(0) = 0.5
        assert y_tanh[2] == 0.0  # x = 0
        assert y_sigmoid[2] == 0.5  # x = 0

        # Relationship: tanh(x) = 2*sigmoid(2x) - 1
        # This verifies a mathematical identity
        y_from_sigmoid = 2 * (1.0 / (1.0 + np.exp(-2 * x))) - 1.0
        np.testing.assert_array_almost_equal(y_tanh, y_from_sigmoid, decimal=10)

    def test_tanh_vs_relu_comparison(self):
        """Compare Tanh with ReLU to highlight differences."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_tanh = tanh_np(x)
        y_relu = np.maximum(0, x)

        # Tanh outputs are in [-1, 1]
        assert np.all(y_tanh >= -1.0)
        assert np.all(y_tanh <= 1.0)

        # ReLU outputs are unbounded
        assert np.any(y_relu > 1.0)

        # Tanh is smooth everywhere, ReLU has a kink at 0
        # For negative inputs: tanh outputs negative, ReLU outputs 0
        neg_mask = x < 0
        assert np.all(y_tanh[neg_mask] < 0)
        assert np.all(y_relu[neg_mask] == 0)
