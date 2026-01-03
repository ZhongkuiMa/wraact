"""Soundness tests for SigmoidHull using template pattern.

This demonstrates how to reuse the BaseSoundnessTest template for
a different activation function (Sigmoid - S-shaped).

Key Features of Sigmoid:
========================
- S-shaped function: sigmoid(x) = 1 / (1 + exp(-x))
- Output range: [0, 1] (unlike ReLU/ELU which are unbounded)
- Symmetric: sigmoid(-x) = 1 - sigmoid(x)
- Smooth derivatives: no kinks or discontinuities
- Historically popular in neural networks, though less common in modern deep learning

Differences from ReLU-like functions:
- Output bounded to [0, 1]
- Requires different bound ranges for meaningful testing
- More complex hull structure due to exponential nature

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
from wraact.acthull import SigmoidHull


def sigmoid_np(x):
    """NumPy implementation of Sigmoid for testing.

    Args:
        x: Input value(s)

    Returns:
        1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


class TestSigmoidSoundness(BaseSoundnessTest):
    """Soundness tests for SigmoidHull.

    Reuses all tests from BaseSoundnessTest by implementing:
    1. activation_fn fixture: The actual function to test
    2. hull_class_to_test fixture: The hull class (SigmoidHull)

    All soundness tests are inherited and run automatically:
    - test_soundness_2d_box_monte_carlo
    - test_soundness_3d_box_monte_carlo
    - test_soundness_4d_box_monte_carlo
    - test_soundness_random_seeds (parametrized, 3 seeds)
    - test_hull_contains_actual_outputs
    - test_deterministic_computation
    - test_soundness_preserved_after_multiple_calls

    KNOWN ISSUE: SOUNDNESS VIOLATIONS
    ==================================
    These soundness tests FAIL due to a bug in the DLP constraint generation
    for the sigmoid hull. The issue is in _construct_dlp_case3 where the parallel
    tangent line intercept is computed incorrectly, resulting in overly tight
    constraints that cut off valid sigmoid points.

    Root Cause:
    - The tangent line computation returns correct intercept (~0.4929)
    - But the constraint matrix stores intercept (~0.5), which is incorrect
    - This causes all random points to violate the constraint by ~0.007
    - The difference is (yl + yu) / 2 vs actual tangent line intercept

    The fix requires correcting how the intercept is propagated through the
    DLP constraint construction in wraact/acthull/_sshape.py _construct_dlp_case3.
    """

    @pytest.fixture
    def activation_fn(self):
        """Return the Sigmoid function."""
        return sigmoid_np

    @pytest.fixture
    def hull_class_to_test(self):
        """Return the SigmoidHull class to test."""
        return SigmoidHull


class TestSigmoidBasicFunctionality:
    """Basic functionality tests for SigmoidHull."""

    def test_cal_hull_returns_ndarray(self):
        """Verify cal_hull() returns an ndarray."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()
        # Sigmoid is symmetric around 0, use symmetric bounds
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # 2D array

    def test_cal_hull_output_shape_2d(self):
        """Verify output shape follows formula: 2*dim + 1 = 5 for 2D."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 2D input: 2*2 + 1 = 5 columns
        assert result.shape[1] == 5

    def test_cal_hull_output_shape_3d(self):
        """Verify output shape follows formula: 2*dim + 1 = 7 for 3D."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 3D input: 2*3 + 1 = 7 columns
        assert result.shape[1] == 7

    def test_cal_hull_output_finite(self):
        """Verify output contains no inf or nan values."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))

    def test_sigmoid_function_characteristics(self):
        """Verify Sigmoid function has expected characteristics."""
        # Test that sigmoid output is always in [0, 1]
        x = np.linspace(-10, 10, 100)
        y = sigmoid_np(x)

        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_sigmoid_symmetry(self):
        """Verify sigmoid symmetry property: sigmoid(-x) = 1 - sigmoid(x)."""
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

        y = sigmoid_np(x)
        y_neg = sigmoid_np(-x)

        # sigmoid(-x) + sigmoid(x) should equal 1
        np.testing.assert_array_almost_equal(y + y_neg, np.ones_like(x))

    def test_sigmoid_at_zero(self):
        """Verify sigmoid(0) = 0.5."""
        y_at_zero = sigmoid_np(0.0)
        assert np.isclose(y_at_zero, 0.5)

    def test_sigmoid_monotonicity(self):
        """Verify sigmoid is strictly increasing."""
        x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        y = sigmoid_np(x)

        # All differences should be positive (strictly increasing)
        dy = np.diff(y)
        assert np.all(dy > 0.0)

    def test_sigmoid_extreme_values(self):
        """Verify sigmoid behavior at extreme values."""
        # For very negative x, sigmoid should approach 0
        y_very_neg = sigmoid_np(-100.0)
        assert y_very_neg < 1e-10

        # For very positive x, sigmoid should approach 1
        y_very_pos = sigmoid_np(100.0)
        assert y_very_pos > 1.0 - 1e-10

    def test_sigmoid_vs_relu_comparison(self):
        """Compare Sigmoid with ReLU to highlight differences."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_sigmoid = sigmoid_np(x)
        y_relu = np.maximum(0, x)

        # Sigmoid outputs are always in [0, 1]
        assert np.all(y_sigmoid >= 0.0)
        assert np.all(y_sigmoid <= 1.0)

        # ReLU outputs are unbounded
        assert np.any(y_relu > 1.0)

        # For x >= 0, sigmoid is bounded while ReLU grows unbounded
        pos_mask = x >= 0
        assert np.all(y_sigmoid[pos_mask] <= 1.0)
        assert np.any(y_relu[pos_mask] > 1.0)

        # Sigmoid is smooth everywhere, ReLU has a kink at 0
        # (this would be more formally tested with derivatives)
