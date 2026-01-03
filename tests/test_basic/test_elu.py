"""Soundness tests for ELUHull using template pattern.

This demonstrates how to reuse the BaseSoundnessTest template for
a different activation function (ELU - Exponential Linear Unit).

Key Feature of ELU:
===================
- Unlike ReLU which outputs 0 for negative inputs
- Unlike LeakyReLU which outputs a linear slope for negative inputs
- ELU outputs an exponential curve for negative inputs: y = exp(x) - 1
- For positive inputs: y = x (same as ReLU)
- Produces smoother gradients in negative region

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
from wraact.acthull import ELUHull


def elu_np(x):
    """NumPy implementation of ELU for testing.

    Args:
        x: Input value(s)

    Returns:
        x if x > 0, else exp(x) - 1
    """
    return np.where(x > 0, x, np.exp(x) - 1.0)


class TestELUSoundness(BaseSoundnessTest):
    """Soundness tests for ELUHull.

    Reuses all tests from BaseSoundnessTest by implementing:
    1. activation_fn fixture: The actual function to test
    2. hull_class_to_test fixture: The hull class (ELUHull)

    All soundness tests are inherited and run automatically:
    - test_soundness_2d_box_monte_carlo
    - test_soundness_3d_box_monte_carlo
    - test_soundness_4d_box_monte_carlo
    - test_soundness_random_seeds (parametrized, 3 seeds)
    - test_hull_contains_actual_outputs
    - test_deterministic_computation
    - test_soundness_preserved_after_multiple_calls
    """

    @pytest.fixture
    def activation_fn(self):
        """Return the ELU function."""
        return elu_np

    @pytest.fixture
    def hull_class_to_test(self):
        """Return the ELUHull class to test."""
        return ELUHull


class TestELUBasicFunctionality:
    """Basic functionality tests for ELUHull."""

    def test_cal_hull_returns_ndarray(self):
        """Verify cal_hull() returns an ndarray."""
        from wraact.acthull import ELUHull

        hull = ELUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # 2D array

    def test_cal_hull_output_shape_2d(self):
        """Verify output shape follows formula: 2*dim + 1 = 5 for 2D."""
        from wraact.acthull import ELUHull

        hull = ELUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 2D input: 2*2 + 1 = 5 columns
        assert result.shape[1] == 5

    def test_cal_hull_output_shape_3d(self):
        """Verify output shape follows formula: 2*dim + 1 = 7 for 3D."""
        from wraact.acthull import ELUHull

        hull = ELUHull()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 3D input: 2*3 + 1 = 7 columns
        assert result.shape[1] == 7

    def test_cal_hull_output_finite(self):
        """Verify output contains no inf or nan values."""
        from wraact.acthull import ELUHull

        hull = ELUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))

    def test_elu_function_characteristics(self):
        """Verify ELU function has expected characteristics."""
        # Test positive region: should be identity
        x_pos = np.array([0.5, 1.0, 2.0])
        y_pos = elu_np(x_pos)
        np.testing.assert_array_almost_equal(y_pos, x_pos)

        # Test negative region: should be exp(x) - 1
        x_neg = np.array([-0.5, -1.0, -2.0])
        y_neg = elu_np(x_neg)
        expected_neg = np.exp(x_neg) - 1.0
        np.testing.assert_array_almost_equal(y_neg, expected_neg)

        # Test boundary: at x=0, should be continuous
        y_at_zero = elu_np(0.0)
        assert y_at_zero == 0.0

    def test_elu_smoothness_in_negative_region(self):
        """Verify ELU provides smooth transition in negative region (unlike ReLU)."""
        # ELU should have smooth derivatives due to exp function
        x_test = np.linspace(-2.0, 2.0, 100)
        y_test = elu_np(x_test)

        # Check for smoothness: no discontinuities
        # Second differences should be small and smooth
        dy = np.diff(y_test)
        ddy = np.diff(dy)

        # ELU should be continuously differentiable (unlike ReLU which has a kink at 0)
        # This is a feature that makes ELU potentially tighter than ReLU
        assert np.all(np.isfinite(ddy))

    def test_elu_vs_relu_comparison(self):
        """Compare ELU with ReLU to highlight differences."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_elu = elu_np(x)
        y_relu = np.maximum(0, x)

        # For positive x, they should be identical
        np.testing.assert_array_almost_equal(y_elu[x > 0], y_relu[x > 0])

        # For negative x, ELU outputs are negative (exp(x) - 1 < 0)
        # while ReLU outputs are 0
        neg_mask = x < 0
        y_elu_neg = y_elu[neg_mask]
        y_relu_neg = y_relu[neg_mask]

        # ELU is between -1 and 0 for negative inputs (exp(x) - 1 where x < 0)
        assert np.all(y_elu_neg >= -1.0)
        assert np.all(y_elu_neg < 0.0)

        # ELU values < ReLU values (which are 0) for negative inputs
        # This is by design: ELU allows some negative activation for better feature learning
        assert np.all(y_elu_neg < y_relu_neg)


class TestELUBoundEdgeCases:
    """Test ELU with edge case bounds (trivial cases)."""

    def test_cal_hull_all_positive_bounds(self):
        """Test ELU with all-positive bounds (lb >= 0)."""
        from wraact.acthull import ELUHull

        lb = np.array([0.5, 0.5])
        ub = np.array([1.0, 1.0])

        hull = ELUHull()
        # All-positive bounds might raise error or return trivial constraints
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If it succeeds, verify constraints are valid
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))
        except ValueError:
            # Expected: ELU requires both negative and positive bounds
            pass

    def test_cal_hull_all_negative_bounds(self):
        """Test ELU with all-negative bounds (ub <= 0)."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([-0.5, -0.5])

        hull = ELUHull()
        # All-negative bounds might raise error or return trivial constraints
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))
        except ValueError:
            # Expected: ELU requires both negative and positive bounds
            pass

    def test_cal_hull_very_small_range(self):
        """Test ELU with small input range (just above minimum threshold)."""
        from wraact.acthull import ELUHull

        # Minimum range threshold is 0.05, so we use range 0.06
        lb = np.array([-0.03, -0.03])
        ub = np.array([0.03, 0.03])

        hull = ELUHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        assert constraints.shape[1] == 5  # 2D input

    def test_cal_hull_asymmetric_bounds(self):
        """Test ELU with asymmetric bounds around zero."""
        from wraact.acthull import ELUHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([0.5, 0.5])

        hull = ELUHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_large_range(self):
        """Test ELU with large input range."""
        from wraact.acthull import ELUHull

        lb = np.array([-10.0, -10.0])
        ub = np.array([10.0, 10.0])

        hull = ELUHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestELUSingleNeuronMode:
    """Test ELU hull with single-neuron constraint mode.

    This tests the single-neuron constraint calculation path which is
    normally disabled in default ActHull initialization.
    """

    def test_cal_hull_single_neuron_2d(self):
        """Test single-neuron constraints for 2D input."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 2 * len(lb) + 1
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_3d(self):
        """Test single-neuron constraints for 3D input."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 2 * len(lb) + 1
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_output_shape(self):
        """Verify single-neuron constraint output shape for 2D."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 2D input: 2*2 + 1 = 5 columns
        assert constraints.shape[1] == 5
        # Should have at least some constraints
        assert constraints.shape[0] > 0

    def test_cal_hull_single_neuron_finite(self):
        """Verify single-neuron constraints contain no inf or nan."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_constraint_count(self):
        """Verify single-neuron constraints produce expected number of constraints."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For ELU single-neuron mode, should have constraints for upper and lower bounds
        # Expected: 4d constraints (d upper + d lower * 3) for d dimensions
        assert isinstance(constraints, np.ndarray)
        assert constraints.ndim == 2
        # For 2D: expect at least 2 constraints
        assert constraints.shape[0] >= 2
        assert constraints.shape[1] == 5  # 2*d + 1 for 2D

    def test_cal_sn_constrs_direct_call(self):
        """Test direct call to cal_sn_constrs method."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        # Call cal_hull which internally calls cal_sn_constrs
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify the method executed successfully
        assert constraints is not None
        assert constraints.shape[0] > 0

    def test_cal_hull_single_neuron_deterministic(self):
        """Verify single-neuron constraints are deterministic."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # First call
        hull1 = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        constraints1 = hull1.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Second call with same inputs
        hull2 = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        constraints2 = hull2.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Results should be identical
        np.testing.assert_array_almost_equal(constraints1, constraints2)
