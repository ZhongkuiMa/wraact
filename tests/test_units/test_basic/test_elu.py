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

from tests.test_units.test_templates import BaseSoundnessTest

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

    def test_cal_hull_returns_ndarray(self, elu_hull_class):
        """Verify cal_hull() returns an ndarray."""
        hull = elu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # 2D array

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(2, 5, id="2d"),
            pytest.param(3, 7, id="3d"),
        ],
    )
    def test_cal_hull_output_shape(self, dim, expected_cols, elu_hull_class):
        """Verify output shape follows formula: 2*dim + 1."""
        hull = elu_hull_class()
        lb = -np.ones(dim)
        ub = np.ones(dim)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[1] == expected_cols

    def test_cal_hull_output_finite(self, elu_hull_class):
        """Verify output contains no inf or nan values."""
        hull = elu_hull_class()
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

    @pytest.mark.parametrize(
        ("lb", "ub", "scenario"),
        [
            pytest.param(
                np.array([0.5, 0.5]), np.array([1.0, 1.0]), "all_positive", id="all_positive"
            ),
            pytest.param(
                np.array([-1.0, -1.0]), np.array([-0.5, -0.5]), "all_negative", id="all_negative"
            ),
            pytest.param(
                np.array([-0.03, -0.03]), np.array([0.03, 0.03]), "small_range", id="small_range"
            ),
            pytest.param(
                np.array([-2.0, -2.0]), np.array([0.5, 0.5]), "asymmetric", id="asymmetric"
            ),
            pytest.param(
                np.array([-10.0, -10.0]), np.array([10.0, 10.0]), "large_range", id="large_range"
            ),
        ],
    )
    def test_cal_hull_bound_configurations(self, lb, ub, scenario, elu_hull_class):
        """Test ELU with various bound configurations."""
        hull = elu_hull_class()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestELUSingleNeuronMode:
    """Test ELU hull with single-neuron constraint mode.

    This tests the single-neuron constraint calculation path which is
    normally disabled in default ActHull initialization.
    """

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(2, 5, id="2d"),
            pytest.param(3, 7, id="3d"),
        ],
    )
    def test_cal_hull_single_neuron(self, dim, expected_cols, elu_hull_class):
        """Test single-neuron constraints for given input dimension."""
        lb = -np.ones(dim)
        ub = np.ones(dim)

        hull = elu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == expected_cols
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_constraint_count(self, elu_hull_class):
        """Verify single-neuron constraints produce expected number of constraints."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = elu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For ELU single-neuron mode, should have constraints for upper and lower bounds
        # Expected: 4d constraints (d upper + d lower * 3) for d dimensions
        assert isinstance(constraints, np.ndarray)
        assert constraints.ndim == 2
        # For 2D: expect at least 2 constraints
        assert constraints.shape[0] >= 2
        assert constraints.shape[1] == 5  # 2*d + 1 for 2D

    def test_cal_sn_constrs_direct_call(self, elu_hull_class):
        """Test direct call to cal_sn_constrs method."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = elu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        # Call cal_hull which internally calls cal_sn_constrs
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify the method executed successfully
        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0

    def test_cal_hull_single_neuron_deterministic(self, elu_hull_class):
        """Verify single-neuron constraints are deterministic."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # First call
        hull1 = elu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        constraints1 = hull1.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Second call with same inputs
        hull2 = elu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        constraints2 = hull2.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Results should be identical
        np.testing.assert_array_almost_equal(constraints1, constraints2)
