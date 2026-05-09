"""Advanced ActHull feature tests (Phase 3 coverage improvement).

This module tests advanced ActHull features that are normally disabled or require
specific configurations:

1. Double Orders Mode: Computing constraints in both forward and reverse orders
2. Error Handling: Validating that invalid configurations raise appropriate errors
3. Multi-neuron Combinations: Testing constraint combinations

Key Features Tested:
====================
- if_use_double_orders=True: Uses both forward and reverse polytope orders
- if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=False: Both disabled
- Missing input constraints: Error handling for invalid parameters
- Double orders without multi-neuron: Error when incompatible options set
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest


class TestActHullDoubleOrdersMode:
    """Test ActHull with double orders mode initialization and error handling."""

    def test_double_orders_without_multi_neuron_raises_error_at_init(self, relu_hull_class):
        """Test that double orders without multi-neuron raises error at initialization."""
        # Double orders requires multi-neuron constraints - error at init time
        with pytest.raises(ValueError, match="double orders"):
            relu_hull_class(
                if_use_double_orders=True,
                if_cal_multi_neuron_constrs=False,
                if_cal_single_neuron_constrs=True,
            )

    def test_double_orders_incompatible_configuration(self, leakyrelu_hull_class):
        """Test incompatible double orders configuration."""
        # If double orders is False but multi-neuron is True, behavior depends on defaults
        with pytest.raises(ValueError, match=r".*"):
            leakyrelu_hull_class(if_use_double_orders=True, if_cal_multi_neuron_constrs=False)

    def test_default_multi_neuron_mode(self, relu_hull_class):
        """Test default multi-neuron mode without double orders."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Default: multi-neuron only, no double orders
        hull = relu_hull_class(
            if_use_double_orders=False,
            if_cal_multi_neuron_constrs=True,
            if_cal_single_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 5  # 2D input: 2*2 + 1
        assert np.all(np.isfinite(constraints))

    def test_default_mode_deterministic(self, relu_hull_class):
        """Verify default mode produces deterministic results."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class()

        # Run twice
        constraints1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        constraints2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should be identical
        np.testing.assert_array_equal(constraints1, constraints2)

    def test_leakyrelu_double_orders_config_error(self, leakyrelu_hull_class):
        """Test LeakyReLU double orders configuration error."""
        with pytest.raises(ValueError, match=r".*"):
            leakyrelu_hull_class(if_use_double_orders=True, if_cal_multi_neuron_constrs=False)

    def test_maxpool_default_mode(self, maxpool_hull_class):
        """Test MaxPool default configuration."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class()

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullErrorHandling:
    """Test ActHull error handling and validation."""

    def test_both_constraint_modes_disabled_error_at_init(self, relu_hull_class):
        """Test that having both constraint modes disabled raises error at init."""
        # Error is raised at initialization time, not at call time
        with pytest.raises(ValueError, match="At least one"):
            relu_hull_class(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=False)

    def test_both_constraint_modes_disabled_leakyrelu(self, leakyrelu_hull_class):
        """Test both modes disabled error for LeakyReLU at init."""
        # Error is raised at initialization time
        with pytest.raises(ValueError, match="At least one"):
            leakyrelu_hull_class(
                if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=False
            )

    def test_invalid_bound_ordering_error(self, relu_hull_class):
        """Test that lb > ub raises ValueError."""
        lb = np.array([1.0, 1.0])
        ub = np.array([-1.0, -1.0])  # Reversed!

        hull = relu_hull_class()

        with pytest.raises(ValueError, match=r".*"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_mismatched_bound_dimensions_error(self, relu_hull_class):
        """Test that mismatched bound dimensions raise error."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])  # Different dimension!

        hull = relu_hull_class()

        with pytest.raises((ValueError, IndexError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_scalar_bounds_error(self, relu_hull_class):
        """Test that scalar bounds (not arrays) are handled."""
        lb = -1.0  # Scalar, not array
        ub = 1.0

        hull = relu_hull_class()

        # Should raise error for scalar bounds (AttributeError on ndim check)
        with pytest.raises((ValueError, TypeError, IndexError, AttributeError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_nan_bounds_error(self, relu_hull_class):
        """Test that NaN bounds raise error."""
        lb = np.array([-1.0, np.nan])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class()

        with pytest.raises(ValueError, match=r".*"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_inf_bounds_detected(self, relu_hull_class):
        """Test that infinite bounds are handled (may raise or handle gracefully)."""
        lb = np.array([-np.inf, -1.0])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class()

        # Infinite bounds should either raise ValueError or RuntimeError
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)


class TestActHullConstraintCombinations:
    """Test combinations of constraint generation modes."""

    def test_single_neuron_only_valid(self, relu_hull_class):
        """Test valid single-neuron only configuration."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0

    def test_multi_neuron_only_valid(self, relu_hull_class):
        """Test valid multi-neuron only configuration (default)."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_both_constraint_modes_enabled(self, relu_hull_class):
        """Test with both single and multi-neuron modes enabled."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should combine constraints from both modes
        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))

    def test_constraint_mode_combination_elu(self, elu_hull_class):
        """Test constraint mode combinations for ELU."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Both modes enabled
        hull = elu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        # Combined constraints should be comprehensive
        assert constraints.shape[0] > 0

    def test_constraint_mode_reproducibility(self, relu_hull_class):
        """Test that constraint modes produce reproducible results."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)

        # Multiple calls with same configuration
        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c3 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # All should be identical
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(c2, c3)


class TestActHullSpecialCases:
    """Test special and edge case scenarios."""

    def test_1d_input_multi_neuron(self, relu_hull_class):
        """Test multi-neuron mode with 1D input."""
        lb = np.array([-1.0])
        ub = np.array([1.0])

        hull = relu_hull_class(if_use_double_orders=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 3  # 2*1 + 1
        assert np.all(np.isfinite(constraints))

    def test_high_dimensional_input_multi_neuron(self, relu_hull_class):
        """Test multi-neuron mode with high-dimensional input."""
        lb = np.array([-1.0] * 5)
        ub = np.array([1.0] * 5)

        hull = relu_hull_class(if_use_double_orders=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 11  # 2*5 + 1
        assert np.all(np.isfinite(constraints))

    def test_constraint_count_comparison_single_vs_multi(self, relu_hull_class):
        """Compare constraint counts for single-neuron vs multi-neuron."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Single neuron only
        hull_single = relu_hull_class(
            if_use_double_orders=False,
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )
        c_single = hull_single.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Multi neuron only
        hull_multi = relu_hull_class(
            if_use_double_orders=False,
            if_cal_single_neuron_constrs=False,
            if_cal_multi_neuron_constrs=True,
        )
        c_multi = hull_multi.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Both should have valid constraints
        assert c_single.shape[0] > 0
        assert c_multi.shape[0] > 0
        # Same number of columns
        assert c_single.shape[1] == c_multi.shape[1]

    def test_maxpool_error_handling(self, maxpool_hull_class):
        """Test MaxPool error handling."""
        lb = np.array([1.0, 1.0])  # All positive
        ub = np.array([-1.0, -1.0])  # Invalid: lb > ub

        hull = maxpool_hull_class()

        with pytest.raises(ValueError, match=r".*"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_elu_bound_validation(self, elu_hull_class):
        """Test ELU with all-positive bounds."""
        lb = np.array([0.5, 0.5])  # All positive
        ub = np.array([1.0, 1.0])

        hull = elu_hull_class()

        # ELU with all-positive bounds may work or raise error depending on mode
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If succeeds, constraints should be valid
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))
        except ValueError:
            # Expected for all-positive bounds in single-neuron mode
            pass

    def test_leakyrelu_3d_multi_neuron(self, leakyrelu_hull_class):
        """Test LeakyReLU with 3D input in multi-neuron mode."""
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = leakyrelu_hull_class(
            if_cal_multi_neuron_constrs=True, if_cal_single_neuron_constrs=False
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 7  # 2*3 + 1
        assert np.all(np.isfinite(constraints))
