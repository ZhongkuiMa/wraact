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

    def test_double_orders_without_multi_neuron_raises_error_at_init(self):
        """Test that double orders without multi-neuron raises error at initialization."""
        from wraact.acthull import ReLUHull

        # Double orders requires multi-neuron constraints - error at init time
        with pytest.raises(ValueError, match="double orders"):
            ReLUHull(
                if_use_double_orders=True,
                if_cal_multi_neuron_constrs=False,
                if_cal_single_neuron_constrs=True,
            )

    def test_double_orders_incompatible_configuration(self):
        """Test incompatible double orders configuration."""
        from wraact.acthull import LeakyReLUHull

        # If double orders is False but multi-neuron is True, behavior depends on defaults
        with pytest.raises(ValueError, match=r".*"):
            LeakyReLUHull(if_use_double_orders=True, if_cal_multi_neuron_constrs=False)

    def test_default_multi_neuron_mode(self):
        """Test default multi-neuron mode without double orders."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Default: multi-neuron only, no double orders
        hull = ReLUHull(
            if_use_double_orders=False,
            if_cal_multi_neuron_constrs=True,
            if_cal_single_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 5  # 2D input: 2*2 + 1
        assert np.all(np.isfinite(constraints))

    def test_default_mode_deterministic(self):
        """Verify default mode produces deterministic results."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull()

        # Run twice
        constraints1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        constraints2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should be identical
        np.testing.assert_array_equal(constraints1, constraints2)

    def test_leakyrelu_double_orders_config_error(self):
        """Test LeakyReLU double orders configuration error."""
        from wraact.acthull import LeakyReLUHull

        with pytest.raises(ValueError, match=r".*"):
            LeakyReLUHull(if_use_double_orders=True, if_cal_multi_neuron_constrs=False)

    def test_maxpool_default_mode(self):
        """Test MaxPool default configuration."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull()

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullErrorHandling:
    """Test ActHull error handling and validation."""

    def test_both_constraint_modes_disabled_error_at_init(self):
        """Test that having both constraint modes disabled raises error at init."""
        from wraact.acthull import ReLUHull

        # Error is raised at initialization time, not at call time
        with pytest.raises(ValueError, match="At least one"):
            ReLUHull(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=False)

    def test_both_constraint_modes_disabled_leakyrelu(self):
        """Test both modes disabled error for LeakyReLU at init."""
        from wraact.acthull import LeakyReLUHull

        # Error is raised at initialization time
        with pytest.raises(ValueError, match="At least one"):
            LeakyReLUHull(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=False)

    def test_invalid_bound_ordering_error(self):
        """Test that lb > ub raises ValueError."""
        from wraact.acthull import ReLUHull

        lb = np.array([1.0, 1.0])
        ub = np.array([-1.0, -1.0])  # Reversed!

        hull = ReLUHull()

        with pytest.raises(ValueError, match=r".*"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_mismatched_bound_dimensions_error(self):
        """Test that mismatched bound dimensions raise error."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])  # Different dimension!

        hull = ReLUHull()

        with pytest.raises((ValueError, IndexError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_scalar_bounds_error(self):
        """Test that scalar bounds (not arrays) are handled."""
        from wraact.acthull import ReLUHull

        lb = -1.0  # Scalar, not array
        ub = 1.0

        hull = ReLUHull()

        # Should raise error for scalar bounds (AttributeError on ndim check)
        with pytest.raises((ValueError, TypeError, IndexError, AttributeError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_nan_bounds_error(self):
        """Test that NaN bounds raise error."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, np.nan])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull()

        with pytest.raises(ValueError, match=r".*"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_inf_bounds_detected(self):
        """Test that infinite bounds are handled (may raise or handle gracefully)."""
        from wraact.acthull import ReLUHull

        lb = np.array([-np.inf, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull()

        # Infinite bounds should either raise ValueError or RuntimeError
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)


class TestActHullConstraintCombinations:
    """Test combinations of constraint generation modes."""

    def test_single_neuron_only_valid(self):
        """Test valid single-neuron only configuration."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0

    def test_multi_neuron_only_valid(self):
        """Test valid multi-neuron only configuration (default)."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_both_constraint_modes_enabled(self):
        """Test with both single and multi-neuron modes enabled."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should combine constraints from both modes
        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))

    def test_constraint_mode_combination_elu(self):
        """Test constraint mode combinations for ELU."""
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Both modes enabled
        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        # Combined constraints should be comprehensive
        assert constraints.shape[0] > 0

    def test_constraint_mode_reproducibility(self):
        """Test that constraint modes produce reproducible results."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)

        # Multiple calls with same configuration
        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c3 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # All should be identical
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(c2, c3)


class TestActHullSpecialCases:
    """Test special and edge case scenarios."""

    def test_1d_input_multi_neuron(self):
        """Test multi-neuron mode with 1D input."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0])
        ub = np.array([1.0])

        hull = ReLUHull(if_use_double_orders=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 3  # 2*1 + 1
        assert np.all(np.isfinite(constraints))

    def test_high_dimensional_input_multi_neuron(self):
        """Test multi-neuron mode with high-dimensional input."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0] * 5)
        ub = np.array([1.0] * 5)

        hull = ReLUHull(if_use_double_orders=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 11  # 2*5 + 1
        assert np.all(np.isfinite(constraints))

    def test_constraint_count_comparison_single_vs_multi(self):
        """Compare constraint counts for single-neuron vs multi-neuron."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Single neuron only
        hull_single = ReLUHull(
            if_use_double_orders=False,
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )
        c_single = hull_single.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Multi neuron only
        hull_multi = ReLUHull(
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

    def test_maxpool_error_handling(self):
        """Test MaxPool error handling."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([1.0, 1.0])  # All positive
        ub = np.array([-1.0, -1.0])  # Invalid: lb > ub

        hull = MaxPoolHull()

        with pytest.raises(ValueError, match=r".*"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_elu_bound_validation(self):
        """Test ELU with all-positive bounds."""
        from wraact.acthull import ELUHull

        lb = np.array([0.5, 0.5])  # All positive
        ub = np.array([1.0, 1.0])

        hull = ELUHull()

        # ELU with all-positive bounds may work or raise error depending on mode
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If succeeds, constraints should be valid
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))
        except ValueError:
            # Expected for all-positive bounds in single-neuron mode
            pass

    def test_leakyrelu_3d_multi_neuron(self):
        """Test LeakyReLU with 3D input in multi-neuron mode."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = LeakyReLUHull(if_cal_multi_neuron_constrs=True, if_cal_single_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 7  # 2*3 + 1
        assert np.all(np.isfinite(constraints))
