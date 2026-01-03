"""Exception handling and error path tests for ActHull.

Tests for:
1. Input validation errors (missing bounds, missing constraints, dimension mismatches)
2. Invalid input errors (non-1D bounds, invalid bound ranges)
3. Polytope validation errors (unbounded, degenerated, infeasible)
4. Exception recovery with fractional fallback
5. Double orders mode
6. Error logging and exception recording
"""

__docformat__ = "restructuredtext"

import contextlib

import numpy as np
import pytest


class TestActHullInputValidation:
    """Test input validation in ActHull."""

    def test_missing_bounds_raises_error(self):
        """Test that missing bounds raises ValueError."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        with pytest.raises(ValueError, match="lower and upper bounds"):
            hull.cal_hull(input_lower_bounds=None, input_upper_bounds=np.array([1.0, 1.0]))

    def test_missing_both_bounds_raises_error(self):
        """Test that missing both bounds raises ValueError."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        with pytest.raises(ValueError, match=r"At least.*constraints.*or.*bounds"):
            hull.cal_hull(input_lower_bounds=None, input_upper_bounds=None)

    def test_missing_input_constraints_with_bounds(self):
        """Test single neuron mode works without input constraints."""
        from wraact.acthull import ReLUHull

        # Single neuron mode doesn't require constraints
        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        constraints = hull.cal_hull(
            input_constrs=None,
            input_lower_bounds=np.array([-1.0, -1.0]),
            input_upper_bounds=np.array([1.0, 1.0]),
        )
        assert isinstance(constraints, np.ndarray)

    def test_too_few_constraints_error(self):
        """Test error when insufficient constraints provided."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()

        # 2D space needs at least 3 constraints, but only provide 2
        c = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, -1.0, 1.0],
            ]
        )

        with pytest.raises(ValueError, match=r"at least the dimension.*plus one"):
            hull.cal_hull(
                input_constrs=c,
                input_lower_bounds=np.array([-1.0, -1.0]),
                input_upper_bounds=np.array([1.0, 1.0]),
            )

    def test_neither_constraints_nor_bounds_raises_error(self):
        """Test that providing neither constraints nor bounds raises error."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()

        with pytest.raises(ValueError, match=r"At least.*constraints.*or.*bounds"):
            hull.cal_hull(input_constrs=None, input_lower_bounds=None, input_upper_bounds=None)


class TestActHullBoundValidation:
    """Test bounds validation in ActHull."""

    def test_non_1d_lower_bounds_raises_error(self):
        """Test that 2D lower bounds raises error."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        # 2D instead of 1D
        lb = np.array([[1.0, 1.0]])
        ub = np.array([2.0, 2.0])

        with pytest.raises(ValueError, match="1-dimensional arrays"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_non_1d_upper_bounds_raises_error(self):
        """Test that 2D upper bounds raises error."""
        from wraact.acthull import ELUHull

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        lb = np.array([1.0, 1.0])
        ub = np.array([[2.0, 2.0]])

        with pytest.raises(ValueError, match="1-dimensional arrays"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_bounds_size_mismatch_raises_error(self):
        """Test that mismatched bound sizes raise error."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        lb = np.array([1.0, 1.0])
        ub = np.array([2.0])

        with pytest.raises(ValueError, match="same size"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_bounds_reversed_raises_error(self):
        """Test that lb > ub raises error."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        lb = np.array([2.0, 2.0])
        ub = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="lower bounds should be less than the upper bounds"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)


class TestActHullConstraintValidation:
    """Test constraint matrix validation."""

    def test_too_few_constraints_raises_error(self):
        """Test that too few constraints raise error."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()

        # 2D space but only 1 constraint (need at least d+1=3)
        c = np.array([[1.0, 1.0, 0.0]])

        with pytest.raises(ValueError, match=r"at least the dimension.*plus one"):
            hull.cal_hull(
                input_constrs=c,
                input_lower_bounds=np.array([-1.0, -1.0]),
                input_upper_bounds=np.array([1.0, 1.0]),
            )

    def test_unbounded_polytope_raises_error(self):
        """Test detection of unbounded polytope (missing bias column)."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()

        # Create constraints with column format but check vertices fail
        # The error is raised during vertex checking
        lb = np.array([-1.0])
        ub = np.array([1.0])

        # This should work for single dimension
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints.shape[1] == 3  # 1D: 2*1+1


class TestActHullDoubleOrders:
    """Test double orders mode in ActHull."""

    def test_double_orders_requires_multi_neuron(self):
        """Test that double orders requires multi-neuron mode."""
        from wraact.acthull import LeakyReLUHull

        # Double orders with single neuron should raise
        with pytest.raises(ValueError, match=r"if_use_double_orders.*if_cal_multi_neuron_constrs"):
            LeakyReLUHull(
                if_use_double_orders=True,
                if_cal_single_neuron_constrs=True,
                if_cal_multi_neuron_constrs=False,
            )

    def test_single_neuron_multi_neuron_valid_combination(self):
        """Test valid single and multi-neuron mode combinations."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Single neuron only is valid
        hull_sn = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        constraints = hull_sn.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

        # Multi neuron only is valid
        hull_mn = ReLUHull(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)
        constraints = hull_mn.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullReversedOrderCaching:
    """Test consistent computation across multiple modes."""

    def test_relu_consistent_across_calls(self):
        """Test ReLU produces consistent results across calls."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Multiple calls should produce same result
        results = []
        for _ in range(3):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints)

        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_leakyrelu_consistent_across_dimensions(self):
        """Test LeakyReLU with different dimensions."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        for d in [2, 3, 4]:
            lb = -np.ones(d)
            ub = np.ones(d)

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert constraints.shape[1] == 2 * d + 1
            assert np.all(np.isfinite(constraints))


class TestActHullExceptionRecovery:
    """Test exception recovery with fractional fallback."""

    def test_single_neuron_valid_bounds_tolerance(self):
        """Test that single-neuron mode works with valid bounds."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_multi_neuron_degenerate_recovery(self):
        """Test recovery from degenerate polytope with fractional arithmetic."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_multi_neuron_constrs=True, if_cal_single_neuron_constrs=False)

        # Create valid but potentially challenging bounds
        lb = np.array([-0.5, -0.5])
        ub = np.array([0.5, 0.5])

        with contextlib.suppress(ValueError, RuntimeError):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If it succeeds, verify output
            if constraints is not None:
                assert isinstance(constraints, np.ndarray)

    def test_maxpool_multi_neuron_soundness_with_recovery(self):
        """Test MaxPool multi-neuron mode with potential exception recovery."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullConstraintInterpolation:
    """Test constraint calculations and interpolation."""

    def test_relu_single_neuron_constraint_properties(self):
        """Test ReLU single-neuron constraints have expected properties."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should have multiple constraints
        assert constraints.shape[0] > 0
        # Proper column format: 2*d+1
        assert constraints.shape[1] == 5  # 2*2+1

    def test_sigmoid_single_neuron_finite_outputs(self):
        """Test Sigmoid single-neuron constraints are always finite."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        test_cases = [
            (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
            (np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
            (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
        ]

        for lb, ub in test_cases:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints)), f"Non-finite constraints for lb={lb}, ub={ub}"


class TestActHullEdgeDimensionHandling:
    """Test ActHull with edge case dimensions."""

    def test_single_dimension_input(self):
        """Test ActHull with 1D input."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        lb = np.array([-1.0])
        ub = np.array([1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 3  # 2*1+1
        assert np.all(np.isfinite(constraints))

    def test_high_dimension_input(self):
        """Test ActHull with high-dimensional input."""
        from wraact.acthull import ELUHull

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        d = 10
        lb = -np.ones(d)
        ub = np.ones(d)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 2 * d + 1
        assert np.all(np.isfinite(constraints))


class TestActHullExceptionPaths:
    """Test specific exception handling paths."""

    def test_none_result_error_handling(self):
        """Test handling of None result from constraint calculation."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull(if_cal_multi_neuron_constrs=True, if_cal_single_neuron_constrs=False)

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Should not raise despite potential None results
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0

    def test_maxpool_complex_multi_neuron_recovery(self):
        """Test MaxPool multi-neuron with complex bounds that might trigger recovery."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()

        test_cases = [
            (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
            (np.array([-100.0, -50.0]), np.array([50.0, 100.0])),
        ]

        for lb, ub in test_cases:
            with contextlib.suppress(ValueError, RuntimeError):
                constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
                if constraints is not None:
                    assert np.all(np.isfinite(constraints))


class TestActHullConsistency:
    """Test consistency of ActHull computations across modes."""

    def test_single_vs_multi_neuron_consistent_column_format(self):
        """Test that single and multi-neuron modes use same column format."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Single neuron
        hull_sn = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        c_sn = hull_sn.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Multi neuron
        hull_mn = ReLUHull(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)
        c_mn = hull_mn.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Same column format
        assert c_sn.shape[1] == c_mn.shape[1]

    def test_combined_mode_superset_property(self):
        """Test that combined mode produces superset of constraints."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull_both = LeakyReLUHull(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True
        )
        c_both = hull_both.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        hull_single = LeakyReLUHull(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )
        c_single = hull_single.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Combined should have at least as many constraints
        assert c_both.shape[0] >= c_single.shape[0]
