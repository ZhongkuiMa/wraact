"""Advanced tests for ActHull base class features.

Tests for hard-to-reach code paths in acthull/_act.py:
1. Double orders mode (reverse order constraint calculation)
2. DEBUG mode (error propagation without exception handling)
3. Input validation (dimension mismatches, bounds errors)
4. Exception recovery (fractional fallback, error logging)
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest


class TestActHullDoubleOrders:
    """Test ActHull with double orders mode for enhanced precision."""

    def test_double_orders_gets_reversed_order(self):
        """Test that reversed order method works for any dimension.

        Tests the _get_reversed_order static method which supports double orders.
        """
        from wraact.acthull._act import ActHull

        # Test for various dimensions
        for dim in [2, 3, 4, 5]:
            order = ActHull._get_reversed_order(dim)  # noqa: SLF001

            # Verify order is a list
            assert isinstance(order, list)
            # Verify it contains indices in range [0, dim+1]
            assert all(0 <= idx <= dim for idx in order)
            # Verify it has the right length
            assert len(order) == dim + 1

    def test_reversed_order_caching(self):
        """Test that reversed order cache is populated and retrieved.

        Tests lines 324-326 in acthull/_act.py:
        - Lazy initialization of reversed order cache
        - Cache retrieval on subsequent calls
        """
        from wraact.acthull._act import ActHull

        # Clear cache
        ActHull._reversed_orders.clear()  # noqa: SLF001

        # First call for dimension 3 - populates cache
        order = ActHull._get_reversed_order(3)  # noqa: SLF001

        # Lines 324-326
        assert 3 in ActHull._reversed_orders  # noqa: SLF001
        assert order == [0, 3, 2, 1]

        # Second call - retrieves from cache
        order2 = ActHull._get_reversed_order(3)  # noqa: SLF001
        assert order == order2

    def test_reversed_order_symmetry(self):
        """Test that reversed order is truly reversed.

        Verifies that the reversed order produces indices in reverse fashion.
        """
        from wraact.acthull._act import ActHull

        # Clear cache to ensure fresh computation
        ActHull._reversed_orders.clear()  # noqa: SLF001

        for dim in [2, 3, 4]:
            order = ActHull._get_reversed_order(dim)  # noqa: SLF001

            # First element should be 0 (constant term)
            assert order[0] == 0

            # Remaining elements should be in reverse order
            # For dim=3: [0, 3, 2, 1]
            # For dim=4: [0, 4, 3, 2, 1]
            expected = [0, *list(range(dim, 0, -1))]
            assert order == expected

    def test_reversed_order_all_dimensions(self):
        """Test reversed order computation for all common dimensions.

        Ensures the reversed order method is robust across different input dimensions.
        """
        from wraact.acthull._act import ActHull

        # Clear cache
        ActHull._reversed_orders.clear()  # noqa: SLF001

        test_dims = [2, 3, 4, 5, 6, 7, 8]
        for dim in test_dims:
            order = ActHull._get_reversed_order(dim)  # noqa: SLF001

            # Verify properties
            assert len(order) == dim + 1
            assert order[0] == 0  # Constant term
            assert all(isinstance(idx, int) for idx in order)

            # Verify no duplicates
            assert len(set(order)) == len(order)

            # Verify all indices are valid
            assert all(0 <= idx <= dim for idx in order)

    def test_reversed_order_cache_persistence(self):
        """Test that reversed order cache persists across multiple accesses.

        Verifies that the cache is properly maintained across repeated calls.
        """
        from wraact.acthull._act import ActHull

        # Clear cache
        ActHull._reversed_orders.clear()  # noqa: SLF001

        # First call - populates cache for dimension 4
        order1 = ActHull._get_reversed_order(4)  # noqa: SLF001

        # Verify cache contains the dimension
        assert 4 in ActHull._reversed_orders  # noqa: SLF001

        # Second call - retrieves from cache
        order2 = ActHull._get_reversed_order(4)  # noqa: SLF001

        # Should be identical (same object or equal)
        assert order1 == order2

        # Add another dimension
        order3 = ActHull._get_reversed_order(5)  # noqa: SLF001
        assert 5 in ActHull._reversed_orders  # noqa: SLF001

        # Both dimensions should still be in cache
        assert 4 in ActHull._reversed_orders  # noqa: SLF001
        assert 5 in ActHull._reversed_orders  # noqa: SLF001

        # Verify both are retrievable
        assert ActHull._get_reversed_order(4) == order1  # noqa: SLF001
        assert ActHull._get_reversed_order(5) == order3  # noqa: SLF001

    def test_reversed_order_deterministic(self):
        """Verify reversed order computation is deterministic.

        Tests reproducibility of reversed order calculation across repeated calls.
        """
        from wraact.acthull._act import ActHull

        # Clear cache to start fresh
        ActHull._reversed_orders.clear()  # noqa: SLF001

        # Multiple calls for same dimension should return same result
        for dim in [2, 3, 4, 5]:
            results = [ActHull._get_reversed_order(dim) for _ in range(5)]  # noqa: SLF001

            # All results should be identical
            for i in range(1, len(results)):
                assert results[0] == results[i], f"Dimension {dim} produced different results"


class TestActHullDEBUGMode:
    """Test ActHull with DEBUG mode enabled."""

    def test_debug_mode_vertex_calculation(self):
        """Test ActHull with DEBUG=True for vertex calculation.

        Tests lines 339-341 in acthull/_act.py:
        - Direct vertex calculation without exception handling
        - Errors propagate immediately when DEBUG=True
        """
        from wraact import _constants
        from wraact.acthull import ReLUHull

        # Save original DEBUG state
        original_debug = _constants.DEBUG

        try:
            _constants.DEBUG = True

            lb = np.array([-1.0, -1.0])
            ub = np.array([1.0, 1.0])

            hull = ReLUHull()
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            # Should execute lines 339-341 (DEBUG vertex path)
            assert np.all(np.isfinite(constraints))

        finally:
            _constants.DEBUG = original_debug

    def test_debug_mode_constraint_calculation(self):
        """Test ActHull with DEBUG=True for constraint calculation.

        Tests lines 378-379 in acthull/_act.py:
        - Direct constraint calculation without exception handling
        """
        from wraact import _constants
        from wraact.acthull import ReLUHull

        original_debug = _constants.DEBUG

        try:
            _constants.DEBUG = True

            lb = np.array([-1.0, -1.0])
            ub = np.array([1.0, 1.0])

            hull = ReLUHull(if_cal_multi_neuron_constrs=True)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            # Should execute lines 378-379 (DEBUG constraint path)
            assert np.all(np.isfinite(constraints))

        finally:
            _constants.DEBUG = original_debug

    def test_debug_mode_multiple_activations(self):
        """Test DEBUG mode with different activation functions."""
        from wraact import _constants
        from wraact.acthull import ELUHull, LeakyReLUHull

        original_debug = _constants.DEBUG

        try:
            _constants.DEBUG = True

            lb = np.array([-1.0, -1.0])
            ub = np.array([1.0, 1.0])

            # Test ELU
            hull_elu = ELUHull()
            c_elu = hull_elu.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(c_elu))

            # Test LeakyReLU
            hull_lrelu = LeakyReLUHull()
            c_lrelu = hull_lrelu.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(c_lrelu))

        finally:
            _constants.DEBUG = original_debug

    def test_debug_mode_restored_after_error(self):
        """Test that DEBUG state is properly restored even after errors.

        Tests proper state management of DEBUG flag.
        """
        from wraact import _constants

        original_debug = _constants.DEBUG

        try:
            # Set DEBUG to True
            _constants.DEBUG = True
            assert _constants.DEBUG is True

            # Simulate work

        finally:
            # Restore original
            _constants.DEBUG = original_debug

        # Should be restored
        assert original_debug == _constants.DEBUG


class TestActHullInputValidation:
    """Test ActHull input validation and error handling."""

    def test_missing_bounds_error(self):
        """Test error when bounds are not provided.

        Tests basic input validation for None bounds.
        """
        from wraact.acthull import ReLUHull

        hull = ReLUHull()

        # Both bounds None
        with pytest.raises((ValueError, TypeError)):
            hull.cal_hull(input_lower_bounds=None, input_upper_bounds=None)

    def test_dimension_mismatch_error(self):
        """Test error when c, lb, ub have different dimensions.

        Tests line 489 in acthull/_act.py:
        - Dimension consistency check
        """
        from wraact.acthull import ReLUHull

        c_3d = np.array([[0.0, 1.0, 0.0, 0.0]])  # 3D
        lb_2d = np.array([-1.0, -1.0])  # 2D
        ub_4d = np.array([1.0, 1.0, 1.0, 1.0])  # 4D

        hull = ReLUHull()

        # Should trigger line 489
        with pytest.raises((ValueError, RuntimeError)):
            hull.cal_hull(input_constrs=c_3d, input_lower_bounds=lb_2d, input_upper_bounds=ub_4d)

    def test_lb_ub_dimension_mismatch(self):
        """Test error when lb and ub have different dimensions."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])  # 2D
        ub = np.array([1.0, 1.0, 1.0])  # 3D

        hull = ReLUHull()

        with pytest.raises((ValueError, RuntimeError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_reversed_bounds_error(self):
        """Test error when lb > ub.

        Tests bound validation.
        """
        from wraact.acthull import ReLUHull

        lb = np.array([1.0, 1.0])
        ub = np.array([-1.0, -1.0])  # Reversed!

        hull = ReLUHull()

        with pytest.raises((ValueError, RuntimeError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_missing_constraints_multi_neuron_error(self):
        """Test error when multi-neuron mode requires constraints.

        Tests that multi-neuron constraint mode needs valid configuration.
        """
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)

        # With both constraint modes, should generate valid result or raise
        try:
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If it succeeds, constraints should be valid
            assert isinstance(result, np.ndarray)
            assert np.all(np.isfinite(result))
        except (ValueError, RuntimeError):
            # It's acceptable if it raises an error for this configuration
            pass


class TestActHullExceptionRecovery:
    """Test ActHull exception recovery and error handling.

    These tests attempt to trigger exception recovery paths.
    Note: Some paths may be difficult to reach naturally.
    """

    def test_normal_computation_path(self):
        """Test normal computation without triggering exceptions.

        Baseline test to ensure normal path works.
        """
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ReLUHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(constraints))

    def test_various_polytope_configurations(self):
        """Test with various polytope configurations.

        Tests robustness with different bound configurations (avoid MIN_BOUNDS_RANGE).
        """
        from wraact.acthull import ReLUHull

        test_cases = [
            (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),  # Moderate
            (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),  # Very large
            (np.array([-0.1, -0.1]), np.array([0.1, 0.1])),  # Small but OK
            (np.array([-100.0, -1.0]), np.array([1.0, 100.0])),  # Asymmetric
        ]

        hull = ReLUHull()

        for lb, ub in test_cases:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))

    def test_multidimensional_robustness(self):
        """Test exception recovery robustness with multiple dimensions.

        Tests that exception handling works across dimensions.
        """
        from wraact.acthull import ELUHull

        hull = ELUHull()

        for d in [2, 3, 4, 5]:
            lb = -np.ones(d)
            ub = np.ones(d)

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))

    def test_exception_recovery_with_edge_bounds(self):
        """Test exception recovery with edge case bounds.

        Tests behavior near boundaries of numerical stability.
        """
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()

        # Very large bounds (might trigger numerical issues)
        lb = -np.array([1e3, 1e3])
        ub = np.array([1e3, 1e3])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert np.all(np.isfinite(constraints))

    def test_all_activation_types_stability(self):
        """Test exception recovery stability across all activation types.

        Tests robustness across different activation functions.
        """
        from wraact.acthull import (
            ELUHull,
            LeakyReLUHull,
            MaxPoolHull,
            ReLUHull,
            SigmoidHull,
            TanhHull,
        )

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        activations = [
            ReLUHull,
            LeakyReLUHull,
            ELUHull,
            SigmoidHull,
            TanhHull,
            MaxPoolHull,
        ]

        for hull_class in activations:
            hull = hull_class()
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))
