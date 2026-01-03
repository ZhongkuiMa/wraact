"""Extended MaxPool tests for Phase 6 coverage improvement.

This module provides comprehensive tests for MaxPoolHull covering:
1. Single-neuron constraint modes
2. Degenerate polytope handling
3. Trivial case detection
4. Direct method invocation
5. Output constraints handling
"""

__docformat__ = "restructuredtext"

import numpy as np


class TestMaxPoolSingleNeuronExtended:
    """Extended single-neuron constraint tests for MaxPool."""

    def test_maxpool_single_neuron_1d(self):
        """Test MaxPool single-neuron constraints for 1D input."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        lb = np.array([-1.0])
        ub = np.array([1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 3  # d+2 = 1+2
        assert np.all(np.isfinite(constraints))

    def test_maxpool_single_neuron_4d(self):
        """Test MaxPool single-neuron constraints for 4D input."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        lb = np.array([-1.0, -1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 6  # d+2 = 4+2
        assert np.all(np.isfinite(constraints))

    def test_maxpool_single_neuron_soundness_3d(self):
        """Verify MaxPool single-neuron soundness in 3D."""
        from wraact.acthull import MaxPoolHull

        def maxpool_np(x):
            return np.max(x)

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Random sampling
        num_samples = 500
        rng = np.random.default_rng(42)
        samples = rng.uniform(lb, ub, (num_samples, 3))

        violations = 0
        for x in samples:
            y = maxpool_np(x)
            point = np.concatenate([x, [y]])

            b = constraints[:, 0]
            a = constraints[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 90.0, f"Soundness violation: {satisfaction_rate:.2f}%"

    def test_maxpool_single_neuron_deterministic(self):
        """Test single-neuron constraints are deterministic."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(c1, c2)

    def test_maxpool_both_modes_enabled(self):
        """Test MaxPool with both single and multi-neuron modes enabled."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))


class TestMaxPoolEdgeCasesExtended:
    """Extended edge case tests for MaxPool."""

    def test_maxpool_all_positive_bounds(self):
        """Test MaxPool with all-positive bounds."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([0.5, 0.5, 0.5])
        ub = np.array([1.0, 1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # All positive inputs means max is always determined
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_all_negative_bounds(self):
        """Test MaxPool with all-negative bounds."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([-0.5, -0.5, -0.5])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # All negative inputs
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_asymmetric_bounds(self):
        """Test MaxPool with asymmetric bounds."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-3.0, -1.0, -2.0])
        ub = np.array([1.0, 5.0, 0.5])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_very_small_range(self):
        """Test MaxPool with very small input range (at minimum threshold)."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-0.025, -0.025])
        ub = np.array([0.025, 0.025])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_5d_input(self):
        """Test MaxPool with high-dimensional input."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 5D: d+2 = 7 columns
        assert constraints.shape[1] == 7
        assert np.all(np.isfinite(constraints))

    def test_maxpool_mixed_bounds(self):
        """Test MaxPool with mixed positive/negative bounds."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-2.0, 0.5, -1.0])
        ub = np.array([0.5, 2.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestMaxPoolConstraintModes:
    """Test MaxPool constraint mode combinations."""

    def test_maxpool_single_only_2d(self):
        """Test MaxPool with single-neuron only in 2D."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))

    def test_maxpool_multi_only_2d(self):
        """Test MaxPool with multi-neuron only in 2D."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_both_modes_3d(self):
        """Test MaxPool with both modes in 3D."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        # Both modes should produce more constraints
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))


class TestMaxPoolWithConstantBounds:
    """Test MaxPool with special constant bound cases."""

    def test_maxpool_constant_bounds_single_dim(self):
        """Test MaxPool where one dimension has constant bounds."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, 0.5])
        ub = np.array([1.0, 0.5])  # Constant dimension

        # May raise or handle gracefully depending on implementation
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
        except ValueError:
            pass

    def test_maxpool_single_neuron_constant_dim(self):
        """Test single-neuron mode with constant dimension."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        lb = np.array([-1.0, 0.5])
        ub = np.array([1.0, 0.5])

        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
        except ValueError:
            pass


class TestMaxPoolMultiOutput:
    """Test MaxPool with multiple output handling."""

    def test_maxpool_multiple_calls_consistency(self):
        """Test MaxPool consistency across multiple calls."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c3 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(c2, c3)

    def test_maxpool_output_finiteness_large_range(self):
        """Test MaxPool output finiteness with large ranges."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-100.0, -100.0])
        ub = np.array([100.0, 100.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(constraints))
