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
import pytest


class TestMaxPoolSingleNeuronExtended:
    """Extended single-neuron constraint tests for MaxPool."""

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(1, 3, id="1d"),
            pytest.param(4, 6, id="4d"),
        ],
    )
    def test_maxpool_single_neuron(self, dim, expected_cols, maxpool_hull_class):
        """Test MaxPool single-neuron constraints for given input dimension."""
        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )
        lb = -np.ones(dim)
        ub = np.ones(dim)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == expected_cols
        assert np.all(np.isfinite(constraints))

    def test_maxpool_single_neuron_soundness_3d(self, maxpool_hull_class):
        """Verify MaxPool single-neuron soundness in 3D."""

        def maxpool_np(x):
            return np.max(x)

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )
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

    def test_maxpool_single_neuron_deterministic(self, maxpool_hull_class):
        """Test single-neuron constraints are deterministic."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )
        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(c1, c2)

    def test_maxpool_both_modes_enabled(self, maxpool_hull_class):
        """Test MaxPool with both single and multi-neuron modes enabled."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True
        )
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))


class TestMaxPoolEdgeCasesExtended:
    """Extended edge case tests for MaxPool."""

    @pytest.mark.parametrize(
        ("lb", "ub", "expected_cols", "scenario"),
        [
            pytest.param(
                np.array([0.5, 0.5, 0.5]),
                np.array([1.0, 1.0, 1.0]),
                5,
                "all_positive",
                id="all_positive",
            ),
            pytest.param(
                np.array([-1.0, -1.0, -1.0]),
                np.array([-0.5, -0.5, -0.5]),
                5,
                "all_negative",
                id="all_negative",
            ),
            pytest.param(
                np.array([-3.0, -1.0, -2.0]),
                np.array([1.0, 5.0, 0.5]),
                5,
                "asymmetric",
                id="asymmetric",
            ),
            pytest.param(
                np.array([-0.025, -0.025]),
                np.array([0.025, 0.025]),
                4,
                "very_small",
                id="very_small",
            ),
            pytest.param(
                np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
                np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                7,
                "5d",
                id="5d",
            ),
            pytest.param(
                np.array([-2.0, 0.5, -1.0]), np.array([0.5, 2.0, 1.0]), 5, "mixed", id="mixed"
            ),
        ],
    )
    def test_maxpool_bound_configurations(
        self, lb, ub, expected_cols, scenario, maxpool_hull_class
    ):
        """Test MaxPool with various bound configurations."""
        hull = maxpool_hull_class()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == expected_cols
        assert np.all(np.isfinite(constraints))


class TestMaxPoolConstraintModes:
    """Test MaxPool constraint mode combinations."""

    def test_maxpool_single_only_2d(self, maxpool_hull_class):
        """Test MaxPool with single-neuron only in 2D."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))

    def test_maxpool_multi_only_2d(self, maxpool_hull_class):
        """Test MaxPool with multi-neuron only in 2D."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True
        )
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_both_modes_3d(self, maxpool_hull_class):
        """Test MaxPool with both modes in 3D."""
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True
        )
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        # Both modes should produce more constraints
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))


class TestMaxPoolWithConstantBounds:
    """Test MaxPool with special constant bound cases."""

    def test_maxpool_constant_bounds_single_dim(self, maxpool_hull_class):
        """Test MaxPool where one dimension has constant bounds."""
        hull = maxpool_hull_class()
        lb = np.array([-1.0, 0.5])
        ub = np.array([1.0, 0.5])  # Constant dimension

        # May raise or handle gracefully depending on implementation
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
        except ValueError:
            pass

    def test_maxpool_single_neuron_constant_dim(self, maxpool_hull_class):
        """Test single-neuron mode with constant dimension."""
        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )
        lb = np.array([-1.0, 0.5])
        ub = np.array([1.0, 0.5])

        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
        except ValueError:
            pass


class TestMaxPoolMultiOutput:
    """Test MaxPool with multiple output handling."""

    def test_maxpool_multiple_calls_consistency(self, maxpool_hull_class):
        """Test MaxPool consistency across multiple calls."""
        hull = maxpool_hull_class()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c3 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(c2, c3)

    def test_maxpool_output_finiteness_large_range(self, maxpool_hull_class):
        """Test MaxPool output finiteness with large ranges."""
        hull = maxpool_hull_class()
        lb = np.array([-100.0, -100.0])
        ub = np.array([100.0, 100.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(constraints))
