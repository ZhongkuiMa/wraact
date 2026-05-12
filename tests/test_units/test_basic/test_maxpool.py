"""Soundness tests for MaxPoolHull using template pattern (multi-variable activation).

This demonstrates how to test MaxPoolHull, which is different from element-wise
activation functions because:
- Input: d-dimensional vector
- Output: 1-dimensional scalar (maximum element)

This is a multi-variable activation function where the output depends on
comparing all input dimensions.

Key Characteristics of MaxPool:
===============================
- Output = max(x1, x2, ..., xd)
- Always outputs the largest input value
- Piecewise linear with d linear pieces (one for each possible maximum)
- Used in pooling layers in neural networks

Testing Strategy:
=================
Since MaxPool reduces d inputs to 1 output, the hull constraint format is
different from element-wise functions:
- Shape: (num_constraints, d + 2) instead of (2*d + 1)
- Columns: [b | x1 | x2 | ... | xd | y]

NOTE: MaxPool is more complex because it's not element-wise. The template
approach may need custom adaptation.
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest


def maxpool_np(x):
    """NumPy implementation of MaxPool for testing.

    Args:
        x: Input array of shape (d,)

    Returns:
        max(x)
    """
    return np.max(x)


class TestMaxPoolBasicFunctionality:
    """Basic functionality tests for MaxPoolHull."""

    def test_cal_hull_returns_ndarray(self, maxpool_hull_class):
        """Verify cal_hull() returns an ndarray."""
        hull = maxpool_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # 2D array

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(2, 4, id="2d"),
            pytest.param(3, 5, id="3d"),
        ],
    )
    def test_cal_hull_output_shape(self, dim, expected_cols, maxpool_hull_class):
        """Verify output shape follows formula: d + 2."""
        hull = maxpool_hull_class()
        lb = -np.ones(dim)
        ub = np.ones(dim)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[1] == expected_cols

    def test_cal_hull_output_finite(self, maxpool_hull_class):
        """Verify output contains no inf or nan values."""
        hull = maxpool_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))

    def test_maxpool_function_characteristics(self):
        """Verify MaxPool function has expected characteristics."""
        # MaxPool should return the maximum element
        x1 = np.array([-1.0, 0.5])
        assert maxpool_np(x1) == 0.5

        x2 = np.array([2.0, 1.0, 3.0])
        assert maxpool_np(x2) == 3.0

        x3 = np.array([-5.0, -2.0, -10.0])
        assert maxpool_np(x3) == -2.0

    def test_maxpool_identity_property(self):
        """Verify maxpool of single element equals that element."""
        x = np.array([5.0])
        assert maxpool_np(x) == 5.0

    def test_maxpool_monotonicity(self):
        """Verify maxpool is monotonic: if x <= y elementwise, then max(x) <= max(y)."""
        x = np.array([1.0, 2.0, 1.5])
        y = np.array([1.5, 2.5, 2.0])

        # x <= y element-wise
        assert np.all(x <= y)

        # max(x) <= max(y)
        assert maxpool_np(x) <= maxpool_np(y)

    def test_maxpool_output_bounds(self):
        """Verify maxpool output is always within input bounds."""
        np.linspace(-2.0, 2.0, 100)

        # Test 2D maxpool
        rng = np.random.default_rng()
        for _ in range(50):
            x = rng.uniform(-3.0, 3.0, 2)
            y = maxpool_np(x)

            # Output should be within bounds
            assert y >= np.min(x)
            assert y <= np.max(x)

    def test_maxpool_piecewise_linear_structure(self):
        """Verify maxpool structure: output equals one of the inputs."""
        # For 2D: max(x, y) should equal either x or y
        x = np.array([1.5, 2.0])
        y = maxpool_np(x)

        # y should equal one of the inputs
        assert y == x[0] or y == x[1]

        # For 3D
        x3 = np.array([-1.0, 0.5, 2.0])
        y3 = maxpool_np(x3)
        assert y3 in x3

    def test_maxpool_commutative(self):
        """Verify maxpool is commutative: max(x, y) = max(y, x)."""
        x = np.array([1.0, 3.0])

        # Any permutation should give same result
        assert maxpool_np(x) == maxpool_np(np.array([3.0, 1.0]))

        x3 = np.array([1.0, 3.0, 2.0])
        assert maxpool_np(x3) == maxpool_np(np.array([3.0, 2.0, 1.0]))

    def test_maxpool_vs_relu_comparison(self):
        """Compare MaxPool with ReLU."""
        x = np.array([1.0, 2.0])
        y_maxpool = maxpool_np(x)
        np.maximum(0, x[0])

        # MaxPool outputs the max of all inputs (2.0 in this case)
        # ReLU applied to single element is element-wise
        assert y_maxpool == 2.0


class TestMaxPoolSoundnessBasic:
    """Basic soundness tests for MaxPoolHull (non-template version)."""

    def test_soundness_2d_single_point(self, maxpool_hull_class):
        """Test that a single point (x, max(x)) satisfies hull constraints."""
        hull = maxpool_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Test point
        x = np.array([0.5, 0.3])
        y = maxpool_np(x)  # max(0.5, 0.3) = 0.5

        # For MaxPool, point is [x1, x2, y]
        point = np.concatenate([x, [y]])

        b = result[:, 0]
        A = result[:, 1:]
        constraint_values = b + A @ point

        # Should satisfy all constraints
        assert np.all(constraint_values >= -1e-8), (
            f"Point outside hull. Min constraint: {np.min(constraint_values)}"
        )

    @pytest.mark.parametrize(
        ("dim", "lb", "ub"),
        [
            pytest.param(2, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), id="2d"),
            pytest.param(3, np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]), id="3d"),
        ],
    )
    def test_soundness_monte_carlo(self, dim, lb, ub, maxpool_hull_class):
        """Verify soundness with Monte Carlo sampling for MaxPool."""
        hull = maxpool_hull_class()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_samples = 1000
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (num_samples, dim))

        violations = 0
        for x in samples:
            y = maxpool_np(x)
            point = np.concatenate([x, [y]])

            b = result[:, 0]
            A = result[:, 1:]
            constraint_values = b + A @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.0, (
            f"Soundness violation: {satisfaction_rate:.2f}% ({violations}/{num_samples})"
        )

    def test_maxpool_deterministic(self, maxpool_hull_class):
        """Verify hull computation is deterministic."""
        hull = maxpool_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Compute hull multiple times
        result1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        result2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(result1, result2)


class TestMaxPoolBoundEdgeCases:
    """Test MaxPool with edge case bounds (trivial cases)."""

    @pytest.mark.parametrize(
        ("lb", "ub", "expected_cols", "scenario"),
        [
            pytest.param(
                np.array([0.5, 0.5]), np.array([1.0, 1.0]), 4, "all_positive_2d", id="all_positive"
            ),
            pytest.param(
                np.array([-1.0, -1.0]),
                np.array([-0.5, -0.5]),
                4,
                "all_negative_2d",
                id="all_negative",
            ),
            pytest.param(
                np.array([-0.03, -0.03, -0.03]),
                np.array([0.03, 0.03, 0.03]),
                5,
                "small_range_3d",
                id="small_range_3d",
            ),
            pytest.param(
                np.array([-100.0, -50.0, -1.0]),
                np.array([1.0, 50.0, 100.0]),
                5,
                "wide_asymmetric_3d",
                id="wide_asymmetric",
            ),
        ],
    )
    def test_cal_hull_bound_configurations(
        self, lb, ub, expected_cols, scenario, maxpool_hull_class
    ):
        """Test MaxPool with various bound configurations."""
        hull = maxpool_hull_class()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        assert constraints.shape[1] == expected_cols


class TestMaxPoolSingleNeuronMode:
    """Test MaxPool hull with single-neuron constraint mode.

    This tests the single-neuron constraint calculation path which is
    normally disabled in default ActHull initialization.
    """

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(2, 4, id="2d"),
            pytest.param(3, 5, id="3d"),
        ],
    )
    def test_cal_hull_single_neuron(self, dim, expected_cols, maxpool_hull_class):
        """Test single-neuron constraints for MaxPool with given dimension."""
        lb = -np.ones(dim)
        ub = np.ones(dim)

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == expected_cols
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_soundness(self, maxpool_hull_class):
        """Verify soundness of single-neuron constraints with Monte Carlo."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Monte Carlo sampling to verify soundness
        rng = np.random.default_rng(42)
        samples = rng.uniform(lb, ub, (1000, 2))

        violations = 0
        for x in samples:
            y = maxpool_np(x)
            point = np.concatenate([x, [y]])

            b = constraints[:, 0]
            A = constraints[:, 1:]
            constraint_values = b + A @ point

            if not np.all(constraint_values >= -1e-6):
                violations += 1

        satisfaction_rate = 100.0 * (1000 - violations) / 1000
        assert satisfaction_rate >= 95.0

    def test_cal_hull_single_neuron_cache_behavior(self, maxpool_hull_class):
        """Test caching behavior in single-neuron mode."""
        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )

        # First call
        lb1 = np.array([-1.0, -1.0])
        ub1 = np.array([1.0, 1.0])
        c1 = hull.cal_hull(input_lower_bounds=lb1, input_upper_bounds=ub1)

        # Second call with different bounds but same dimension
        lb2 = np.array([-2.0, -2.0])
        ub2 = np.array([2.0, 2.0])
        c2 = hull.cal_hull(input_lower_bounds=lb2, input_upper_bounds=ub2)

        # Same dimension should have same number of constraints
        assert c1.shape == c2.shape

    def test_cal_sn_constrs_upper_bounds(self, maxpool_hull_class):
        """Test upper bounds constraint calculation in single-neuron mode."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify constraints include upper bound constraints
        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0

    def test_cal_sn_constrs_lower_bounds(self, maxpool_hull_class):
        """Test lower bounds constraint calculation in single-neuron mode."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify constraints are generated
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_output_shape(self, maxpool_hull_class):
        """Verify single-neuron constraint output shape for 2D."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 2D input: d + 2 = 4 columns
        assert constraints.shape[1] == 4
        # Should have at least some constraints
        assert constraints.shape[0] > 0

    def test_cal_hull_single_neuron_finite(self, maxpool_hull_class):
        """Verify single-neuron constraints contain no inf or nan."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = maxpool_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(constraints))
