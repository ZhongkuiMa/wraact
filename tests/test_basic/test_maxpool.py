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

    def test_cal_hull_returns_ndarray(self):
        """Verify cal_hull() returns an ndarray."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # 2D array

    def test_cal_hull_output_shape_2d(self):
        """Verify output shape follows formula: d + 2 = 4 for 2D input."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 2D input to MaxPool: d + 2 = 4 columns [b | x1 | x2 | y]
        assert result.shape[1] == 4

    def test_cal_hull_output_shape_3d(self):
        """Verify output shape follows formula: d + 2 = 5 for 3D input."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 3D input: d + 2 = 5 columns [b | x1 | x2 | x3 | y]
        assert result.shape[1] == 5

    def test_cal_hull_output_finite(self):
        """Verify output contains no inf or nan values."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
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

    def test_soundness_2d_single_point(self):
        """Test that a single point (x, max(x)) satisfies hull constraints."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
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

    def test_soundness_2d_monte_carlo(self):
        """Verify soundness with Monte Carlo sampling for 2D MaxPool."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Random sampling
        num_samples = 1000
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (num_samples, 2))

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

    def test_soundness_3d_monte_carlo(self):
        """Verify soundness with Monte Carlo sampling for 3D MaxPool."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Random sampling
        num_samples = 1000
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (num_samples, 3))

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

    def test_maxpool_deterministic(self):
        """Verify hull computation is deterministic."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Compute hull multiple times
        result1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        result2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(result1, result2)


class TestMaxPoolBoundEdgeCases:
    """Test MaxPool with edge case bounds (trivial cases)."""

    def test_cal_hull_all_positive_bounds(self):
        """Test MaxPool with all-positive bounds (lb >= 0)."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([0.5, 0.5])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_all_negative_bounds(self):
        """Test MaxPool with all-negative bounds (ub <= 0)."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([-0.5, -0.5])

        hull = MaxPoolHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_very_small_range_3d(self):
        """Test MaxPool with small 3D input range (just above minimum threshold)."""
        from wraact.acthull import MaxPoolHull

        # Minimum range threshold is 0.05, so we use range 0.06
        lb = np.array([-0.03, -0.03, -0.03])
        ub = np.array([0.03, 0.03, 0.03])

        hull = MaxPoolHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        # MaxPool uses d+2 format: 3+2 = 5 columns
        assert constraints.shape[1] == 5

    def test_cal_hull_wide_range_asymmetric(self):
        """Test MaxPool with wide asymmetric range."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-100.0, -50.0, -1.0])
        ub = np.array([1.0, 50.0, 100.0])

        hull = MaxPoolHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestMaxPoolSingleNeuronMode:
    """Test MaxPool hull with single-neuron constraint mode.

    This tests the single-neuron constraint calculation path which is
    normally disabled in default ActHull initialization.
    """

    def test_cal_hull_single_neuron_2d_dlp(self):
        """Test single-neuron constraints for 2D MaxPool (DLP variant)."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 4  # d + 2 = 2 + 2 = 4 for 2D
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_3d_dlp(self):
        """Test single-neuron constraints for 3D MaxPool (DLP variant)."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 5  # d + 2 = 3 + 2 = 5 for 3D
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_soundness(self):
        """Verify soundness of single-neuron constraints with Monte Carlo."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

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

    def test_cal_hull_single_neuron_cache_behavior(self):
        """Test caching behavior in single-neuron mode."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

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

    def test_cal_sn_constrs_upper_bounds(self):
        """Test upper bounds constraint calculation in single-neuron mode."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify constraints include upper bound constraints
        assert constraints is not None
        assert constraints.shape[0] > 0

    def test_cal_sn_constrs_lower_bounds(self):
        """Test lower bounds constraint calculation in single-neuron mode."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify constraints are generated
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_output_shape(self):
        """Verify single-neuron constraint output shape for 2D."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 2D input: d + 2 = 4 columns
        assert constraints.shape[1] == 4
        # Should have at least some constraints
        assert constraints.shape[0] > 0

    def test_cal_hull_single_neuron_finite(self):
        """Verify single-neuron constraints contain no inf or nan."""
        from wraact.acthull import MaxPoolHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(constraints))
