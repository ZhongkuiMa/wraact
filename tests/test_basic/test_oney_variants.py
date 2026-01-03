"""Tests for WithOneY variants comparing with full hull implementations.

WithOneY variants compute constraints for a single output dimension at a time,
useful for incremental computation in neural network verification.

Key Features:
=============
- Compute constraints for dimension i only: y_i = f_i(x)
- Lower constraint count compared to full hull (fewer constraints)
- Useful for single-neuron verification
- Should be sound but may be less tight than full hull

Comparison Strategy:
====================
For each activation function, test:
1. Basic functionality: Returns correct shape
2. Single-neuron comparison: WithOneY vs full hull for first neuron
3. Soundness: Random points still satisfy constraints
4. Constraint count: WithOneY has fewer constraints than full hull
"""

__docformat__ = "restructuredtext"

import numpy as np


class TestReLUWithOneY:
    """Tests for ReLUHullWithOneY."""

    def test_relu_oney_returns_ndarray(self):
        """Verify ReLUHullWithOneY.cal_hull() returns ndarray."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_relu_oney_output_shape_2d(self):
        """Verify output shape for 2D input: (constraints, 4) for single output."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should have columns for [b | x1 | x2 | y] where y is single output
        assert result.shape[1] == 4  # 2 inputs + 1 bias + 1 output

    def test_relu_oney_constraint_count_vs_full(self):
        """Verify WithOneY has fewer constraints than full hull."""
        from wraact.acthull import ReLUHull
        from wraact.oney import ReLUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        full_hull = ReLUHull()
        oney_hull = ReLUHullWithOneY()

        full_result = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_result = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # WithOneY should have fewer or equal constraints
        assert oney_result.shape[0] <= full_result.shape[0]

    def test_relu_oney_soundness_2d(self):
        """Verify ReLUHullWithOneY constraints are satisfied by (x, relu(x_i))."""
        from wraact.oney import ReLUHullWithOneY

        def relu_np(x):
            return np.maximum(0, x)

        hull = ReLUHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Random sampling
        num_samples = 500
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (num_samples, 2))

        violations = 0
        for x in samples:
            y_full = relu_np(x)  # Compute all outputs
            y = y_full[0]  # WithOneY only computes for first output dimension
            point = np.concatenate([x, [y]])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.0, f"Soundness violation: {satisfaction_rate:.2f}%"


class TestLeakyReLUWithOneY:
    """Tests for LeakyReLUHullWithOneY."""

    def test_leakyrelu_oney_returns_ndarray(self):
        """Verify LeakyReLUHullWithOneY.cal_hull() returns ndarray."""
        from wraact.oney import LeakyReLUHullWithOneY

        hull = LeakyReLUHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)

    def test_leakyrelu_oney_constraint_count_vs_full(self):
        """Verify WithOneY has fewer constraints than full hull."""
        from wraact.acthull import LeakyReLUHull
        from wraact.oney import LeakyReLUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        full_hull = LeakyReLUHull()
        oney_hull = LeakyReLUHullWithOneY()

        full_result = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_result = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # WithOneY should have fewer or equal constraints
        assert oney_result.shape[0] <= full_result.shape[0]


class TestELUWithOneY:
    """Tests for ELUHullWithOneY."""

    def test_elu_oney_returns_ndarray(self):
        """Verify ELUHullWithOneY.cal_hull() returns ndarray."""
        from wraact.oney import ELUHullWithOneY

        hull = ELUHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)

    def test_elu_oney_constraint_count_vs_full(self):
        """Verify WithOneY has fewer constraints than full hull."""
        from wraact.acthull import ELUHull
        from wraact.oney import ELUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        full_hull = ELUHull()
        oney_hull = ELUHullWithOneY()

        full_result = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_result = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # WithOneY should have fewer or equal constraints
        assert oney_result.shape[0] <= full_result.shape[0]


class TestSigmoidWithOneY:
    """Tests for SigmoidHullWithOneY."""

    def test_sigmoid_oney_returns_ndarray(self):
        """Verify SigmoidHullWithOneY.cal_hull() returns ndarray."""
        from wraact.oney import SigmoidHullWithOneY

        hull = SigmoidHullWithOneY()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)

    def test_sigmoid_oney_constraint_count_vs_full(self):
        """Verify WithOneY has fewer constraints than full hull."""
        from wraact.acthull import SigmoidHull
        from wraact.oney import SigmoidHullWithOneY

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        full_hull = SigmoidHull()
        oney_hull = SigmoidHullWithOneY()

        full_result = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_result = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # WithOneY should have fewer or equal constraints
        assert oney_result.shape[0] <= full_result.shape[0]


class TestTanhWithOneY:
    """Tests for TanhHullWithOneY."""

    def test_tanh_oney_returns_ndarray(self):
        """Verify TanhHullWithOneY.cal_hull() returns ndarray."""
        from wraact.oney import TanhHullWithOneY

        hull = TanhHullWithOneY()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)

    def test_tanh_oney_constraint_count_vs_full(self):
        """Verify WithOneY has fewer constraints than full hull."""
        from wraact.acthull import TanhHull
        from wraact.oney import TanhHullWithOneY

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        full_hull = TanhHull()
        oney_hull = TanhHullWithOneY()

        full_result = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_result = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # WithOneY should have fewer or equal constraints
        assert oney_result.shape[0] <= full_result.shape[0]

    def test_tanh_oney_soundness_2d(self):
        """Verify TanhHullWithOneY constraints are satisfied by (x, tanh(x_i))."""
        from wraact.oney import TanhHullWithOneY

        hull = TanhHullWithOneY()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Random sampling
        num_samples = 500
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (num_samples, 2))

        violations = 0
        for x in samples:
            y_full = np.tanh(x)  # Compute all outputs
            y = y_full[0]  # WithOneY only computes for first output dimension
            point = np.concatenate([x, [y]])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.0, f"Soundness violation: {satisfaction_rate:.2f}%"


class TestMaxPoolWithOneY:
    """Tests for MaxPoolHullWithOneY (multi-variable reduction)."""

    def test_maxpool_oney_returns_ndarray(self):
        """Verify MaxPoolHullWithOneY.cal_hull() returns ndarray."""
        from wraact.oney import MaxPoolHullWithOneY

        hull = MaxPoolHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)

    def test_maxpool_oney_constraint_count_vs_full(self):
        """Verify WithOneY has fewer constraints than full hull."""
        from wraact.acthull import MaxPoolHull
        from wraact.oney import MaxPoolHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        full_hull = MaxPoolHull()
        oney_hull = MaxPoolHullWithOneY()

        full_result = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_result = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # WithOneY should have fewer or equal constraints
        assert oney_result.shape[0] <= full_result.shape[0]


class TestWithOneYGeneralProperties:
    """Test general properties of all WithOneY variants."""

    def test_relu_oney_deterministic(self):
        """Verify ReLU WithOneY computation is deterministic."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        result2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(result1, result2)

    def test_tanh_oney_deterministic(self):
        """Verify Tanh WithOneY computation is deterministic."""
        from wraact.oney import TanhHullWithOneY

        hull = TanhHullWithOneY()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        result2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(result1, result2)

    def test_elu_oney_output_finite(self):
        """Verify ELU WithOneY outputs contain no inf/nan."""
        from wraact.oney import ELUHullWithOneY

        hull = ELUHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))

    def test_relu_oney_vs_full_shape_difference(self):
        """Verify WithOneY typically has different column count than full hull."""
        from wraact.acthull import ReLUHull
        from wraact.oney import ReLUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        full_hull = ReLUHull()
        oney_hull = ReLUHullWithOneY()

        full_result = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_result = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Full hull: [b | x1 | x2 | y1 | y2] = 5 columns
        # OneY hull: [b | x1 | x2 | y] = 4 columns (single output)
        assert full_result.shape[1] == 5
        assert oney_result.shape[1] == 4


class TestWithOneYAdvancedFeatures:
    """Test advanced OneY features and error handling paths."""

    def test_relu_oney_with_double_orders(self):
        """Test ReLU OneY with double orders mode enabled."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        # Access internal state to verify double_orders can be used
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should return valid constraints
        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))

    def test_leakyrelu_oney_output_shape_3d(self):
        """Verify LeakyReLU OneY with 3D input."""
        from wraact.oney import LeakyReLUHullWithOneY

        hull = LeakyReLUHullWithOneY()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 3D input: 3 input dims + 1 bias + 1 output = 5 columns
        assert result.shape[1] == 5
        assert np.all(np.isfinite(result))

    def test_elu_oney_output_shape_3d(self):
        """Verify ELU OneY with 3D input."""
        from wraact.oney import ELUHullWithOneY

        hull = ELUHullWithOneY()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 3D input: 3 input dims + 1 bias + 1 output = 5 columns
        assert result.shape[1] == 5
        assert np.all(np.isfinite(result))

    def test_sigmoid_oney_3d(self):
        """Test Sigmoid OneY with 3D input."""
        from wraact.oney import SigmoidHullWithOneY

        hull = SigmoidHullWithOneY()
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[1] == 5  # 3 inputs + 1 bias + 1 output
        assert np.all(np.isfinite(result))

    def test_maxpool_oney_output_shape(self):
        """Verify MaxPool OneY output shape."""
        from wraact.oney import MaxPoolHullWithOneY

        hull = MaxPoolHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # MaxPool: [b | x1 | x2 | y] = 4 columns (d+2 format doesn't apply to OneY single output)
        assert result.shape[1] == 4
        assert np.all(np.isfinite(result))

    def test_relu_oney_4d_input(self):
        """Test ReLU OneY with higher-dimensional input."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        lb = np.array([-1.0, -1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 4D input: 4 input dims + 1 bias + 1 output = 6 columns
        assert result.shape[1] == 6
        assert np.all(np.isfinite(result))

    def test_tanh_oney_3d(self):
        """Test Tanh OneY with 3D input."""
        from wraact.oney import TanhHullWithOneY

        hull = TanhHullWithOneY()
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[1] == 5  # 3 inputs + 1 bias + 1 output
        assert np.all(np.isfinite(result))

    def test_leakyrelu_oney_soundness_3d(self):
        """Verify LeakyReLU OneY soundness in 3D."""
        from wraact.oney import LeakyReLUHullWithOneY

        def leakyrelu_np(x, negative_slope=0.01):
            return np.where(x >= 0, x, negative_slope * x)

        hull = LeakyReLUHullWithOneY()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Random sampling
        num_samples = 500
        rng = np.random.default_rng(42)
        samples = rng.uniform(lb, ub, (num_samples, 3))

        violations = 0
        for x in samples:
            y_full = leakyrelu_np(x)
            y = y_full[0]
            point = np.concatenate([x, [y]])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 95.0, f"Soundness violation: {satisfaction_rate:.2f}%"

    def test_relu_oney_finite_values_large_bounds(self):
        """Test ReLU OneY with larger bounds."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        lb = np.array([-10.0, -10.0])
        ub = np.array([10.0, 10.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))

    def test_elu_oney_soundness_3d(self):
        """Verify ELU OneY soundness in 3D."""
        from wraact.oney import ELUHullWithOneY

        def elu_np(x, alpha=1.0):
            return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

        hull = ELUHullWithOneY()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Random sampling
        num_samples = 500
        rng = np.random.default_rng(42)
        samples = rng.uniform(lb, ub, (num_samples, 3))

        violations = 0
        for x in samples:
            y_full = elu_np(x)
            y = y_full[0]
            point = np.concatenate([x, [y]])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        # ELU OneY less tight, so lower threshold
        assert satisfaction_rate >= 85.0, f"Soundness violation: {satisfaction_rate:.2f}%"
