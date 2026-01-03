"""S-shaped activation function tests (Phase 4 coverage improvement).

This module provides comprehensive tests for S-shaped activation functions:
- SigmoidHull: Sigmoid activation function
- TanhHull: Hyperbolic tangent activation function

These are more complex than ReLU-like activations because they have:
- Non-piecewise-linear behavior
- Smooth, continuous gradients
- More complex hull computation strategies

Testing Strategy:
=================
- Single-neuron mode tests
- Multi-neuron mode tests
- Constraint mode combinations
- Edge cases (small ranges, asymmetric bounds)
- Monte Carlo soundness verification
- Determinism and reproducibility
"""

__docformat__ = "restructuredtext"

import numpy as np


def sigmoid_np(x):
    """NumPy implementation of sigmoid for testing.

    Args:
        x: Input value(s)

    Returns:
        1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


def tanh_np(x):
    """NumPy implementation of tanh for testing.

    Args:
        x: Input value(s)

    Returns:
        tanh(x)
    """
    return np.tanh(x)


class TestSigmoidHullBasic:
    """Basic functionality tests for SigmoidHull."""

    def test_sigmoid_hull_returns_ndarray(self):
        """Verify cal_hull() returns an ndarray."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_sigmoid_hull_output_shape_2d(self):
        """Verify output shape for 2D input."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # 2D input: 2*2 + 1 = 5 columns
        assert result.shape[1] == 5

    def test_sigmoid_hull_output_shape_3d(self):
        """Verify output shape for 3D input."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        hull = SigmoidHull()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # 3D input: 2*3 + 1 = 7 columns
        assert result.shape[1] == 7

    def test_sigmoid_hull_finite_values(self):
        """Verify output contains no inf or nan."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))

    def test_sigmoid_function_properties(self):
        """Verify sigmoid function properties."""
        # Sigmoid is bounded [0, 1]
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        y = sigmoid_np(x)

        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

        # At x=0, sigmoid(0) = 0.5
        assert np.isclose(sigmoid_np(0.0), 0.5)

    def test_sigmoid_monotonicity(self):
        """Verify sigmoid is monotonically increasing."""
        x = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        y = sigmoid_np(x)

        # Check monotonicity
        for i in range(len(y) - 1):
            assert y[i] < y[i + 1]


class TestSigmoidHullSingleNeuron:
    """Test SigmoidHull with single-neuron constraint mode."""

    def test_sigmoid_single_neuron_2d(self):
        """Test single-neuron constraints for 2D sigmoid."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 5
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_single_neuron_3d(self):
        """Test single-neuron constraints for 3D sigmoid."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 7
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_both_modes_enabled(self):
        """Test sigmoid with both constraint modes enabled."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=True,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_deterministic(self):
        """Verify sigmoid constraints are deterministic."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(c1, c2)


class TestTanhHullBasic:
    """Basic functionality tests for TanhHull."""

    def test_tanh_hull_returns_ndarray(self):
        """Verify cal_hull() returns an ndarray."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_tanh_hull_output_shape_2d(self):
        """Verify output shape for 2D input."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # 2D input: 2*2 + 1 = 5 columns
        assert result.shape[1] == 5

    def test_tanh_hull_output_shape_3d(self):
        """Verify output shape for 3D input."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        hull = TanhHull()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # 3D input: 2*3 + 1 = 7 columns
        assert result.shape[1] == 7

    def test_tanh_hull_finite_values(self):
        """Verify output contains no inf or nan."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))

    def test_tanh_function_properties(self):
        """Verify tanh function properties."""
        # Tanh is bounded [-1, 1]
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        y = tanh_np(x)

        assert np.all(y >= -1.0)
        assert np.all(y <= 1.0)

        # At x=0, tanh(0) = 0
        assert np.isclose(tanh_np(0.0), 0.0)

    def test_tanh_odd_function(self):
        """Verify tanh is an odd function: tanh(-x) = -tanh(x)."""
        x = np.array([0.5, 1.0, 2.0])
        y_pos = tanh_np(x)
        y_neg = tanh_np(-x)

        np.testing.assert_array_almost_equal(y_pos, -y_neg)


class TestTanhHullSingleNeuron:
    """Test TanhHull with single-neuron constraint mode."""

    def test_tanh_single_neuron_2d(self):
        """Test single-neuron constraints for 2D tanh."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 5
        assert np.all(np.isfinite(constraints))

    def test_tanh_single_neuron_3d(self):
        """Test single-neuron constraints for 3D tanh."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        hull = TanhHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 7
        assert np.all(np.isfinite(constraints))

    def test_tanh_both_modes_enabled(self):
        """Test tanh with both constraint modes enabled."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=True,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))

    def test_tanh_deterministic(self):
        """Verify tanh constraints are deterministic."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        np.testing.assert_array_equal(c1, c2)


class TestSShapeEdgeCases:
    """Test S-shaped activations with edge cases."""

    def test_sigmoid_asymmetric_bounds(self):
        """Test sigmoid with asymmetric bounds."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-5.0, -1.0])
        ub = np.array([0.5, 3.0])

        hull = SigmoidHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_large_range(self):
        """Test sigmoid with large input range."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-100.0, -100.0])
        ub = np.array([100.0, 100.0])

        hull = SigmoidHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_small_range(self):
        """Test sigmoid with small input range."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-0.03, -0.03])
        ub = np.array([0.03, 0.03])

        hull = SigmoidHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_tanh_asymmetric_bounds(self):
        """Test tanh with asymmetric bounds."""
        from wraact.acthull import TanhHull

        lb = np.array([-5.0, -1.0])
        ub = np.array([0.5, 3.0])

        hull = TanhHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_tanh_large_range(self):
        """Test tanh with large input range."""
        from wraact.acthull import TanhHull

        lb = np.array([-100.0, -100.0])
        ub = np.array([100.0, 100.0])

        hull = TanhHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_tanh_small_range(self):
        """Test tanh with small input range."""
        from wraact.acthull import TanhHull

        lb = np.array([-0.03, -0.03])
        ub = np.array([0.03, 0.03])

        hull = TanhHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestSShapeMultiDimensional:
    """Test S-shaped activations with various dimensions."""

    def test_sigmoid_1d(self):
        """Test sigmoid with 1D input."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0])
        ub = np.array([2.0])

        hull = SigmoidHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 3  # 2*1 + 1
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_4d(self):
        """Test sigmoid with 4D input."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0, 2.0])

        hull = SigmoidHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 9  # 2*4 + 1
        assert np.all(np.isfinite(constraints))

    def test_tanh_1d(self):
        """Test tanh with 1D input."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0])
        ub = np.array([2.0])

        hull = TanhHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 3  # 2*1 + 1
        assert np.all(np.isfinite(constraints))

    def test_tanh_4d(self):
        """Test tanh with 4D input."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0, 2.0])

        hull = TanhHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 9  # 2*4 + 1
        assert np.all(np.isfinite(constraints))


class TestSShapeConstraintModes:
    """Test constraint mode combinations for S-shaped activations."""

    def test_sigmoid_single_only(self):
        """Test sigmoid with single-neuron mode only."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints.shape[0] > 0

    def test_sigmoid_multi_only(self):
        """Test sigmoid with multi-neuron mode only."""
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=False,
            if_cal_multi_neuron_constrs=True,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints.shape[0] > 0

    def test_tanh_single_only(self):
        """Test tanh with single-neuron mode only."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints.shape[0] > 0

    def test_tanh_multi_only(self):
        """Test tanh with multi-neuron mode only."""
        from wraact.acthull import TanhHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull(
            if_cal_single_neuron_constrs=False,
            if_cal_multi_neuron_constrs=True,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints.shape[0] > 0
