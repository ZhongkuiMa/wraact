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
import pytest

from wraact.acthull import SigmoidHull, TanhHull
from wraact.oney import SigmoidHullWithOneY, TanhHullWithOneY


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

    def test_sigmoid_hull_returns_ndarray(self, sigmoid_hull_class):
        """Verify cal_hull() returns an ndarray."""
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = sigmoid_hull_class()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(2, 5, id="2d"),
            pytest.param(3, 7, id="3d"),
        ],
    )
    def test_sigmoid_hull_output_shape(self, dim, expected_cols, sigmoid_hull_class):
        """Verify output shape for given input dimension."""
        lb = -2.0 * np.ones(dim)
        ub = 2.0 * np.ones(dim)

        hull = sigmoid_hull_class()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[1] == expected_cols

    def test_sigmoid_hull_finite_values(self, sigmoid_hull_class):
        """Verify output contains no inf or nan."""
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = sigmoid_hull_class()
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

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(2, 5, id="2d"),
            pytest.param(3, 7, id="3d"),
        ],
    )
    def test_sigmoid_single_neuron(self, dim, expected_cols):
        """Test single-neuron constraints for sigmoid with given dimension."""
        lb = -2.0 * np.ones(dim)
        ub = 2.0 * np.ones(dim)

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == expected_cols
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_both_modes_enabled(self):
        """Test sigmoid with both constraint modes enabled."""
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
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert c1 is not None
        assert c2 is not None

        np.testing.assert_array_equal(c1, c2)


class TestTanhHullBasic:
    """Basic functionality tests for TanhHull."""

    def test_tanh_hull_returns_ndarray(self, tanh_hull_class):
        """Verify cal_hull() returns an ndarray."""
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = tanh_hull_class()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(2, 5, id="2d"),
            pytest.param(3, 7, id="3d"),
        ],
    )
    def test_tanh_hull_output_shape(self, dim, expected_cols, tanh_hull_class):
        """Verify output shape for given input dimension."""
        lb = -2.0 * np.ones(dim)
        ub = 2.0 * np.ones(dim)

        hull = tanh_hull_class()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[1] == expected_cols

    def test_tanh_hull_finite_values(self, tanh_hull_class):
        """Verify output contains no inf or nan."""
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = tanh_hull_class()
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

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(2, 5, id="2d"),
            pytest.param(3, 7, id="3d"),
        ],
    )
    def test_tanh_single_neuron(self, dim, expected_cols):
        """Test single-neuron constraints for tanh with given dimension."""
        lb = -2.0 * np.ones(dim)
        ub = 2.0 * np.ones(dim)

        hull = TanhHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == expected_cols
        assert np.all(np.isfinite(constraints))

    def test_tanh_both_modes_enabled(self):
        """Test tanh with both constraint modes enabled."""
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
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )

        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert c1 is not None
        assert c2 is not None

        np.testing.assert_array_equal(c1, c2)


class TestSShapeEdgeCases:
    """Test S-shaped activations with edge cases."""

    @pytest.mark.parametrize(
        ("lb", "ub", "scenario"),
        [
            pytest.param(
                np.array([-5.0, -1.0]), np.array([0.5, 3.0]), "asymmetric", id="sigmoid_asymmetric"
            ),
            pytest.param(
                np.array([-100.0, -100.0]),
                np.array([100.0, 100.0]),
                "large_range",
                id="sigmoid_large_range",
            ),
            pytest.param(
                np.array([-0.03, -0.03]),
                np.array([0.03, 0.03]),
                "small_range",
                id="sigmoid_small_range",
            ),
        ],
    )
    def test_sigmoid_bound_configurations(self, lb, ub, scenario, sigmoid_hull_class):
        """Test sigmoid with various bound configurations."""
        hull = sigmoid_hull_class()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    @pytest.mark.parametrize(
        ("lb", "ub", "scenario"),
        [
            pytest.param(
                np.array([-5.0, -1.0]), np.array([0.5, 3.0]), "asymmetric", id="tanh_asymmetric"
            ),
            pytest.param(
                np.array([-100.0, -100.0]),
                np.array([100.0, 100.0]),
                "large_range",
                id="tanh_large_range",
            ),
            pytest.param(
                np.array([-0.03, -0.03]),
                np.array([0.03, 0.03]),
                "small_range",
                id="tanh_small_range",
            ),
        ],
    )
    def test_tanh_bound_configurations(self, lb, ub, scenario, tanh_hull_class):
        """Test tanh with various bound configurations."""
        hull = tanh_hull_class()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    @pytest.mark.parametrize(
        ("hull_class", "activation", "lower", "upper"),
        [
            pytest.param(SigmoidHull, sigmoid_np, 0.2, 2.0, id="sigmoid-positive"),
            pytest.param(SigmoidHull, sigmoid_np, -10.0, -2.0, id="sigmoid-negative"),
            pytest.param(TanhHull, tanh_np, -10.0, -2.0, id="tanh-negative"),
        ],
    )
    def test_multi_neuron_fallback_contains_asymmetric_graph(
        self, hull_class, activation, lower, upper
    ):
        """Verify nearly parallel DLP fallback lines remain conservative."""
        lb = np.array([lower])
        ub = np.array([upper])
        constraints = hull_class().cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        x = np.linspace(lower, upper, 10_001)[:, None]
        points = np.hstack((x, activation(x)))
        margins = constraints[:, :1] + constraints[:, 1:] @ points.T

        assert np.min(margins) >= -1e-8

    @pytest.mark.parametrize(
        ("hull_class", "activation", "lower", "upper"),
        [
            pytest.param(SigmoidHull, sigmoid_np, 90.0, 100.0, id="sigmoid-positive"),
            pytest.param(SigmoidHull, sigmoid_np, -100.0, -90.0, id="sigmoid-negative"),
            pytest.param(SigmoidHullWithOneY, sigmoid_np, 900.0, 1000.0, id="sigmoid-oney"),
            pytest.param(TanhHull, tanh_np, 90.0, 100.0, id="tanh-positive"),
            pytest.param(TanhHullWithOneY, tanh_np, -100.0, -90.0, id="tanh-oney"),
        ],
    )
    def test_saturated_interval_uses_finite_output_range(
        self, hull_class, activation, lower, upper
    ):
        """Verify saturated derivatives degrade to finite codomain bounds."""
        lb = np.array([lower])
        ub = np.array([upper])
        constraints = hull_class().cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        x = np.linspace(lower, upper, 1001)[:, None]
        points = np.hstack((x, activation(x)))
        margins = constraints[:, :1] + constraints[:, 1:] @ points.T

        assert np.all(np.isfinite(constraints))
        assert np.min(margins) >= -1e-8


class TestSShapeMultiDimensional:
    """Test S-shaped activations with various dimensions."""

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(1, 3, id="1d"),
            pytest.param(4, 9, id="4d"),
        ],
    )
    def test_sigmoid_various_dimensions(self, dim, expected_cols, sigmoid_hull_class):
        """Test sigmoid with various input dimensions."""
        lb = -2.0 * np.ones(dim)
        ub = 2.0 * np.ones(dim)

        hull = sigmoid_hull_class()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == expected_cols
        assert np.all(np.isfinite(constraints))

    @pytest.mark.parametrize(
        ("dim", "expected_cols"),
        [
            pytest.param(1, 3, id="1d"),
            pytest.param(4, 9, id="4d"),
        ],
    )
    def test_tanh_various_dimensions(self, dim, expected_cols, tanh_hull_class):
        """Test tanh with various input dimensions."""
        lb = -2.0 * np.ones(dim)
        ub = 2.0 * np.ones(dim)

        hull = tanh_hull_class()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == expected_cols
        assert np.all(np.isfinite(constraints))


class TestSShapeConstraintModes:
    """Test constraint mode combinations for S-shaped activations."""

    @pytest.mark.parametrize(
        ("single_neuron", "multi_neuron", "mode"),
        [
            pytest.param(True, False, "single_only", id="sigmoid_single_only"),
            pytest.param(False, True, "multi_only", id="sigmoid_multi_only"),
        ],
    )
    def test_sigmoid_constraint_modes(self, single_neuron, multi_neuron, mode):
        """Test sigmoid with different constraint mode combinations."""
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull(
            if_cal_single_neuron_constrs=single_neuron,
            if_cal_multi_neuron_constrs=multi_neuron,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints is not None
        assert constraints.shape[0] > 0

    @pytest.mark.parametrize(
        ("single_neuron", "multi_neuron", "mode"),
        [
            pytest.param(True, False, "single_only", id="tanh_single_only"),
            pytest.param(False, True, "multi_only", id="tanh_multi_only"),
        ],
    )
    def test_tanh_constraint_modes(self, single_neuron, multi_neuron, mode):
        """Test tanh with different constraint mode combinations."""
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = TanhHull(
            if_cal_single_neuron_constrs=single_neuron,
            if_cal_multi_neuron_constrs=multi_neuron,
        )

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints is not None
        assert constraints.shape[0] > 0

    @pytest.mark.parametrize(
        ("hull_class", "activation", "lower", "upper"),
        [
            pytest.param(SigmoidHull, sigmoid_np, 0.2, 2.0, id="sigmoid-positive"),
            pytest.param(TanhHull, tanh_np, 0.2, 2.0, id="tanh-positive"),
            pytest.param(SigmoidHull, sigmoid_np, -0.1, 0.1, id="sigmoid-crossing"),
            pytest.param(TanhHull, tanh_np, -0.1, 0.1, id="tanh-crossing"),
        ],
    )
    def test_single_neuron_constraints_contain_graph(self, hull_class, activation, lower, upper):
        """Verify single-neuron inequalities have the correct orientation."""
        hull = hull_class(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )
        constraints = hull.cal_hull(
            input_lower_bounds=np.array([lower]),
            input_upper_bounds=np.array([upper]),
        )

        x = np.linspace(lower, upper, 10_001)[:, None]
        points = np.hstack((x, activation(x)))
        margins = constraints[:, :1] + constraints[:, 1:] @ points.T

        assert np.min(margins) >= -1e-8

    @pytest.mark.parametrize(
        ("hull_class", "activation"),
        [
            pytest.param(SigmoidHull, sigmoid_np, id="sigmoid"),
            pytest.param(TanhHull, tanh_np, id="tanh"),
        ],
    )
    def test_single_neuron_equal_bounds_emit_equality(self, hull_class, activation):
        """Verify a fixed input still produces constraints for its fixed output."""
        bound = np.array([0.5])
        hull = hull_class(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=False,
        )
        constraints = hull.cal_hull(
            input_lower_bounds=bound,
            input_upper_bounds=bound,
        )

        point = np.array([bound[0], activation(bound)[0]])
        margins = constraints[:, 0] + constraints[:, 1:] @ point

        assert constraints.shape == (2, 3)
        np.testing.assert_allclose(margins, 0.0, atol=1e-8)
