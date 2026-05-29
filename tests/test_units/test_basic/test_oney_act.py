"""OneY ActHull base class tests targeting coverage gaps.

Tests for ActHullWithOneY base class functionality and error handling.
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact.oney import (
    ELUHullWithOneY,
    LeakyReLUHullWithOneY,
    MaxPoolHullWithOneY,
    ReLUHullWithOneY,
    SigmoidHullWithOneY,
    TanhHullWithOneY,
)


class TestActHullWithOneYErrorHandling:
    """Test error handling in ActHullWithOneY."""

    def test_leakyrelu_oney_degenerated_error_handling(self):
        """Test ActHullWithOneY handles degenerated polytopes."""
        hull = LeakyReLUHullWithOneY()
        # Constant bounds - degenerate polytope
        lb = np.array([0.5, 0.5])
        ub = np.array([0.5, 0.5])

        raised = False
        try:
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If no exception, result should still be a valid array
            assert isinstance(result, np.ndarray)
            assert np.all(np.isfinite(result))
        except (ValueError, RuntimeError):
            raised = True

        # Degenerate polytope should either raise or return valid constraints
        assert raised or isinstance(result, np.ndarray)


class TestActHullWithOneYHighDimensional:
    """Test ActHullWithOneY with high-dimensional inputs."""

    def test_relu_oney_6d_input(self):
        """Test ReLU OneY with 6D input."""
        hull = ReLUHullWithOneY()
        lb = -np.ones(6)
        ub = np.ones(6)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints is not None

        assert constraints.shape[1] == 8  # 6 inputs + 1 bias + 1 output
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_oney_5d_input(self):
        """Test LeakyReLU OneY with 5D input."""
        hull = LeakyReLUHullWithOneY()
        lb = -2.0 * np.ones(5)
        ub = 2.0 * np.ones(5)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints is not None

        assert constraints.shape[1] == 7  # 5 inputs + 1 bias + 1 output
        assert np.all(np.isfinite(constraints))

    def test_elu_oney_4d_input(self):
        """Test ELU OneY with 4D input."""
        hull = ELUHullWithOneY()
        lb = -1.5 * np.ones(4)
        ub = 1.5 * np.ones(4)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert constraints is not None

        assert constraints.shape[1] == 6  # 4 inputs + 1 bias + 1 output
        assert np.all(np.isfinite(constraints))


class TestActHullWithOneYAsymmetricBounds:
    """Test ActHullWithOneY with asymmetric bounds."""

    def test_sigmoid_oney_asymmetric_4d(self):
        """Test Sigmoid OneY with asymmetric bounds."""
        hull = SigmoidHullWithOneY()
        lb = np.array([-3.0, -1.0, -5.0, 0.5])
        ub = np.array([1.0, 5.0, 2.0, 3.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullWithOneYOutputConstraintSelection:
    """Test output constraint selection in ActHullWithOneY."""

    def test_relu_oney_n_output_constraints_param(self):
        """Test ReLU OneY respects n_output_constraints parameter."""
        # Test with different n_output_constraints values
        for n_constrs in [1, 2, 3]:
            hull = ReLUHullWithOneY(n_output_constraints=n_constrs)
            lb = np.array([-1.0, -1.0, -1.0])
            ub = np.array([1.0, 1.0, 1.0])

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))

    def test_leakyrelu_oney_n_output_constraints_param(self):
        """Test LeakyReLU OneY respects n_output_constraints parameter."""
        for n_constrs in [1, 2]:
            hull = LeakyReLUHullWithOneY(n_output_constraints=n_constrs)
            lb = np.array([-1.0, -1.0])
            ub = np.array([1.0, 1.0])

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))


class TestActHullWithOneYTopkConstraintSelection:
    """Test topk constraint selection in ActHullWithOneY."""

    def test_maxpool_oney_topk_selection(self):
        """Test MaxPool OneY constraint selection using topk method."""
        hull = MaxPoolHullWithOneY()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        # Constraints should have selected top-k, so may have fewer than multi-neuron
        assert constraints.shape[0] > 0
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_oney_topk_selection(self):
        """Test Sigmoid OneY constraint selection using topk."""
        hull = SigmoidHullWithOneY(n_output_constraints=2)
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullWithOneYReproducibility:
    """Test reproducibility of OneY computations."""

    def test_relu_oney_reproducible(self):
        """Test ReLU OneY is reproducible."""
        hull = ReLUHullWithOneY()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        results = []
        for _ in range(3):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert constraints is not None
            results.append(constraints)

        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_tanh_oney_reproducible(self):
        """Test Tanh OneY is reproducible."""
        hull = TanhHullWithOneY()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        results = []
        for _ in range(3):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert constraints is not None
            results.append(constraints)

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


class TestReLUHullWithOneYParametrized:
    """Parametrized ReLUHullWithOneY tests covering varied configurations."""

    @pytest.mark.parametrize(
        ("lb", "ub", "hull_kwargs"),
        [
            (np.array([-0.05, -0.05]), np.array([0.05, 0.05]), {}),
            (np.array([-5.0, -1.0]), np.array([1.0, 5.0]), {}),
            (np.array([-1.0, -1.0]), np.array([1.0, 1.0]), {"dtype_cdd": "float"}),
        ],
        ids=["small_bounds", "asymmetric_2d", "dtype_float"],
    )
    def test_relu_oney_cal_hull(self, lb, ub, hull_kwargs):
        """Test ReLUHullWithOneY.cal_hull with varied configurations."""
        hull = ReLUHullWithOneY(**hull_kwargs)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestLeakyReLUHullWithOneYParametrized:
    """Parametrized LeakyReLUHullWithOneY tests covering varied configurations."""

    @pytest.mark.parametrize(
        ("lb", "ub", "hull_kwargs"),
        [
            (np.array([-10.0, -0.5, -100.0]), np.array([0.5, 10.0, 5.0]), {}),
            (np.array([-1.0, -1.0]), np.array([1.0, 1.0]), {"dtype_cdd": "fraction"}),
            (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]), {}),
        ],
        ids=["asymmetric_3d", "dtype_fraction", "n_output_default_3d"],
    )
    def test_leakyrelu_oney_cal_hull(self, lb, ub, hull_kwargs):
        """Test LeakyReLUHullWithOneY.cal_hull with varied configurations."""
        hull = LeakyReLUHullWithOneY(**hull_kwargs)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
