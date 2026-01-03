"""OneY ActHull base class tests targeting coverage gaps.

Tests for ActHullWithOneY base class functionality and error handling.
"""

__docformat__ = "restructuredtext"

import contextlib

import numpy as np


class TestActHullWithOneYErrorHandling:
    """Test error handling in ActHullWithOneY."""

    def test_relu_oney_small_bounds(self):
        """Test ReLU OneY with small but valid bounds."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        # Small bounds that exceed minimum range threshold (0.05)
        lb = np.array([-0.05, -0.05])
        ub = np.array([0.05, 0.05])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_oney_degenerated_error_handling(self):
        """Test ActHullWithOneY handles degenerated polytopes."""
        from wraact.oney import LeakyReLUHullWithOneY

        hull = LeakyReLUHullWithOneY()
        # Constant bounds - degenerate polytope
        lb = np.array([0.5, 0.5])
        ub = np.array([0.5, 0.5])

        with contextlib.suppress(ValueError, RuntimeError):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)


class TestActHullWithOneYHighDimensional:
    """Test ActHullWithOneY with high-dimensional inputs."""

    def test_relu_oney_6d_input(self):
        """Test ReLU OneY with 6D input."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        lb = -np.ones(6)
        ub = np.ones(6)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 8  # 6 inputs + 1 bias + 1 output
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_oney_5d_input(self):
        """Test LeakyReLU OneY with 5D input."""
        from wraact.oney import LeakyReLUHullWithOneY

        hull = LeakyReLUHullWithOneY()
        lb = -2.0 * np.ones(5)
        ub = 2.0 * np.ones(5)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 7  # 5 inputs + 1 bias + 1 output
        assert np.all(np.isfinite(constraints))

    def test_elu_oney_4d_input(self):
        """Test ELU OneY with 4D input."""
        from wraact.oney import ELUHullWithOneY

        hull = ELUHullWithOneY()
        lb = -1.5 * np.ones(4)
        ub = 1.5 * np.ones(4)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 6  # 4 inputs + 1 bias + 1 output
        assert np.all(np.isfinite(constraints))


class TestActHullWithOneYAsymmetricBounds:
    """Test ActHullWithOneY with asymmetric bounds."""

    def test_relu_oney_asymmetric_2d(self):
        """Test ReLU OneY with asymmetric bounds."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        lb = np.array([-5.0, -1.0])
        ub = np.array([1.0, 5.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_oney_asymmetric_3d(self):
        """Test LeakyReLU OneY with very asymmetric bounds."""
        from wraact.oney import LeakyReLUHullWithOneY

        hull = LeakyReLUHullWithOneY()
        lb = np.array([-10.0, -0.5, -100.0])
        ub = np.array([0.5, 10.0, 5.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_oney_asymmetric_4d(self):
        """Test Sigmoid OneY with asymmetric bounds."""
        from wraact.oney import SigmoidHullWithOneY

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
        from wraact.oney import ReLUHullWithOneY

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
        from wraact.oney import LeakyReLUHullWithOneY

        for n_constrs in [1, 2]:
            hull = LeakyReLUHullWithOneY(n_output_constraints=n_constrs)
            lb = np.array([-1.0, -1.0])
            ub = np.array([1.0, 1.0])

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))


class TestActHullWithOneYDtypeCddParameter:
    """Test dtype_cdd parameter in ActHullWithOneY."""

    def test_relu_oney_dtype_float(self):
        """Test ReLU OneY with float dtype for CDD."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY(dtype_cdd="float")
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_oney_dtype_fraction(self):
        """Test LeakyReLU OneY with fraction dtype for CDD."""
        from wraact.oney import LeakyReLUHullWithOneY

        hull = LeakyReLUHullWithOneY(dtype_cdd="fraction")
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullWithOneYReturnVertices:
    """Test return_input_bounds_by_vertices parameter."""

    def test_relu_oney_return_input_bounds_by_vertices(self):
        """Test ReLU OneY with return_input_bounds_by_vertices parameter."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY(if_return_input_bounds_by_vertices=True)
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_oney_return_input_bounds_by_vertices(self):
        """Test LeakyReLU OneY with return_input_bounds_by_vertices parameter."""
        from wraact.oney import LeakyReLUHullWithOneY

        hull = LeakyReLUHullWithOneY(if_return_input_bounds_by_vertices=True)
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullWithOneYTopkConstraintSelection:
    """Test topk constraint selection in ActHullWithOneY."""

    def test_maxpool_oney_topk_selection(self):
        """Test MaxPool OneY constraint selection using topk method."""
        from wraact.oney import MaxPoolHullWithOneY

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
        from wraact.oney import SigmoidHullWithOneY

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
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        results = []
        for _ in range(3):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints)

        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_tanh_oney_reproducible(self):
        """Test Tanh OneY is reproducible."""
        from wraact.oney import TanhHullWithOneY

        hull = TanhHullWithOneY()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        results = []
        for _ in range(3):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints)

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
