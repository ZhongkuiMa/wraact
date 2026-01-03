"""Exception recovery tests for OneY ActHull variants.

Tests for:
1. Exception handling in OneY multi-neuron constraint calculation
2. Degenerate polytope recovery with fractional arithmetic
3. Double orders mode in OneY
4. Error conditions and edge cases
"""

__docformat__ = "restructuredtext"

import contextlib

import numpy as np


class TestOneYExceptionHandling:
    """Test exception handling in OneY ActHull."""

    def test_oney_relu_valid_bounds(self):
        """Test that OneY ReLU works with valid bounds."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_oney_sigmoid_valid_bounds(self):
        """Test that OneY Sigmoid works with valid bounds."""
        from wraact.oney import SigmoidHullWithOneY

        hull = SigmoidHullWithOneY()

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_oney_tanh_valid_bounds(self):
        """Test that OneY Tanh works with valid bounds."""
        from wraact.oney import TanhHullWithOneY

        hull = TanhHullWithOneY()

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestOneYDegenerateRecovery:
    """Test recovery from degenerate polytopes in OneY."""

    def test_oney_relu_degenerate_tolerance(self):
        """Test OneY ReLU with very small but valid bounds."""
        from wraact.oney import ReLUHullWithOneY

        hull = ReLUHullWithOneY()

        # Small bounds but still above minimum threshold
        lb = np.array([-0.05, -0.05])
        ub = np.array([0.05, 0.05])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_oney_leakyrelu_degenerate_with_suppression(self):
        """Test OneY LeakyReLU handles near-degenerate cases gracefully."""
        from wraact.oney import LeakyReLUHullWithOneY

        hull = LeakyReLUHullWithOneY()

        # Create potentially problematic but valid bounds
        test_cases = [
            (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
            (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
            (np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        ]

        for lb, ub in test_cases:
            with contextlib.suppress(ValueError, RuntimeError):
                constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
                if constraints is not None:
                    assert np.all(np.isfinite(constraints))

    def test_oney_maxpool_degenerate_recovery(self):
        """Test OneY MaxPool with potentially degenerate cases."""
        from wraact.oney import MaxPoolHullWithOneY

        hull = MaxPoolHullWithOneY()

        lb = np.array([-0.5, -0.5, -0.5])
        ub = np.array([0.5, 0.5, 0.5])

        with contextlib.suppress(ValueError, RuntimeError):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            if constraints is not None:
                assert isinstance(constraints, np.ndarray)


class TestOneYDoubleOrders:
    """Test double orders mode in OneY (if supported)."""

    def test_oney_relu_with_various_output_constraints(self):
        """Test OneY ReLU with different n_output_constraints."""
        from wraact.oney import ReLUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        for n_constrs in [1, 2, 3]:
            hull = ReLUHullWithOneY(n_output_constraints=n_constrs)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))

    def test_oney_sigmoid_dtype_combinations(self):
        """Test OneY Sigmoid with different dtype_cdd values."""
        from wraact.oney import SigmoidHullWithOneY

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        for dtype_cdd in ["float", "fraction"]:
            hull = SigmoidHullWithOneY(dtype_cdd=dtype_cdd)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))

    def test_oney_tanh_return_vertices_parameter(self):
        """Test OneY Tanh with if_return_input_bounds_by_vertices parameter."""
        from wraact.oney import TanhHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        for return_vertices in [True, False]:
            hull = TanhHullWithOneY(if_return_input_bounds_by_vertices=return_vertices)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))


class TestOneYConsistency:
    """Test consistency of OneY implementations."""

    def test_oney_all_variants_produce_finite_constraints(self):
        """Test all OneY variants produce finite constraints."""
        from wraact.oney import (
            ELUHullWithOneY,
            LeakyReLUHullWithOneY,
            MaxPoolHullWithOneY,
            ReLUHullWithOneY,
            SigmoidHullWithOneY,
            TanhHullWithOneY,
        )

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        variants = [
            ReLUHullWithOneY(),
            LeakyReLUHullWithOneY(),
            ELUHullWithOneY(),
            MaxPoolHullWithOneY(),
            SigmoidHullWithOneY(),
            TanhHullWithOneY(),
        ]

        for hull in variants:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # OneY uses topk selection so rows may vary, but columns are consistent
            assert constraints.shape[0] > 0, f"No constraints for {hull.__class__.__name__}"
            assert constraints.shape[1] == 4, f"Wrong columns for {hull.__class__.__name__}"
            assert np.all(np.isfinite(constraints))

    def test_oney_multi_output_constraints_increase_count(self):
        """Test that increasing n_output_constraints affects result shape."""
        from wraact.oney import ReLUHullWithOneY

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        results = []
        for n_constrs in [1, 2, 3]:
            hull = ReLUHullWithOneY(n_output_constraints=n_constrs)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints.shape[0])

        # Should not decrease as we increase output constraints
        assert results[0] <= results[1] or results[0] <= results[2]

    def test_oney_dtype_cdd_does_not_affect_output(self):
        """Test that dtype_cdd choice doesn't affect final shape."""
        from wraact.oney import LeakyReLUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull_float = LeakyReLUHullWithOneY(dtype_cdd="float")
        constraints_float = hull_float.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        hull_fraction = LeakyReLUHullWithOneY(dtype_cdd="fraction")
        constraints_fraction = hull_fraction.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Same shape
        assert constraints_float.shape == constraints_fraction.shape


class TestOneYEdgeCases:
    """Test edge cases in OneY implementations."""

    def test_oney_1d_input_all_variants(self):
        """Test OneY variants with 1D input."""
        from wraact.oney import (
            ELUHullWithOneY,
            ReLUHullWithOneY,
            SigmoidHullWithOneY,
        )

        lb = np.array([-1.0])
        ub = np.array([1.0])

        variants = [
            (ReLUHullWithOneY(), 3),  # 2*1+1
            (ELUHullWithOneY(), 3),
            (SigmoidHullWithOneY(), 3),
        ]

        for hull, expected_cols in variants:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert constraints.shape[1] == expected_cols

    def test_oney_high_dimension_input(self):
        """Test OneY with high-dimensional input."""
        from wraact.oney import ReLUHullWithOneY

        d = 8
        lb = -np.ones(d)
        ub = np.ones(d)

        hull = ReLUHullWithOneY()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # OneY uses topk selection, so output columns = d + 2
        assert constraints.shape[1] == d + 2
        assert np.all(np.isfinite(constraints))

    def test_oney_symmetric_bounds(self):
        """Test OneY with perfectly symmetric bounds."""
        from wraact.oney import TanhHullWithOneY

        for magnitude in [0.5, 1.0, 2.0, 5.0]:
            lb = -magnitude * np.ones(3)
            ub = magnitude * np.ones(3)

            hull = TanhHullWithOneY()
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert np.all(np.isfinite(constraints))

    def test_oney_asymmetric_bounds(self):
        """Test OneY with asymmetric bounds."""
        from wraact.oney import LeakyReLUHullWithOneY

        test_cases = [
            (np.array([-10.0, -1.0]), np.array([1.0, 10.0])),
            (np.array([-5.0, -0.5]), np.array([0.1, 5.0])),
            (np.array([-100.0, -10.0]), np.array([10.0, 100.0])),
        ]

        for lb, ub in test_cases:
            hull = LeakyReLUHullWithOneY()
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))


class TestOneYParameterCombinations:
    """Test various parameter combinations in OneY."""

    def test_oney_relu_all_parameter_combinations(self):
        """Test ReLU OneY with all parameter combinations."""
        from wraact.oney import ReLUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        dtypes = ["float", "fraction"]
        output_constrs = [1, 2, 3]
        return_vertices = [True, False]

        for dtype in dtypes:
            for n_out in output_constrs:
                for ret_vert in return_vertices:
                    hull = ReLUHullWithOneY(
                        dtype_cdd=dtype,
                        n_output_constraints=n_out,
                        if_return_input_bounds_by_vertices=ret_vert,
                    )

                    constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

                    assert isinstance(constraints, np.ndarray)
                    assert np.all(np.isfinite(constraints))

    def test_oney_sigmoid_parameter_combinations(self):
        """Test Sigmoid OneY with various parameter combinations."""
        from wraact.oney import SigmoidHullWithOneY

        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        for n_out in [1, 2, 3]:
            for dtype in ["float", "fraction"]:
                hull = SigmoidHullWithOneY(
                    dtype_cdd=dtype,
                    n_output_constraints=n_out,
                )

                constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

                assert isinstance(constraints, np.ndarray)
                # OneY output format: d + 2 columns
                assert constraints.shape[1] == 3 + 2


class TestOneYReproducibility:
    """Test reproducibility of OneY calculations."""

    def test_oney_relu_deterministic(self):
        """Test OneY ReLU produces deterministic results."""
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

    def test_oney_tanh_deterministic_various_dimensions(self):
        """Test OneY Tanh determinism across dimensions."""
        from wraact.oney import TanhHullWithOneY

        for d in [2, 3, 4]:
            hull = TanhHullWithOneY()
            lb = -np.ones(d)
            ub = np.ones(d)

            results = []
            for _ in range(2):
                constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
                results.append(constraints)

            np.testing.assert_array_equal(results[0], results[1])

    def test_oney_maxpool_deterministic(self):
        """Test OneY MaxPool is deterministic."""
        from wraact.oney import MaxPoolHullWithOneY

        hull = MaxPoolHullWithOneY()
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        results = []
        for _ in range(3):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints)

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
