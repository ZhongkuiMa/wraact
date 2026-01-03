"""Final push to 95% coverage - targeting remaining gaps.

Tests for hard-to-reach code paths in:
- acthull/_act.py: Complex polytope handling
- oney/_act.py: OneY base class edge cases
- _tangent_lines.py: Numerical edge cases
"""

__docformat__ = "restructuredtext"

import numpy as np


class TestActHullComplexPolytopes:
    """Test ActHull with complex polytope configurations."""

    def test_relu_varying_bound_magnitudes(self):
        """Test ReLU with very different bound magnitudes."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()
        # Bounds with very different magnitudes per dimension
        lb = np.array([-0.001, -1000.0, -0.1])
        ub = np.array([10.0, 2000.0, 100.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_near_zero_mixed_bounds(self):
        """Test LeakyReLU with bounds near zero mixed with large bounds."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull()
        lb = np.array([-0.05, -1000.0, -0.025])
        ub = np.array([0.05, 1000.0, 0.025])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_mixed_positive_negative_bounds(self):
        """Test Sigmoid with mixed positive/negative bounds."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()
        lb = np.array([-10.0, -0.05, -5.0])
        ub = np.array([-5.0, 0.05, 5.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_tanh_extreme_crossing_points(self):
        """Test Tanh with bounds crossing zero at extreme values."""
        from wraact.acthull import TanhHull

        hull = TanhHull()
        lb = np.array([-100.0, -0.0001, -10.0])
        ub = np.array([0.0001, 100.0, 10.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_all_dimensions_extreme(self):
        """Test MaxPool with all dimensions at extreme values."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-1000.0, -0.05, -10.0, 1.0])
        ub = np.array([-100.0, 0.05, 10.0, 1000.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullParameterCombinations:
    """Test ActHull with various parameter combinations."""

    def test_relu_single_neuron_multi_neuron_consistency(self):
        """Test that single + multi modes produce more constraints than either alone."""
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        # Single only
        hull_single = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        c_single = hull_single.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Multi only
        hull_multi = ReLUHull(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)
        c_multi = hull_multi.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Both
        hull_both = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)
        c_both = hull_both.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Both should have >= constraints from either mode
        assert c_both.shape[0] >= c_single.shape[0]
        assert c_both.shape[0] >= c_multi.shape[0]

    def test_leakyrelu_different_initializations(self):
        """Test LeakyReLU with different initialization parameters."""
        from wraact.oney import LeakyReLUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Different dtype_cdd
        for dtype in ["float", "fraction"]:
            hull = LeakyReLUHullWithOneY(dtype_cdd=dtype)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)

    def test_sigmoid_oney_with_output_constraints(self):
        """Test Sigmoid OneY with various output constraint counts."""
        from wraact.oney import SigmoidHullWithOneY

        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        for n_constrs in [1, 2, 3]:
            hull = SigmoidHullWithOneY(n_output_constraints=n_constrs)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))


class TestActHullBoundaryValues:
    """Test ActHull with boundary and extreme values."""

    def test_relu_symmetric_around_zero(self):
        """Test ReLU with perfectly symmetric bounds around zero."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()
        for magnitude in [0.05, 0.5, 1.0, 10.0, 100.0]:
            lb = -magnitude * np.ones(4)
            ub = magnitude * np.ones(4)

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))

    def test_leakyrelu_asymmetric_scaling(self):
        """Test LeakyReLU with asymmetric scaling in each dimension."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull()
        # Each dimension has different magnitudes
        scales = [0.1, 1.0, 10.0, 100.0]
        lb = -np.array(scales)
        ub = np.array(scales) * 2

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_elu_crossing_zero_various_asymmetries(self):
        """Test ELU with various asymmetric zero-crossings."""
        from wraact.acthull import ELUHull

        hull = ELUHull()
        test_cases = [
            (np.array([-10.0, -1.0, -0.1]), np.array([0.1, 1.0, 10.0])),
            (np.array([-0.1, -0.1, -0.1]), np.array([10.0, 100.0, 1000.0])),
            (np.array([-1000.0, -100.0, -10.0]), np.array([0.1, 0.01, 0.001])),
        ]

        for lb, ub in test_cases:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))


class TestActHullNumericalStability:
    """Test ActHull numerical stability with challenging inputs."""

    def test_sigmoid_very_large_inputs(self):
        """Test Sigmoid with very large input bounds."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()
        lb = np.array([-1e6, -1e5, -1e4])
        ub = np.array([1e4, 1e5, 1e6])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert np.all(np.isfinite(constraints))

    def test_tanh_symmetric_extreme_bounds(self):
        """Test Tanh with symmetric extreme bounds."""
        from wraact.acthull import TanhHull

        hull = TanhHull()
        for exp in [2, 3, 4, 5]:
            magnitude = 10.0**exp
            lb = -magnitude * np.ones(2)
            ub = magnitude * np.ones(2)

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))

    def test_maxpool_mixed_magnitude_bounds(self):
        """Test MaxPool with mixed magnitude bounds."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        lb = np.array([-0.05, -1.0, -1e6])
        ub = np.array([0.05, 1.0, 1e6])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert np.all(np.isfinite(constraints))


class TestOneYVariantFinalTests:
    """Final comprehensive tests for OneY variants."""

    def test_all_oney_variants_consistency(self):
        """Test all OneY variants produce consistent results."""
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
            c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            np.testing.assert_array_equal(c1, c2)

    def test_oney_parameter_combinations(self):
        """Test OneY variants with various parameter combinations."""
        from wraact.oney import ReLUHullWithOneY

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        for dtype_cdd in ["float", "fraction"]:
            for n_out_constrs in [1, 2]:
                hull = ReLUHullWithOneY(
                    dtype_cdd=dtype_cdd,
                    n_output_constraints=n_out_constrs,
                    if_return_input_bounds_by_vertices=False,
                )
                constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
                assert isinstance(constraints, np.ndarray)
