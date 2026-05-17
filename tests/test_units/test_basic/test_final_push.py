"""Final push to 95% coverage - targeting remaining gaps.

Tests for hard-to-reach code paths in:
- acthull/_act.py: Complex polytope handling
- oney/_act.py: OneY base class edge cases
- _tangent_lines.py: Numerical edge cases
"""

__docformat__ = "restructuredtext"

from typing import Literal

import numpy as np

from wraact.oney import (
    ELUHullWithOneY,
    LeakyReLUHullWithOneY,
    MaxPoolHullWithOneY,
    ReLUHullWithOneY,
    SigmoidHullWithOneY,
    TanhHullWithOneY,
)


class TestActHullComplexPolytopes:
    """Test ActHull with complex polytope configurations."""

    def test_relu_varying_bound_magnitudes(self, relu_hull_class):
        """Test ReLU with very different bound magnitudes."""
        hull = relu_hull_class()
        # Bounds with very different magnitudes per dimension
        lb = np.array([-0.001, -1000.0, -0.1])
        ub = np.array([10.0, 2000.0, 100.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_near_zero_mixed_bounds(self, leakyrelu_hull_class):
        """Test LeakyReLU with bounds near zero mixed with large bounds."""
        hull = leakyrelu_hull_class()
        lb = np.array([-0.05, -1000.0, -0.025])
        ub = np.array([0.05, 1000.0, 0.025])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_mixed_positive_negative_bounds(self, sigmoid_hull_class):
        """Test Sigmoid with mixed positive/negative bounds."""
        hull = sigmoid_hull_class()
        lb = np.array([-10.0, -0.05, -5.0])
        ub = np.array([-5.0, 0.05, 5.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_tanh_extreme_crossing_points(self, tanh_hull_class):
        """Test Tanh with bounds crossing zero at extreme values."""
        hull = tanh_hull_class()
        lb = np.array([-100.0, -0.0001, -10.0])
        ub = np.array([0.0001, 100.0, 10.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_all_dimensions_extreme(self, maxpool_hull_class):
        """Test MaxPool with all dimensions at extreme values."""
        hull = maxpool_hull_class()
        lb = np.array([-1000.0, -0.05, -10.0, 1.0])
        ub = np.array([-100.0, 0.05, 10.0, 1000.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullParameterCombinations:
    """Test ActHull with various parameter combinations."""

    def test_relu_single_neuron_multi_neuron_consistency(self, relu_hull_class):
        """Test that single + multi modes produce more constraints than either alone."""
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        # Single only
        hull_single = relu_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False
        )
        c_single = hull_single.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Multi only
        hull_multi = relu_hull_class(
            if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True
        )
        c_multi = hull_multi.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Both
        hull_both = relu_hull_class(
            if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True
        )
        c_both = hull_both.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Both should have >= constraints from either mode
        assert c_both.shape[0] >= c_single.shape[0]
        assert c_both.shape[0] >= c_multi.shape[0]

    def test_leakyrelu_different_initializations(self):
        """Test LeakyReLU with different initialization parameters."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Different dtype_cdd
        dtype_opts: list[Literal["fraction", "float"]] = ["float", "fraction"]
        for dtype in dtype_opts:
            hull = LeakyReLUHullWithOneY(dtype_cdd=dtype)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)

    def test_sigmoid_oney_with_output_constraints(self):
        """Test Sigmoid OneY with various output constraint counts."""
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        for n_constrs in [1, 2, 3]:
            hull = SigmoidHullWithOneY(n_output_constraints=n_constrs)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))


class TestActHullBoundaryValues:
    """Test ActHull with boundary and extreme values."""

    def test_relu_symmetric_around_zero(self, relu_hull_class):
        """Test ReLU with perfectly symmetric bounds around zero."""
        hull = relu_hull_class()
        for magnitude in [0.05, 0.5, 1.0, 10.0, 100.0]:
            lb = -magnitude * np.ones(4)
            ub = magnitude * np.ones(4)

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))

    def test_leakyrelu_asymmetric_scaling(self, leakyrelu_hull_class):
        """Test LeakyReLU with asymmetric scaling in each dimension."""
        hull = leakyrelu_hull_class()
        # Each dimension has different magnitudes
        scales = [0.1, 1.0, 10.0, 100.0]
        lb = -np.array(scales)
        ub = np.array(scales) * 2

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_elu_crossing_zero_various_asymmetries(self, elu_hull_class):
        """Test ELU with various asymmetric zero-crossings."""
        hull = elu_hull_class()
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

    def test_sigmoid_very_large_inputs(self, sigmoid_hull_class):
        """Test Sigmoid with very large input bounds."""
        hull = sigmoid_hull_class()
        lb = np.array([-1e6, -1e5, -1e4])
        ub = np.array([1e4, 1e5, 1e6])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert np.all(np.isfinite(constraints))

    def test_tanh_symmetric_extreme_bounds(self, tanh_hull_class):
        """Test Tanh with symmetric extreme bounds."""
        hull = tanh_hull_class()
        for exp in [2, 3, 4, 5]:
            magnitude = 10.0**exp
            lb = -magnitude * np.ones(2)
            ub = magnitude * np.ones(2)

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))

    def test_maxpool_mixed_magnitude_bounds(self, maxpool_hull_class):
        """Test MaxPool with mixed magnitude bounds."""
        hull = maxpool_hull_class()
        lb = np.array([-0.05, -1.0, -1e6])
        ub = np.array([0.05, 1.0, 1e6])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert np.all(np.isfinite(constraints))


class TestOneYVariantFinalTests:
    """Final comprehensive tests for OneY variants."""

    def test_all_oney_variants_consistency(self):
        """Test all OneY variants produce consistent results."""
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
            assert c1 is not None
            assert c2 is not None
            np.testing.assert_array_equal(c1, c2)

    def test_oney_parameter_combinations(self):
        """Test OneY variants with various parameter combinations."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        dtype_cdd_opts: list[Literal["fraction", "float"]] = ["float", "fraction"]
        for dtype_cdd in dtype_cdd_opts:
            for n_out_constrs in [1, 2]:
                hull = ReLUHullWithOneY(
                    dtype_cdd=dtype_cdd,
                    n_output_constraints=n_out_constrs,
                    if_return_input_bounds_by_vertices=False,
                )
                constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
                assert isinstance(constraints, np.ndarray)
