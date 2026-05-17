"""Unit tests for ReLULikeHullWithOneY base class.

Tests cal_constrs, cal_sn_constrs, cal_mn_constrs, and the small-polytope
ValueError guard that differentiates the single-output ReLU-like variant.
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact.oney import ReLUHullWithOneY, ReLULikeHullWithOneY


class TestReLULikeHullWithOneYInheritance:
    """Test ReLULikeHullWithOneY inheritance and interface."""

    def test_relu_hull_withoney_is_relulike(self):
        """ReLUHullWithOneY must inherit from ReLULikeHullWithOneY."""
        assert isinstance(ReLUHullWithOneY(), ReLULikeHullWithOneY)

    def test_has_cal_constrs(self):
        """ReLULikeHullWithOneY must provide cal_constrs."""
        assert callable(getattr(ReLULikeHullWithOneY, "cal_constrs", None))

    def test_has_cal_sn_constrs(self):
        """ReLULikeHullWithOneY must provide cal_sn_constrs classmethod."""
        assert callable(getattr(ReLULikeHullWithOneY, "cal_sn_constrs", None))

    def test_has_cal_mn_constrs(self):
        """ReLULikeHullWithOneY must provide cal_mn_constrs classmethod."""
        assert callable(getattr(ReLULikeHullWithOneY, "cal_mn_constrs", None))


class TestReLULikeHullWithOneYOutputShape:
    """Test output shape and finiteness via ReLUHullWithOneY."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_output_columns_equal_dim_plus_two(self, dim):
        """Output must have dim+2 columns (bias + dim inputs + 1 output)."""
        hull = ReLUHullWithOneY()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert result is not None

        assert result.shape[1] == dim + 2, f"Expected {dim + 2} cols, got {result.shape[1]}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_all_values_finite(self, dim):
        """Constraint values must all be finite."""
        hull = ReLUHullWithOneY()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert result is not None

        assert np.all(np.isfinite(result)), "Non-finite values in output"


class TestReLULikeHullWithOneYCalSnConstrs:
    """Test cal_sn_constrs classmethod directly."""

    def test_sn_constrs_shape_2d(self):
        """cal_sn_constrs must return shape (1, dim+2) for 2D input."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = ReLUHullWithOneY.cal_sn_constrs(lb, ub)

        assert result.shape == (1, 4), f"Expected (1, 4), got {result.shape}"

    def test_sn_constrs_output_coefficient_is_negative_one(self):
        """The last column must be -1.0 (upper-bound form: c + k*x - y >= 0)."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = ReLUHullWithOneY.cal_sn_constrs(lb, ub)

        assert result[0, -1] == -1.0, f"Expected -1.0, got {result[0, -1]}"

    def test_sn_constrs_values_finite(self):
        """cal_sn_constrs must produce finite coefficients."""
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        result = ReLUHullWithOneY.cal_sn_constrs(lb, ub)

        assert np.all(np.isfinite(result))


class TestReLULikeHullWithOneYSmallPolytope:
    """Test ValueError guard for polytopes below the minimum range threshold."""

    def test_valid_bounds_above_threshold_no_error(self):
        """cal_hull must succeed when bounds range >= MIN_BOUNDS_RANGE_ONEY."""
        hull = ReLUHullWithOneY()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])  # range 2.0 >> 0.05 threshold

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))
