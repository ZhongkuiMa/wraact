"""Unit tests for ReLULikeHull base class.

Tests cal_constrs and cal_mn_constrs — the DLP-based multi-neuron constraint
computation inherited by all piecewise-linear acthull variants.
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact.acthull import ReLULikeHull


class TestReLULikeHullInheritance:
    """Test ReLULikeHull inheritance and interface."""

    def test_relu_hull_is_relulike(self, relu_hull_class):
        """ReLUHull must inherit from ReLULikeHull."""
        assert isinstance(relu_hull_class(), ReLULikeHull)

    def test_relulike_has_cal_constrs(self):
        """ReLULikeHull must provide cal_constrs."""
        assert callable(getattr(ReLULikeHull, "cal_constrs", None))

    def test_relulike_has_cal_mn_constrs(self):
        """ReLULikeHull must provide cal_mn_constrs classmethod."""
        assert callable(getattr(ReLULikeHull, "cal_mn_constrs", None))


class TestReLULikeHullOutputShape:
    """Test constraint matrix shape via cal_hull."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_output_columns_match_formula(self, dim, relu_hull_class):
        """Output must have 1 + 2*dim columns (bias + inputs + outputs)."""
        hull = relu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[1] == 1 + 2 * dim, f"Expected {1 + 2 * dim} cols, got {result.shape[1]}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_output_has_positive_row_count(self, dim, relu_hull_class):
        """cal_hull must produce at least one constraint."""
        hull = relu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[0] > 0, "No constraints produced"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_all_constraint_values_finite(self, dim, relu_hull_class):
        """All constraint coefficients must be finite (no NaN or Inf)."""
        hull = relu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result)), "Non-finite values in constraint matrix"


class TestReLULikeHullSoundness:
    """Test that constraints are sound (no valid input-output pair violates them)."""

    def test_valid_relu_pairs_satisfy_constraints_2d(self, relu_hull_class):
        """Valid (x, relu(x)) pairs must satisfy every constraint."""
        hull = relu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        b = result[:, 0]
        a = result[:, 1:]
        rng = np.random.default_rng(0)

        for _ in range(200):
            x = rng.uniform(lb, ub)
            y = np.maximum(x, 0.0)
            point = np.concatenate([x, y])
            assert np.all(b + a @ point >= -1e-8), f"Point {point} violates constraints"

    def test_asymmetric_bounds_produce_finite_constraints(self, relu_hull_class):
        """Asymmetric bounds must yield a finite constraint matrix."""
        hull = relu_hull_class()
        lb = np.array([-2.0, -0.5])
        ub = np.array([0.5, 3.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[1] == 5  # 1 + 2*2
        assert np.all(np.isfinite(result))
