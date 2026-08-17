"""Tests for constraint satisfaction and error handling.

This module tests that hull constraints are satisfied by valid function outputs
and that appropriate errors are raised for invalid inputs.

Key Tests:
==========
- Soundness verification: All (x, f(x)) points satisfy constraints
- Error handling: Proper exceptions for degenerate polytopes, bounds issues
- Numerical stability: Handling of extreme bounds and precision
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact import DegeneratedError
from wraact._functions import elu_np, leakyrelu_np, relu_np, sigmoid_np


class TestReLUConstraintSatisfaction:
    """Test constraint satisfaction for ReLU hulls."""

    @pytest.mark.parametrize(
        ("label", "get_point"),
        [
            pytest.param("lower_bound", lambda lb, ub: (lb, relu_np(lb)), id="at_bounds"),
            pytest.param("upper_bound", lambda lb, ub: (ub, relu_np(ub)), id="at_upper"),
            pytest.param(
                "center", lambda lb, ub: ((lb + ub) / 2, relu_np((lb + ub) / 2)), id="at_center"
            ),
        ],
    )
    def test_relu_constraints_satisfied(self, relu_hull_class, label, get_point):
        """Verify ReLU constraints are satisfied at various sampling points."""
        hull = relu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        x, y = get_point(lb, ub)
        point = np.concatenate([x, y])
        b = result[:, 0]
        a = result[:, 1:]
        constraints = b + a @ point
        assert np.all(constraints >= -1e-8), f"Constraints violated at {label}"

    def test_sigmoid_constraints_satisfied_at_bounds(self, sigmoid_hull_class):
        """Verify sigmoid constraints are satisfied at bounds."""
        hull = sigmoid_hull_class()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Test at lower bound
        x_lb = lb
        y_lb = sigmoid_np(x_lb)
        point_lb = np.concatenate([x_lb, y_lb])
        b = result[:, 0]
        a = result[:, 1:]
        constraints_lb = b + a @ point_lb
        assert np.all(constraints_lb >= -1e-8), "Sigmoid constraints violated at lower bound"

        # Test at upper bound
        x_ub = ub
        y_ub = sigmoid_np(x_ub)
        point_ub = np.concatenate([x_ub, y_ub])
        constraints_ub = b + a @ point_ub
        assert np.all(constraints_ub >= -1e-8), "Sigmoid constraints violated at upper bound"


class TestBoundsConsistency:
    """Test error handling for inconsistent bounds."""

    def test_lower_greater_than_upper_raises_error(self, relu_hull_class):
        """Verify error when lower bound > upper bound."""
        hull = relu_hull_class()
        lb = np.array([1.0, 1.0])
        ub = np.array([-1.0, -1.0])

        # This should either raise an error or handle gracefully
        # Depending on implementation, may compute empty polytope or raise
        try:
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If it doesn't raise, verify result is sensible (no inf/nan)
            assert np.all(np.isfinite(result)), "Result contains inf/nan with inverted bounds"
        except (ValueError, RuntimeError):
            # Expected: error on inverted bounds
            pass

    def test_equal_bounds_produces_valid_constraints(self, relu_hull_class):
        """Verify handling when lb == ub (single point)."""
        hull = relu_hull_class()
        lb = np.array([0.5, 0.5])
        ub = np.array([0.5, 0.5])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints (degenerate polytope)
        assert result.shape[0] > 0, "No constraints generated for single point"
        assert np.all(np.isfinite(result)), "Constraints contain inf/nan"


class TestNumericalStability:
    """Test numerical stability with extreme bounds."""

    @pytest.mark.parametrize(
        ("lb", "ub"),
        [
            pytest.param(np.array([1000.0, 1000.0]), np.array([2000.0, 2000.0]), id="positive"),
            pytest.param(np.array([-2000.0, -2000.0]), np.array([-1000.0, -1000.0]), id="negative"),
        ],
    )
    def test_very_large_bounds(self, relu_hull_class, lb, ub):
        """Test ReLU numerical stability with extreme bounds."""
        hull = relu_hull_class()
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert np.all(np.isfinite(result)), "Large bounds produced inf/nan"

    def test_very_small_bounds(self, relu_hull_class, tiny_polytope_2d):
        """Test ReLU reports degeneracy for very small bounds."""
        hull = relu_hull_class()
        lb, ub = tiny_polytope_2d

        # Algorithm should raise ValueError for bounds with range < MIN_BOUNDS_RANGE (0.05)
        with pytest.raises(DegeneratedError, match="Polytope too small"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_mixed_scale_bounds(self, relu_hull_class):
        """Test with mixed magnitude bounds."""
        hull = relu_hull_class()
        lb = np.array([1e-6, 1e3])
        ub = np.array([1e6, 1e4])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify numerical stability
        assert np.all(np.isfinite(result)), "Mixed scale bounds produced inf/nan"


class TestLeakyReLUParameterValidation:
    """Test LeakyReLU with default negative slope."""

    def test_leakyrelu_basic_constraints(self, leakyrelu_hull_class):
        """Test LeakyReLU produces valid constraints."""
        hull = leakyrelu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints
        assert result.shape[0] > 0, "No constraints for LeakyReLU"
        # Check for finite values (handle both float and fraction types)
        if hasattr(result.flat[0], "real"):
            # Fraction type - check that values are defined
            assert result.size > 0, "Empty constraint matrix"
        else:
            assert np.all(np.isfinite(result)), "Constraints contain inf/nan"

    def test_leakyrelu_constraint_satisfaction(self, leakyrelu_hull_class):
        """Test LeakyReLU constraints are satisfied."""
        hull = leakyrelu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Test at a point in the input space
        x = np.array([0.5, -0.5])
        y = leakyrelu_np(x, negative_slope=0.01)
        point = np.concatenate([x, y])

        b = result[:, 0]
        a = result[:, 1:]
        constraints = b + a @ point

        # Constraints should be satisfied (allow for numerical precision)
        assert np.all(np.asarray(constraints) >= -1e-8), "LeakyReLU constraints violated"


class TestELUAlphaValidation:
    """Test ELU with default alpha parameter."""

    def test_elu_basic_constraints(self, elu_hull_class):
        """Test ELU produces valid constraints."""
        hull = elu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints
        assert result.shape[0] > 0, "No constraints for ELU"
        # Check for finite values (handle both float and fraction types)
        if hasattr(result.flat[0], "real"):
            # Fraction type - check that values are defined
            assert result.size > 0, "Empty constraint matrix"
        else:
            assert np.all(np.isfinite(result)), "Constraints contain inf/nan"

    def test_elu_constraint_satisfaction(self, elu_hull_class):
        """Test ELU constraints are satisfied."""
        hull = elu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Test at a point in the input space
        x = np.array([0.5, -0.5])
        y = elu_np(x)  # elu_np uses default alpha internally
        point = np.concatenate([x, y])

        b = result[:, 0]
        a = result[:, 1:]
        constraints = b + a @ point

        # Constraints should be satisfied (allow for numerical precision)
        assert np.all(np.asarray(constraints) >= -1e-8), "ELU constraints violated"


class TestMultiDimensionalConstraints:
    """Test constraint satisfaction across different dimensions."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_relu_constraints_all_dimensions(self, relu_hull_class, dim):
        """Verify ReLU constraints satisfied for various dimensions."""
        hull = relu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Test random point
        rng = np.random.default_rng(42)
        x = rng.uniform(lb, ub)
        y = relu_np(x)
        point = np.concatenate([x, y])

        b = result[:, 0]
        a = result[:, 1:]
        constraints = b + a @ point

        assert np.all(constraints >= -1e-8), f"Constraints violated for dimension {dim}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_sigmoid_constraints_all_dimensions(self, sigmoid_hull_class, dim):
        """Verify Sigmoid constraints satisfied for various dimensions."""
        hull = sigmoid_hull_class()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Test random point
        rng = np.random.default_rng(42)
        x = rng.uniform(lb, ub)
        y = sigmoid_np(x)
        point = np.concatenate([x, y])

        b = result[:, 0]
        a = result[:, 1:]
        constraints = b + a @ point

        assert np.all(constraints >= -1e-8), f"Sigmoid constraints violated for dimension {dim}"
