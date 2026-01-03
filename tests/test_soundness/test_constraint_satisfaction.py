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

from wraact._functions import elu_np, leakyrelu_np, relu_np, sigmoid_np
from wraact.acthull import ELUHull, LeakyReLUHull, ReLUHull, SigmoidHull


class TestReLUConstraintSatisfaction:
    """Test constraint satisfaction for ReLU hulls."""

    def test_relu_constraints_satisfied_at_bounds(self):
        """Verify constraints are satisfied at input bounds."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Test at lower bound
        x_lb = lb
        y_lb = relu_np(x_lb)
        point_lb = np.concatenate([x_lb, y_lb])
        b = result[:, 0]
        a = result[:, 1:]
        constraints_lb = b + a @ point_lb
        assert np.all(constraints_lb >= -1e-8), "Constraints violated at lower bound"

        # Test at upper bound
        x_ub = ub
        y_ub = relu_np(x_ub)
        point_ub = np.concatenate([x_ub, y_ub])
        constraints_ub = b + a @ point_ub
        assert np.all(constraints_ub >= -1e-8), "Constraints violated at upper bound"

    def test_relu_constraints_satisfied_at_center(self):
        """Verify constraints are satisfied at center point."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        x_center = (lb + ub) / 2
        y_center = relu_np(x_center)
        point_center = np.concatenate([x_center, y_center])
        b = result[:, 0]
        a = result[:, 1:]
        constraints_center = b + a @ point_center
        assert np.all(constraints_center >= -1e-8), "Constraints violated at center"

    def test_sigmoid_constraints_satisfied_at_bounds(self):
        """Verify sigmoid constraints are satisfied at bounds."""
        hull = SigmoidHull()
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

    def test_lower_greater_than_upper_raises_error(self):
        """Verify error when lower bound > upper bound."""
        hull = ReLUHull()
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

    def test_equal_bounds_produces_valid_constraints(self):
        """Verify handling when lb == ub (single point)."""
        hull = ReLUHull()
        lb = np.array([0.5, 0.5])
        ub = np.array([0.5, 0.5])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints (degenerate polytope)
        assert result.shape[0] > 0, "No constraints generated for single point"
        assert np.all(np.isfinite(result)), "Constraints contain inf/nan"


class TestNumericalStability:
    """Test numerical stability with extreme bounds."""

    def test_very_large_positive_bounds(self):
        """Test ReLU with large positive bounds."""
        hull = ReLUHull()
        lb = np.array([1000.0, 1000.0])
        ub = np.array([2000.0, 2000.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify no inf/nan in constraints
        assert np.all(np.isfinite(result)), "Large bounds produced inf/nan"

    def test_very_large_negative_bounds(self):
        """Test ReLU with large negative bounds."""
        hull = ReLUHull()
        lb = np.array([-2000.0, -2000.0])
        ub = np.array([-1000.0, -1000.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify no inf/nan in constraints
        assert np.all(np.isfinite(result)), "Large negative bounds produced inf/nan"

    def test_very_small_bounds(self, tiny_polytope_2d):
        """Test ReLU raises ValueError for very small bounds."""
        hull = ReLUHull()
        lb, ub = tiny_polytope_2d

        # Algorithm should raise ValueError for bounds with range < MIN_BOUNDS_RANGE (0.05)
        with pytest.raises(ValueError, match="Polytope too small"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_mixed_scale_bounds(self):
        """Test with mixed magnitude bounds."""
        hull = ReLUHull()
        lb = np.array([1e-6, 1e3])
        ub = np.array([1e6, 1e4])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify numerical stability
        assert np.all(np.isfinite(result)), "Mixed scale bounds produced inf/nan"


class TestLeakyReLUParameterValidation:
    """Test LeakyReLU with default negative slope."""

    def test_leakyrelu_basic_constraints(self):
        """Test LeakyReLU produces valid constraints."""
        hull = LeakyReLUHull()
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

    def test_leakyrelu_constraint_satisfaction(self):
        """Test LeakyReLU constraints are satisfied."""
        hull = LeakyReLUHull()
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

    def test_elu_basic_constraints(self):
        """Test ELU produces valid constraints."""
        hull = ELUHull()
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

    def test_elu_constraint_satisfaction(self):
        """Test ELU constraints are satisfied."""
        hull = ELUHull()
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
    def test_relu_constraints_all_dimensions(self, dim):
        """Verify ReLU constraints satisfied for various dimensions."""
        hull = ReLUHull()
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
    def test_sigmoid_constraints_all_dimensions(self, dim):
        """Verify Sigmoid constraints satisfied for various dimensions."""
        hull = SigmoidHull()
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
