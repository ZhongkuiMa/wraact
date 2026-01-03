"""Tests for edge cases and degenerate polytopes.

This module tests hull computation on boundary cases, degenerate polytopes,
and extreme input scenarios.

Key Tests:
==========
- Single point polytopes (lb == ub)
- Very large/small bounds
- All-active and all-inactive neuron cases
- Numerical precision limits
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact.acthull import ELUHull, ReLUHull, SigmoidHull, TanhHull


class TestSinglePointPolytope:
    """Test behavior with degenerate single-point polytopes."""

    def test_relu_single_point_positive(self):
        """Test ReLU on single positive point."""
        hull = ReLUHull()
        lb = np.array([1.0, 1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints
        assert result.shape[0] > 0, "No constraints for single point"
        assert np.all(np.isfinite(result)), "Constraints contain inf/nan"

    def test_relu_single_point_negative(self):
        """Test ReLU on single negative point."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([-1.0, -1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints
        assert result.shape[0] > 0, "No constraints for single point"
        assert np.all(np.isfinite(result)), "Constraints contain inf/nan"

    def test_relu_single_point_zero(self):
        """Test ReLU on single point at origin."""
        hull = ReLUHull()
        lb = np.array([0.0, 0.0])
        ub = np.array([0.0, 0.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints
        assert result.shape[0] > 0, "No constraints for single point"
        assert np.all(np.isfinite(result)), "Constraints contain inf/nan"

    def test_sigmoid_single_point(self):
        """Test Sigmoid on single point."""
        hull = SigmoidHull()
        lb = np.array([0.0, 0.0])
        ub = np.array([0.0, 0.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert result.shape[0] > 0, "No constraints for single point"
        assert np.all(np.isfinite(result)), "Constraints contain inf/nan"


class TestAllActiveNeurons:
    """Test ReLU when all neurons are guaranteed active."""

    def test_relu_all_positive_inputs(self):
        """Test ReLU when all inputs guaranteed positive."""
        hull = ReLUHull()
        lb = np.array([0.1, 0.1])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For all-positive inputs, ReLU should be identity: y = x
        # Constraints should reflect this
        assert result.shape[0] > 0, "No constraints"
        assert np.all(np.isfinite(result)), "Constraints contain inf/nan"

        # Verify identity relationship: testing at bounds
        x = lb
        y = x  # ReLU with positive input = identity
        point = np.concatenate([x, y])
        b = result[:, 0]
        a = result[:, 1:]
        constraints = b + a @ point
        assert np.all(constraints >= -1e-8), "Constraints violated for identity case"


class TestAllInactiveNeurons:
    """Test ReLU when all neurons are guaranteed inactive."""

    def test_relu_all_negative_inputs(self):
        """Test ReLU when all inputs guaranteed negative."""
        hull = ReLUHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([-0.1, -0.1])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For all-negative inputs, ReLU should be zero: y = 0
        # Constraints should reflect this
        assert result.shape[0] > 0, "No constraints"
        assert np.all(np.isfinite(result)), "Constraints contain inf/nan"

        # Verify zero relationship: testing at bounds
        x = lb
        y = np.zeros_like(x)  # ReLU with negative input = 0
        point = np.concatenate([x, y])
        b = result[:, 0]
        a = result[:, 1:]
        constraints = b + a @ point
        assert np.all(constraints >= -1e-8), "Constraints violated for zero case"


class TestVeryLargeBounds:
    """Test with extremely large input bounds."""

    def test_relu_very_large_positive(self):
        """Test ReLU with very large positive bounds."""
        hull = ReLUHull()
        lb = np.array([1e6, 1e6])
        ub = np.array([1e7, 1e7])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result)), "Large bounds produced inf/nan"
        assert result.shape[0] > 0, "No constraints"

    def test_relu_very_large_negative(self):
        """Test ReLU with very large negative bounds."""
        hull = ReLUHull()
        lb = np.array([-1e7, -1e7])
        ub = np.array([-1e6, -1e6])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result)), "Large negative bounds produced inf/nan"
        assert result.shape[0] > 0, "No constraints"

    def test_sigmoid_large_bounds(self):
        """Test Sigmoid with large bounds."""
        hull = SigmoidHull()
        lb = np.array([-100.0, -100.0])
        ub = np.array([100.0, 100.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Sigmoid should be numerically stable even with extreme inputs
        assert np.all(np.isfinite(result)), "Large bounds in sigmoid produced inf/nan"
        assert result.shape[0] > 0, "No constraints"


class TestVerySmallBounds:
    """Test with very small input ranges."""

    def test_relu_tiny_range(self, tiny_polytope_2d):
        """Test ReLU raises ValueError for tiny polytope (range < 0.05)."""
        hull = ReLUHull()
        lb, ub = tiny_polytope_2d

        # Algorithm should raise ValueError for polytopes with range < MIN_BOUNDS_RANGE
        with pytest.raises(ValueError, match=r"Polytope too small.*range.*< threshold"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_sigmoid_tiny_range(self, tiny_polytope_2d):
        """Test Sigmoid raises ValueError for tiny polytope."""
        hull = SigmoidHull()
        lb, ub = tiny_polytope_2d

        with pytest.raises(ValueError, match="Polytope too small"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)


class TestMixedScaleBounds:
    """Test with bounds on very different scales."""

    def test_relu_mixed_scale(self, extreme_scale_polytope_2d):
        """Test ReLU raises ValueError for extreme scale difference (500,000x).

        Dimension 0: range = 0.002 (very small)
        Dimension 1: range = 999 (very large)
        Minimum dimension triggers ValueError.
        """
        hull = ReLUHull()
        lb, ub = extreme_scale_polytope_2d

        # Minimum dimension (dim 0) has range 0.002 < 0.05
        with pytest.raises(ValueError, match="Polytope too small"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_sigmoid_mixed_scale(self):
        """Test Sigmoid raises ValueError for mixed scale (2000x difference).

        Dimension 0: range = 0.002
        Dimension 1: range = 4.0
        Minimum dimension triggers ValueError.
        """
        hull = SigmoidHull()
        lb = np.array([-1e-3, -2.0], dtype=np.float64)
        ub = np.array([1e-3, 2.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Polytope too small"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)


class TestDegeneratePolytopes:
    """Test graceful handling of degenerate polytopes."""

    def test_collapsed_dimension(self, collapsed_dimension_polytope):
        """Test ReLU handles collapsed dimension (lb[0] == ub[0]).

        When one dimension has zero width (lb == ub), the polytope is degenerate.
        Algorithm should either:
        1. Detect degeneracy and handle gracefully (return valid constraints), or
        2. Raise DegeneratedError after retry with fraction arithmetic
        """
        hull = ReLUHull()
        lb, ub = collapsed_dimension_polytope

        # Algorithm should handle degeneracy gracefully
        # Either: return valid result OR raise ValueError/DegeneratedError
        try:
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If succeeds, verify result is valid
            assert isinstance(result, np.ndarray), "Result should be ndarray"
            assert result.shape[0] > 0, "Should have at least one constraint"
            assert np.all(np.isfinite(result)), "All values should be finite"
        except (ValueError, RuntimeError):
            # Acceptable: degeneracy or infeasibility detected
            pass

    def test_line_segment_polytope(self, line_segment_polytope):
        """Test ReLU with near-degenerate polytope (line segment).

        This polytope is nearly degenerate: one dimension has zero width.
        Algorithm may either:
        1. Raise DegeneratedError if insufficient vertices detected, or
        2. Return valid constraints if dimension can be handled
        """
        from wraact._exceptions import DegeneratedError

        hull = ReLUHull()
        constraints, lb, ub = line_segment_polytope

        # Algorithm should handle this gracefully - either raise error or return result
        try:
            result = hull.cal_hull(constraints, lb, ub)
            # If succeeds, verify result is valid
            assert isinstance(result, np.ndarray), "Result should be ndarray"
            assert result.shape[0] > 0, "Should have at least one constraint"
            assert np.all(np.isfinite(result)), "All values should be finite"
        except DegeneratedError:
            # Acceptable: polytope detected as degenerate
            pass

    def test_all_dimensions_positive(self):
        """Test ReLU with all-positive inputs (no mixed signs).

        ReLU requires lb < 0 < ub for non-trivial case.
        All-positive violates this requirement.
        Algorithm should still compute result (identity mapping in positive region).
        """
        hull = ReLUHull()
        lb = np.array([0.1, 0.1], dtype=np.float64)
        ub = np.array([2.0, 2.0], dtype=np.float64)

        # Should succeed: ReLU(x) = x for all x > 0
        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray), "Result should be ndarray"
        assert result.shape[0] > 0, "Should have at least one constraint"
        assert np.all(np.isfinite(result)), "All values should be finite"


class TestDimensionality:
    """Test behavior across different dimensions."""

    @pytest.mark.parametrize("dim", [1, 2, 3, 4])
    def test_relu_various_dimensions(self, dim):
        """Test ReLU hull computation for various dimensions."""
        hull = ReLUHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify basic properties
        assert result.shape[0] > 0, f"No constraints for dimension {dim}"
        assert np.all(np.isfinite(result)), f"Constraints contain inf/nan for dim {dim}"
        assert result.shape[1] == dim + dim + 1, f"Wrong shape for dimension {dim}"

    @pytest.mark.parametrize("dim", [1, 2, 3, 4])
    def test_sigmoid_various_dimensions(self, dim):
        """Test Sigmoid hull computation for various dimensions."""
        hull = SigmoidHull()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify basic properties
        assert result.shape[0] > 0, f"No constraints for dimension {dim}"
        assert np.all(np.isfinite(result)), f"Constraints contain inf/nan for dim {dim}"


class TestSymmetry:
    """Test symmetric input bounds."""

    def test_relu_symmetric_bounds(self):
        """Test ReLU on symmetric bounds [-a, a]."""
        hull = ReLUHull()
        for a in [0.1, 1.0, 10.0]:
            lb = np.array([-a, -a])
            ub = np.array([a, a])

            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert np.all(np.isfinite(result)), f"Symmetric {a}: inf/nan produced"
            assert result.shape[0] > 0, f"Symmetric {a}: no constraints"

    def test_sigmoid_symmetric_bounds(self):
        """Test Sigmoid on symmetric bounds [-a, a]."""
        hull = SigmoidHull()
        for a in [0.5, 2.0, 5.0]:
            lb = np.array([-a, -a])
            ub = np.array([a, a])

            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert np.all(np.isfinite(result)), f"Symmetric {a}: inf/nan produced"
            assert result.shape[0] > 0, f"Symmetric {a}: no constraints"

    def test_tanh_symmetric_bounds(self):
        """Test Tanh on symmetric bounds [-a, a]."""
        hull = TanhHull()
        for a in [0.5, 2.0, 5.0]:
            lb = np.array([-a, -a])
            ub = np.array([a, a])

            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert np.all(np.isfinite(result)), f"Symmetric {a}: inf/nan produced"
            assert result.shape[0] > 0, f"Symmetric {a}: no constraints"


class TestELUStability:
    """Test ELU stability across different input ranges."""

    def test_elu_standard_range(self):
        """Test ELU with standard input range."""
        hull = ELUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints
        if result is not None:
            assert result.shape[0] > 0, "No constraints"
            if hasattr(result.flat[0], "real"):
                assert result.size > 0, "Empty constraint matrix"
            else:
                assert np.all(np.isfinite(result)), "Standard range produced inf/nan"

    def test_elu_large_positive(self):
        """Test ELU with large positive bounds."""
        hull = ELUHull()
        lb = np.array([100.0, 100.0])
        ub = np.array([1000.0, 1000.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        if result is not None:
            assert result.shape[0] > 0, "No constraints"
            if not hasattr(result.flat[0], "real"):
                assert np.all(np.isfinite(result)), "Large positive produced inf/nan"
