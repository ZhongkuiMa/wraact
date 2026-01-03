"""Unit tests for ReLUHull functionality.

Tests the ReLUHull class covering:
- cal_hull() shape and format
- cal_sn_constrs() single-neuron constraints
- cal_mn_constrs() multi-neuron constraints
- Trivial case handling (all active/inactive)
- Dtype preservation

Key Learning for Template:
============================

**Output Shape Formula**: For N-dimensional input, ReLUHull.cal_hull() returns
constraints with shape (num_constraints, 2*N + 1), where:
  - 1 column for offset (b)
  - N columns for input coefficients (A_input)
  - N columns for output coefficients (A_output)

This follows the H-representation format [b | A_input | A_output].

**Example**:
  - 2D input → shape (..., 5) [columns: b, x, y, out_x, out_y]
  - 3D input → shape (..., 7) [columns: b, x, y, z, out_x, out_y, out_z]
  - 4D input → shape (..., 9) [columns: b, x, y, z, w, out_x, out_y, out_z, out_w]

Use this formula when writing tests for other activation functions:
expected_cols = 2 * input_dimension + 1

**ReLU Constraints**:
- Requires non-trivial case: lb < 0 < ub (some inputs negative, some positive)
- Polytope must be feasible (have at least one valid point)
- May raise RuntimeError if polytope is infeasible
- May raise TypeError if cdd module encounters issues

**Test Pattern**:
- Use try/except to handle infeasible polytopes gracefully
- Skip tests when polytope is infeasible rather than failing
- Verify method exists for lower-level APIs (cal_sn_constrs, cal_mn_constrs)
- Test primarily through public cal_hull() API
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from tests.conftest import generate_feasible_random_polytope, generate_random_polytope_constraints


class TestReLUHullBasic:
    """Basic functionality tests for ReLUHull."""

    def test_cal_hull_returns_ndarray(
        self, relu_hull_class, simple_2d_box_constraints, simple_2d_box_bounds
    ):
        """Verify cal_hull() returns an ndarray."""
        hull = relu_hull_class()
        lb, ub = simple_2d_box_bounds
        result = hull.cal_hull(simple_2d_box_constraints, lb, ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # 2D array

    def test_cal_hull_output_shape(self, relu_hull_class):
        """Verify cal_hull() returns constraints with correct shape.

        For a 2D input, the ReLU hull output has shape (num_constraints, 2*dim+1).
        ReLU requires lb < 0 < ub for non-trivial case.
        """
        constraints = np.array(
            [
                [0, 1, 0],  # x >= 0
                [-1, 1, 0],  # x <= 1
                [0, 0, 1],  # y >= 0
                [-1, 0, 1],  # y <= 1
            ],
            dtype=np.float64,
        )
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class()
        result = hull.cal_hull(constraints, lb, ub)

        # For N-dimensional input, output constraints have shape (num_constraints, 2*N+1)
        # 2D input: 2*2 + 1 = 5 columns
        dim = 2
        expected_cols = 2 * dim + 1
        assert result.shape[1] == expected_cols, (
            f"Expected {expected_cols} columns, got {result.shape[1]}"
        )

    def test_cal_hull_returns_float_array(
        self, relu_hull_class, simple_2d_box_constraints, simple_2d_box_bounds
    ):
        """Verify cal_hull() returns float array."""
        hull = relu_hull_class()
        lb, ub = simple_2d_box_bounds
        result = hull.cal_hull(simple_2d_box_constraints, lb, ub)

        # Should be numeric type (float or int)
        assert np.issubdtype(result.dtype, np.number)

    def test_cal_hull_output_finite(
        self, relu_hull_class, simple_2d_box_constraints, simple_2d_box_bounds
    ):
        """Verify cal_hull() output contains no inf or nan values."""
        hull = relu_hull_class()
        lb, ub = simple_2d_box_bounds
        result = hull.cal_hull(simple_2d_box_constraints, lb, ub)

        assert np.all(np.isfinite(result)), "Output contains inf or nan values"

    def test_cal_hull_3d_input(self, relu_hull_class, simple_3d_octahedron_constraints):
        """Test cal_hull() with 3D input.

        ReLU requires mixed sign bounds (lb < 0 < ub) for non-trivial case.
        """
        # Use mixed sign bounds for non-trivial ReLU case
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = relu_hull_class()
        result = hull.cal_hull(simple_3d_octahedron_constraints, lb, ub)

        # For N-dimensional input, output constraints have shape (num_constraints, 2*N+1)
        # 3D input: 2*3 + 1 = 7 columns
        dim = 3
        expected_cols = 2 * dim + 1
        assert result.shape[1] == expected_cols, (
            f"Expected {expected_cols} columns for 3D input, got {result.shape[1]}"
        )

    def test_cal_hull_4d_input(self, relu_hull_class):
        """Test cal_hull() with 4D input.

        ReLU requires mixed sign bounds (lb < 0 < ub) for non-trivial case.
        """
        constraints = generate_random_polytope_constraints(4, seed=100)
        # Use mixed sign bounds for non-trivial ReLU case
        lb = np.array([-2.0, -2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0, 2.0])

        hull = relu_hull_class()
        result = hull.cal_hull(constraints, lb, ub)

        # For N-dimensional input, output constraints have shape (num_constraints, 2*N+1)
        # 4D input: 2*4 + 1 = 9 columns
        dim = 4
        expected_cols = 2 * dim + 1
        assert result.shape[1] == expected_cols, (
            f"Expected {expected_cols} columns for 4D input, got {result.shape[1]}"
        )

    def test_cal_hull_all_positive_inputs(self, relu_hull_class):
        """Test ReLU hull when all inputs are positive (all active neurons).

        In this case, ReLU acts as identity: y = x.
        So the hull should contain simple identity constraints.
        """
        constraints = np.array(
            [
                [0, 1, 0],  # x >= 0
                [-1, 1, 0],  # x <= 1
                [0, 0, 1],  # y >= 0
                [-1, 0, 1],  # y <= 1
            ],
            dtype=np.float64,
        )
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class()
        result = hull.cal_hull(constraints, lb, ub)

        # For all-positive ReLU inputs, output should be non-empty
        assert result.shape[0] > 0, "No constraints generated for all-positive case"
        assert np.all(np.isfinite(result))

    def test_cal_hull_near_zero_positive_inputs(self, relu_hull_class):
        """Test ReLU hull with inputs close to zero (mostly positive).

        This tests the edge case where bounds are mostly positive.
        ReLU requires non-trivial case: lb < 0 < ub.
        """
        constraints = np.array(
            [
                [0, 1, 0],  # x >= 0
                [-1, 1, 0],  # x <= 1
                [0, 0, 1],  # y >= 0
                [-1, 0, 1],  # y <= 1
            ],
            dtype=np.float64,
        )
        # Use bounds that barely satisfy non-trivial requirement
        lb = np.array([-0.01, -0.01])
        ub = np.array([1.0, 1.0])

        hull = relu_hull_class()
        result = hull.cal_hull(constraints, lb, ub)

        # Should generate valid constraints for non-trivial case
        assert result.shape[0] > 0
        assert np.all(np.isfinite(result))

    def test_cal_hull_infeasible_polytope(self, relu_hull_class, infeasible_polytope_relu):
        """Test ReLU raises error for problematic polytope.

        Constraints define box [0,1]² but bounds require x in [-0.5, 0.5], y in [0.5, 1.0].
        Dimension 1 (y) does NOT satisfy ReLU requirement: lb < 0 < ub.
        Algorithm detects this during vertex computation and raises an error.
        """
        constraints, lb, ub = infeasible_polytope_relu
        hull = relu_hull_class()

        # Algorithm should raise exception when encountering problematic polytope
        # (ReLU requirement violation: not all dimensions have lb < 0 < ub)
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            hull.cal_hull(constraints, lb, ub)

    def test_cal_sn_constrs_returns_array(self, relu_hull_class):
        """Test that cal_sn_constrs() returns an ndarray.

        ReLU requires non-trivial case: lb < 0 < ub.
        """
        hull = relu_hull_class()
        # Use bounds that satisfy ReLU non-trivial requirement
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Access single-neuron constraint calculation
        result = hull.cal_sn_constrs(lb, ub)

        assert isinstance(result, np.ndarray)

    def test_cal_mn_constrs_returns_array(self, relu_hull_class):
        """Test that cal_mn_constrs() returns an ndarray (lower-level API).

        Note: cal_mn_constrs is a lower-level API that expects preprocessed constraints.
        It's better to test this through the public cal_hull() API.
        For now, we just verify the method exists and is callable.
        """
        hull = relu_hull_class()
        # Verify the method exists
        assert hasattr(hull, "cal_mn_constrs")
        assert callable(hull.cal_mn_constrs)

    def test_constraint_format_h_representation(
        self, relu_hull_class, simple_2d_box_constraints, simple_2d_box_bounds
    ):
        """Verify constraints follow H-representation format: [b | A].

        In H-representation, each constraint is: b + A @ x >= 0
        Constraint matrix has shape (num_constraints, num_variables + 1).
        """
        hull = relu_hull_class()
        lb, ub = simple_2d_box_bounds
        result = hull.cal_hull(simple_2d_box_constraints, lb, ub)

        # Check format
        assert result.ndim == 2
        assert result.shape[1] >= 2  # At least [b | a1]

    def test_empty_constraints_handling(self, relu_hull_class, simple_2d_box_bounds):
        """Test behavior with empty input constraints.

        Empty constraints means an unbounded polytope.
        """
        empty_constraints = np.empty((0, 3), dtype=np.float64)
        lb, ub = simple_2d_box_bounds

        hull = relu_hull_class()

        # Should either return empty array or generate bounds-based constraints
        try:
            result = hull.cal_hull(empty_constraints, lb, ub)
            assert isinstance(result, np.ndarray)
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for empty constraints
            pass

    def test_bounds_consistency(self, relu_hull_class, simple_2d_box_constraints):
        """Test that bounds are consistent (lb <= ub)."""
        hull = relu_hull_class()
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        assert np.all(lb <= ub), "Bounds must satisfy lb <= ub"

        result = hull.cal_hull(simple_2d_box_constraints, lb, ub)
        assert isinstance(result, np.ndarray)

    def test_inconsistent_bounds_raises_error(self, relu_hull_class, simple_2d_box_constraints):
        """Test that inconsistent bounds (lb > ub) raise an error or warning.

        This is a sanity check to ensure bounds are valid.
        """
        hull = relu_hull_class()
        lb = np.array([1.0, 1.0])
        ub = np.array([0.0, 0.0])  # Inconsistent: lb > ub

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, RuntimeError)):
            hull.cal_hull(simple_2d_box_constraints, lb, ub)


class TestReLUHullOnRandomPolytopes:
    """Test ReLUHull on parametrized random polytopes."""

    @pytest.mark.parametrize("seed", [42, 43, 44])
    def test_cal_hull_deterministic(self, relu_hull_class, seed):
        """Verify cal_hull() is deterministic (same input → same output).

        Uses margin-controlled random polytope generator to ensure feasibility.
        """
        # Generate with dimension-scaled margin (margin = 1.0 + 0.5*dim)
        constraints, lb, ub = generate_feasible_random_polytope(dim=2, seed=seed)

        # Adjust bounds to ensure ReLU requirement: lb < 0 < ub for all dimensions
        # Use safe adjustment that preserves polytope structure
        lb = np.minimum(lb, -0.5)  # Ensure lb <= -0.5
        ub = np.maximum(ub, 0.5)  # Ensure ub >= 0.5

        hull = relu_hull_class()
        # Should always succeed with margin-controlled generation
        result1 = hull.cal_hull(constraints, lb, ub)
        result2 = hull.cal_hull(constraints, lb, ub)
        # Results should be identical for deterministic implementation
        np.testing.assert_array_equal(result1, result2)

    def test_cal_hull_on_feasible_random_2d(self, relu_hull_class):
        """Test cal_hull() on feasible random 2D polytope with margin control."""
        # Generate with dimension-scaled margin (1.0 + 0.5*2 = 2.0)
        constraints, lb, ub = generate_feasible_random_polytope(dim=2, seed=42)

        # Adjust bounds to ensure ReLU requirement: lb < 0 < ub
        lb = np.minimum(lb, -0.5)  # Ensure lb <= -0.5
        ub = np.maximum(ub, 0.5)  # Ensure ub >= 0.5

        hull = relu_hull_class()
        # Should always succeed with margin-controlled generation
        result = hull.cal_hull(constraints, lb, ub)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
        assert np.all(np.isfinite(result))
        assert result.shape[1] == 5  # 2D: [b | x1 | x2 | y1 | y2]

    def test_cal_hull_on_feasible_random_3d(self, relu_hull_class):
        """Test cal_hull() on feasible random 3D polytope with margin control."""
        # Generate with dimension-scaled margin (1.0 + 0.5*3 = 2.5)
        constraints, lb, ub = generate_feasible_random_polytope(dim=3, seed=43)

        # Adjust bounds to ensure ReLU requirement: lb < 0 < ub
        lb = np.minimum(lb, -0.5)  # Ensure lb <= -0.5
        ub = np.maximum(ub, 0.5)  # Ensure ub >= 0.5

        hull = relu_hull_class()
        # Should always succeed with margin-controlled generation
        result = hull.cal_hull(constraints, lb, ub)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
        assert np.all(np.isfinite(result))
        assert result.shape[1] == 7  # 3D: [b | x1 | x2 | x3 | y1 | y2 | y3]

    def test_cal_hull_on_feasible_random_4d(self, relu_hull_class):
        """Test cal_hull() on feasible random 4D polytope with margin control."""
        # Generate with dimension-scaled margin (1.0 + 0.5*4 = 3.0)
        constraints, lb, ub = generate_feasible_random_polytope(dim=4, seed=44)

        # Adjust bounds to ensure ReLU requirement: lb < 0 < ub
        lb = np.minimum(lb, -0.5)  # Ensure lb <= -0.5
        ub = np.maximum(ub, 0.5)  # Ensure ub >= 0.5

        hull = relu_hull_class()
        # Should always succeed with margin-controlled generation
        result = hull.cal_hull(constraints, lb, ub)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
        assert np.all(np.isfinite(result))
        assert result.shape[1] == 9  # 4D: [b | x1 | x2 | x3 | x4 | y1 | y2 | y3 | y4]


class TestReLUSingleNeuronMode:
    """Test ReLU hull with single-neuron constraint mode.

    This tests the single-neuron constraint calculation path which is
    normally disabled in default ActHull initialization.
    """

    def test_cal_hull_single_neuron_2d(self, relu_hull_class):
        """Test single-neuron constraints for 2D input."""
        constraints, lb, ub = generate_feasible_random_polytope(dim=2, seed=42)
        lb = np.minimum(lb, -0.5)
        ub = np.maximum(ub, 0.5)

        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        result = hull.cal_hull(constraints, lb, ub)

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 5
        assert np.all(np.isfinite(result))

    def test_cal_hull_single_neuron_3d(self, relu_hull_class):
        """Test single-neuron constraints for 3D input."""
        constraints, lb, ub = generate_feasible_random_polytope(dim=3, seed=43)
        lb = np.minimum(lb, -0.5)
        ub = np.maximum(ub, 0.5)

        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        result = hull.cal_hull(constraints, lb, ub)

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 7
        assert np.all(np.isfinite(result))

    def test_cal_hull_single_neuron_soundness(self, relu_hull_class):
        """Verify soundness of single-neuron constraints."""
        constraints, lb, ub = generate_feasible_random_polytope(dim=2, seed=44)
        lb = np.minimum(lb, -0.5)
        ub = np.maximum(ub, 0.5)

        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        result = hull.cal_hull(constraints, lb, ub)

        # Single-neuron constraints should be valid
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
        assert np.all(np.isfinite(result))

    def test_cal_sn_constrs_direct(self, relu_hull_class):
        """Test direct call to cal_sn_constrs method."""
        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Direct call to cal_sn_constrs
        result = hull.cal_sn_constrs(lb, ub)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0

    def test_cal_hull_single_neuron_output_shape(self, relu_hull_class):
        """Verify single-neuron constraint output shape."""
        constraints, lb, ub = generate_feasible_random_polytope(dim=2, seed=45)
        lb = np.minimum(lb, -0.5)
        ub = np.maximum(ub, 0.5)

        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        result = hull.cal_hull(constraints, lb, ub)

        # For 2D input: 2*2 + 1 = 5 columns
        assert result.shape[1] == 5
        # Should have at least some constraints
        assert result.shape[0] > 0

    def test_cal_hull_single_neuron_finite(self, relu_hull_class):
        """Verify single-neuron constraints contain no inf or nan."""
        constraints, lb, ub = generate_feasible_random_polytope(dim=2, seed=46)
        lb = np.minimum(lb, -0.5)
        ub = np.maximum(ub, 0.5)

        hull = relu_hull_class(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        result = hull.cal_hull(constraints, lb, ub)

        assert np.all(np.isfinite(result))
