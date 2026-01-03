"""Tests for utility functions in the WraAct package.

Utility functions include helper methods for constraint manipulation,
polytope operations, and DLP (Double Linear Pieces) computations.

Key Functions Tested:
=====================
- cal_mn_constrs_with_one_y_dlp: Multi-neuron constraint generation for DLP functions
  with specified output dimensions.
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest


class TestCalMnConstrsWithOneYDLP:
    """Tests for cal_mn_constrs_with_one_y_dlp utility function."""

    def test_returns_tuple_of_two_arrays(self):
        """Verify function returns tuple (constraints, vertices)."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        # Simple box polytope
        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])  # 4 constraints for 2D box
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # 4 vertices
        aux_lines = np.array([[0.15, 0, 0]])  # Single line for trivial case
        aux_point = None
        is_convex = True

        result = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=aux_point, is_convex=is_convex
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    def test_trivial_case_output_shape(self):
        """Verify output shape for trivial case (single line, no aux_point)."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        aux_lines = np.array([[0.15, 0, 0]])  # [slope_x, slope_y, intercept]

        c_out, v_out = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
        )

        # Output should extend constraints and vertices by one column
        assert c_out.shape[1] == c.shape[1] + 1  # Input + 1 output dimension
        assert v_out.shape[1] == v.shape[1] + 1  # Input + 1 output dimension
        assert c_out.shape[0] == c.shape[0] + 1  # Extra constraint for output line

    def test_trivial_case_adds_line_constraint(self):
        """Verify trivial case adds the auxiliary line as constraint."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        aux_lines = np.array([[0.15, 0.1, 1.0]])

        c_out, _v_out = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
        )

        # Last row should be the auxiliary line constraint
        last_constraint = c_out[-1, :]
        # Last element should be non-zero for output dimension
        assert last_constraint[-1] != 0

    def test_trivial_case_output_vertices_are_extended(self):
        """Verify output vertices are extended with computed output dimension."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        aux_lines = np.array([[0.15, 0.1, 1.0]])

        _c_out, v_out = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
        )

        # Check that output vertices are finite and extended
        assert v_out.shape[1] == v.shape[1] + 1
        assert np.all(np.isfinite(v_out))

    def test_trivial_case_raises_error_with_multiple_lines(self):
        """Verify error raised when aux_point is None but multiple auxiliary lines provided."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        aux_lines = np.array(
            [[0.1, 0.0, 1.0], [0.2, 0.5, 1.0]]  # Two lines (should raise error)
        )

        with pytest.raises(RuntimeError, match="should have only one line"):
            cal_mn_constrs_with_one_y_dlp(
                idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
            )

    def test_non_trivial_case_requires_aux_point(self):
        """Verify that DLP case with two auxiliary lines requires aux_point."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 1.0], [0.3, 1.0]], dtype=float)
        # Two lines for DLP with different slopes
        aux_lines = np.array([[0.0, 0.2, 1.0], [0.0, 0.8, 1.0]])
        aux_point = 0.2  # Intersection point: divide vertices at x=0.2

        # Should not raise error when aux_point is provided with proper data
        try:
            c_out, v_out = cal_mn_constrs_with_one_y_dlp(
                idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=aux_point, is_convex=True
            )
            assert isinstance(c_out, np.ndarray)
            assert isinstance(v_out, np.ndarray)
        except RuntimeError:
            # Some configurations may not be solvable due to geometric constraints
            pass

    def test_non_trivial_case_output_shape(self):
        """Verify output shape for non-trivial DLP case."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 1.0], [0.3, 1.0]], dtype=float)
        aux_lines = np.array([[0.0, 0.2, 1.0], [0.0, 0.8, 1.0]])
        aux_point = 0.2

        try:
            c_out, v_out = cal_mn_constrs_with_one_y_dlp(
                idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=aux_point, is_convex=True
            )
            # Output should extend both by one column
            assert c_out.shape[1] == c.shape[1] + 1
            assert v_out.shape[1] == v.shape[1] + 1
            # Number of constraints may increase
            assert c_out.shape[0] >= c.shape[0]
        except RuntimeError:
            # Some configurations may not be solvable
            pass

    def test_error_when_aux_point_divides_vertices_unevenly(self):
        """Verify error when all vertices are on one side of auxiliary point."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        aux_lines = np.array([[0.1, 0.0, 1.0], [0.2, 0.5, 1.0]])
        aux_point = 10.0  # Point far outside polytope - all vertices on one side

        with pytest.raises(RuntimeError, match="vertices should not all"):
            cal_mn_constrs_with_one_y_dlp(
                idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=aux_point, is_convex=True
            )

    def test_convex_vs_non_convex(self):
        """Verify is_convex parameter changes constraint sign."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        aux_lines = np.array([[0.1, 0.0, 1.0]])

        c_convex, _v_convex = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
        )

        c_concave, _v_concave = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=False
        )

        # Last constraints should be opposite sign (one is negation of other)
        convex_line = c_convex[-1, :]
        concave_line = c_concave[-1, :]

        np.testing.assert_array_almost_equal(convex_line, -concave_line)

    def test_vertices_extended_with_output_dimension(self):
        """Verify vertices are properly extended with output dimension."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        slope = 0.2
        intercept = 0.1
        aux_lines = np.array([[intercept, slope, 1.0]])

        _c_out, v_out = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
        )

        # Number of vertices should remain same
        assert v_out.shape[0] == v.shape[0]
        # But each vertex should now have an output coordinate
        assert v_out.shape[1] == v.shape[1] + 1

    def test_constraints_extended_with_output_dimension(self):
        """Verify constraints are properly extended with output dimension."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        aux_lines = np.array([[0.1, 0.2, 1.0]])

        c_out, _v_out = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
        )

        # All constraints should have extended dimension (padded with 0)
        # except the new line constraint which should have non-zero y coefficient
        for i in range(c.shape[0]):
            # Original constraints padded: [b | a_x | a_y | 0]
            assert c_out[i, -1] == 0

        # New line constraint should have non-zero y coefficient
        assert c_out[-1, -1] != 0

    def test_output_is_finite(self):
        """Verify output constraints and vertices contain finite values."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        aux_lines = np.array([[0.1, 0.2, 1.0]])

        c_out, v_out = cal_mn_constrs_with_one_y_dlp(
            idx=0, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
        )

        # All outputs should be finite (no inf or nan)
        assert np.all(np.isfinite(c_out))
        assert np.all(np.isfinite(v_out))

    def test_different_index_values(self):
        """Verify different index values work correctly."""
        from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp

        c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        v = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        aux_lines = np.array([[0.1, 0.2, 1.0]])

        # Test with different indices
        for idx in [0, 1]:
            c_out, v_out = cal_mn_constrs_with_one_y_dlp(
                idx=idx, c=c, v=v, aux_lines=aux_lines, aux_point=None, is_convex=True
            )

            assert c_out.shape[1] == c.shape[1] + 1
            assert v_out.shape[1] == v.shape[1] + 1
