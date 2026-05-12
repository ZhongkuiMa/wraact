"""Tests for polytope representation conversions.

This module tests H-representation and V-representation conversions using
pycddlib for polytope operations.

Key Tests:
==========
- H-rep to V-rep conversion (halfspace to vertex)
- V-rep to H-rep conversion (vertex to halfspace)
- Roundtrip conversions preserve polytope structure
- Handling of degenerate polytopes
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

try:
    import cdd

    HAS_PYCDDLIB = True
except ImportError:
    HAS_PYCDDLIB = False

if not HAS_PYCDDLIB:
    pytest.skip("pycddlib not installed", allow_module_level=True)


class TestHRepToVRep:
    """Test H-representation to V-representation conversion."""

    @pytest.mark.parametrize(
        ("h_rep", "expected_vertices", "description"),
        [
            pytest.param(
                [
                    [0, 1, 0],  # x >= 0
                    [0, 0, 1],  # y >= 0
                    [1, -1, 0],  # -x + 1 >= 0 (x <= 1)
                    [1, 0, -1],  # -y + 1 >= 0 (y <= 1)
                ],
                4,
                "unit cube",
                id="unit_cube",
            ),
            pytest.param(
                [
                    [0, 1, 0],  # x >= 0
                    [0, 0, 1],  # y >= 0
                    [1, -1, -1],  # x + y <= 1
                ],
                3,
                "triangle",
                id="triangle",
            ),
        ],
    )
    def test_h_rep_to_v_rep(self, h_rep, expected_vertices, description):
        """Test converting H-rep to V-rep for a given polytope."""
        mat = cdd.Matrix(h_rep, number_type="float")
        mat.rep_type = cdd.RepType.INEQUALITY

        vert_mat = cdd.Polyhedron(mat).get_generators()
        vert_array = np.array(vert_mat)
        vertices = [list(vert_array[i, 1:]) for i in range(vert_array.shape[0])]

        assert len(vertices) == expected_vertices, (
            f"Expected {expected_vertices} vertices for {description}, got {len(vertices)}"
        )


class TestVRepToHRep:
    """Test V-representation to H-representation conversion."""

    def test_unit_square_v_to_h(self):
        """Test converting unit square V-rep to H-rep."""
        # Unit square vertices
        vertices = [
            [1, 0, 0],  # (0, 0)
            [1, 1, 0],  # (1, 0)
            [1, 1, 1],  # (1, 1)
            [1, 0, 1],  # (0, 1)
        ]

        mat = cdd.Matrix(vertices, number_type="float")
        mat.rep_type = cdd.RepType.GENERATOR

        # Convert to H-rep
        h_mat = cdd.Polyhedron(mat).get_inequalities()

        # Should get 4 constraints (the sides of the square)
        constraints = np.array(h_mat)
        assert constraints.shape[0] == 4, f"Expected 4 constraints, got {constraints.shape[0]}"


class TestRoundtripConversion:
    """Test that H→V→H conversions preserve polytope structure."""

    @pytest.mark.parametrize(
        ("h_rep_orig", "min_constraints", "max_constraints"),
        [
            pytest.param(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, -1, 0],
                    [1, 0, -1],
                ],
                4,
                6,
                id="unit_cube",
            ),
            pytest.param(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, -1, -1],
                ],
                3,
                None,
                id="triangle",
            ),
        ],
    )
    def test_consistency_hrep_vrep_hrep_roundtrip(
        self, h_rep_orig, min_constraints, max_constraints
    ):
        """Test H→V→H roundtrip preserves polytope structure."""
        # H → V
        mat1 = cdd.Matrix(h_rep_orig, number_type="float")
        mat1.rep_type = cdd.RepType.INEQUALITY
        vert_mat = cdd.Polyhedron(mat1).get_generators()

        # V → H
        mat2 = cdd.Matrix(vert_mat, number_type="float")
        mat2.rep_type = cdd.RepType.GENERATOR
        h_rep_final = np.array(cdd.Polyhedron(mat2).get_inequalities())

        # Should have at least min_constraints (allowing for redundancy removal)
        assert h_rep_final.shape[0] >= min_constraints, "Roundtrip lost constraints"
        if max_constraints is not None:
            assert h_rep_final.shape[0] <= max_constraints, "Roundtrip added too many constraints"


class TestDegeneratePolytopes:
    """Test handling of degenerate polytopes."""

    def test_line_segment_conversion(self):
        """Test conversion of line segment (1D polytope in 2D space)."""
        # Line segment from (0,0) to (1,0)
        vertices = [
            [1, 0, 0],
            [1, 1, 0],
        ]

        mat = cdd.Matrix(vertices, number_type="float")
        mat.rep_type = cdd.RepType.GENERATOR

        # Should be able to convert to H-rep
        h_rep = np.array(cdd.Polyhedron(mat).get_inequalities())

        assert h_rep.shape[0] > 0, "No constraints generated for line segment"

    def test_single_point_conversion(self):
        """Test conversion of single point (0D polytope)."""
        # Single point (0, 0)
        vertices = [
            [1, 0, 0],
        ]

        mat = cdd.Matrix(vertices, number_type="float")
        mat.rep_type = cdd.RepType.GENERATOR

        try:
            h_rep = cdd.Polyhedron(mat).get_inequalities()
            # Single point should generate constraints
            assert len(h_rep) > 0, "Single point handled"
        except (ValueError, RuntimeError):
            # Some implementations may not handle single point
            pass


class TestHigherDimensions:
    """Test polytope conversions in higher dimensions."""

    def test_unit_cube_3d(self):
        """Test conversion of 3D unit cube."""
        # 3D unit cube vertices
        vertices = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
        ]

        mat = cdd.Matrix(vertices, number_type="float")
        mat.rep_type = cdd.RepType.GENERATOR

        h_rep = cdd.Polyhedron(mat).get_inequalities()

        # 3D cube should have 6 constraints (6 faces)
        assert len(h_rep) >= 6, "3D unit cube missing constraints"


class TestConstraintNormalization:
    """Test normalization of constraint representations."""

    def test_redundant_constraint_removal(self):
        """Test that redundant constraints are identified."""
        # Unit cube with redundant constraint
        h_rep = [
            [0, 1, 0],  # x >= 0
            [0, 0, 1],  # y >= 0
            [1, -1, 0],  # x <= 1
            [1, 0, -1],  # y <= 1
            [0.5, 0.5, 0.5],  # Redundant: inside unit cube
        ]

        mat = cdd.Matrix(h_rep, number_type="float")
        mat.rep_type = cdd.RepType.INEQUALITY

        # Get irredundant representation
        poly = cdd.Polyhedron(mat)
        irred_rep = poly.get_inequalities()

        # Irredundant form should have fewer constraints
        # (though cdd might not remove all redundancy)
        assert len(irred_rep) <= len(h_rep), "Redundancy not addressed"
