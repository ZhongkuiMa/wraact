"""MaxPool DLP-specific tests targeting cache and trivial case coverage.

This module tests MaxPoolHullDLP specific functionality:
1. Lower constraint caching (class variable _lower_constraints)
2. Trivial case handling (one vertex, one piece)
3. Non-trivial index detection
4. DLP construction and multi-neuron constraint generation
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact import DegeneratedError
from wraact.acthull import MaxPoolHullDLP


class TestMaxPoolDLPLowerConstraintCache:
    """Test lower constraint caching in MaxPoolHullDLP."""

    def test_maxpool_dlp_cache_same_dimension(self, maxpool_hull_class):
        """Test that lower constraints are cached for repeated dimensions."""
        hull = maxpool_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # First call - should compute and cache
        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Second call with same dimension - should use cache
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Results should be identical
        np.testing.assert_array_equal(c1, c2)

    def test_maxpool_dlp_cache_2d_then_3d(self, maxpool_hull_class):
        """Test cache behavior when switching between dimensions."""
        hull = maxpool_hull_class()

        # First: 2D
        c1 = hull.cal_hull(
            input_lower_bounds=np.array([-1.0, -1.0]), input_upper_bounds=np.array([1.0, 1.0])
        )

        # Second: 3D
        hull.cal_hull(
            input_lower_bounds=np.array([-1.0, -1.0, -1.0]),
            input_upper_bounds=np.array([1.0, 1.0, 1.0]),
        )

        # Third: Back to 2D - should use cached constraints
        c3 = hull.cal_hull(
            input_lower_bounds=np.array([-1.0, -1.0]), input_upper_bounds=np.array([1.0, 1.0])
        )

        # c1 and c3 should be identical
        np.testing.assert_array_equal(c1, c3)

    def test_maxpool_dlp_cache_multiple_2d_calls(self, maxpool_hull_class):
        """Test cache consistency across multiple 2D calls."""
        hull = maxpool_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        results = []
        for _ in range(5):
            c = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(c)

        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


class TestMaxPoolDLPTrivialCases:
    """Test trivial case detection in MaxPool DLP."""

    def test_maxpool_dlp_single_vertex_polytope(self, maxpool_hull_class):
        """Test MaxPool with single vertex polytope (constant output)."""
        hull = maxpool_hull_class()
        # Single point (not actually single vertex, but nearly constant)
        lb = np.array([0.5, 0.5])
        ub = np.array([0.5001, 0.5001])

        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
        except DegeneratedError:
            # May raise due to minimum range threshold
            pass

    def test_maxpool_dlp_one_piece_case(self, maxpool_hull_class):
        """Test MaxPool case where one input dominates (one piece)."""
        hull = maxpool_hull_class()
        # One dimension much larger than others
        lb = np.array([10.0, -1.0])
        ub = np.array([11.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should return valid constraints
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_dlp_dominant_dimension(self, maxpool_hull_class):
        """Test MaxPool where one dimension is always largest."""
        hull = maxpool_hull_class()
        # First dimension always larger
        lb = np.array([5.0, -10.0, -10.0])
        ub = np.array([10.0, 0.0, 0.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestMaxPoolDLPMultiNeuronConstraints:
    """Test multi-neuron constraint generation in MaxPool DLP."""

    def test_maxpool_dlp_mn_constrs_direct_2d(self):
        """Test cal_mn_constrs method directly for 2D."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Create input constraints for polytope
        c = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Compute vertices
        hull = MaxPoolHullDLP()
        v, _ = hull.cal_vertices(c, "float")

        # Call cal_mn_constrs directly
        cc = MaxPoolHullDLP.cal_mn_constrs(c, v, lb, ub)

        assert isinstance(cc, np.ndarray)
        assert np.all(np.isfinite(cc))

    def test_maxpool_dlp_mn_constrs_3d(self):
        """Test cal_mn_constrs for 3D input."""
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        # Create polytope constraints
        c = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )

        hull = MaxPoolHullDLP()
        v, _ = hull.cal_vertices(c, "float")

        # Call directly
        cc = MaxPoolHullDLP.cal_mn_constrs(c, v, lb, ub)

        assert isinstance(cc, np.ndarray)
        assert np.all(np.isfinite(cc))


class TestMaxPoolDLPSingleNeuronConstraints:
    """Test single-neuron constraint generation."""

    @pytest.mark.parametrize(
        ("dim", "lb", "ub", "expected_cols"),
        [
            pytest.param(2, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 4, id="2d"),
            pytest.param(
                4, np.array([-2.0, -1.0, -3.0, -0.5]), np.array([2.0, 1.0, 3.0, 0.5]), 6, id="4d"
            ),
        ],
    )
    def test_maxpool_dlp_sn_constrs(self, dim, lb, ub, expected_cols):
        """Test single-neuron constraints for MaxPool with given dimension."""
        cc = MaxPoolHullDLP.cal_sn_constrs(lb, ub)

        assert isinstance(cc, np.ndarray)
        assert cc.shape[1] == expected_cols
        assert np.all(np.isfinite(cc))

    def test_maxpool_dlp_sn_constrs_consistency(self):
        """Test single-neuron constraints are consistent."""
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        cc1 = MaxPoolHullDLP.cal_sn_constrs(lb, ub)
        cc2 = MaxPoolHullDLP.cal_sn_constrs(lb, ub)

        np.testing.assert_array_equal(cc1, cc2)


class TestMaxPoolDLPVariousInputs:
    """Test MaxPool DLP with various input configurations."""

    @pytest.mark.parametrize(
        ("lb", "ub"),
        [
            pytest.param(
                np.array([-5.0, 2.0, -1.0]),
                np.array([1.0, 8.0, 1.0]),
                id="mixed_signs",
            ),
            pytest.param(
                np.array([-100.0, -50.0, -200.0]),
                np.array([100.0, 50.0, 200.0]),
                id="wide_ranges",
            ),
            pytest.param(
                np.array([-0.1, -0.1]),
                np.array([0.1, 0.1]),
                id="narrow_ranges",
            ),
        ],
    )
    def test_maxpool_dlp_various_input_ranges(self, maxpool_hull_class, lb, ub):
        """Test MaxPool DLP with various input bound configurations."""
        hull = maxpool_hull_class()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        # [REVIEW] Deleted: test_maxpool_dlp_all_positive_all_negative_mix, test_maxpool_dlp_wide_ranges, test_maxpool_dlp_narrow_ranges

    def test_maxpool_dlp_uniform_bounds(self, maxpool_hull_class):
        """Test MaxPool with uniform bounds."""
        hull = maxpool_hull_class()
        for d in range(2, 8):
            lb = -5.0 * np.ones(d)
            ub = 5.0 * np.ones(d)

            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert constraints.shape[1] == d + 2
            assert np.all(np.isfinite(constraints))


class TestMaxPoolDLPConstraintProperties:
    """Test properties of MaxPool DLP constraints."""

    def test_maxpool_dlp_constraints_finite_2d(self, maxpool_hull_class):
        """Test MaxPool constraints are always finite."""
        hull = maxpool_hull_class()
        test_cases = [
            (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
            (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
            (np.array([-100.0, -50.0]), np.array([50.0, 100.0])),
        ]

        for lb, ub in test_cases:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints)), f"Non-finite constraints for lb={lb}, ub={ub}"

    def test_maxpool_dlp_constraints_deterministic_2d(self, maxpool_hull_class):
        """Test MaxPool constraints are deterministic."""
        hull = maxpool_hull_class()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        results = []
        for _ in range(5):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints)

        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_maxpool_dlp_upper_bound_constraints_dominate(self, maxpool_hull_class):
        """Test that MaxPool upper bound constraints are sensible."""

        def maxpool_np(x):
            return np.max(x)

        hull = maxpool_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify constraints at extreme points
        test_points = [
            np.array([1.0, 1.0]),  # Both max
            np.array([1.0, -1.0]),  # One max
            np.array([-1.0, 1.0]),  # Other max
            np.array([-1.0, -1.0]),  # Both min
        ]

        for x in test_points:
            y = maxpool_np(x)
            point = np.concatenate([x, [y]])

            b = constraints[:, 0]
            a = constraints[:, 1:]
            constraint_values = b + a @ point

            # All constraints should be satisfied
            assert np.all(constraint_values >= -1e-6), f"Constraint violation for x={x}, y={y}"
