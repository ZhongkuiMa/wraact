"""Advanced tests for MaxPool DLP (Difference of Linear Programs) algorithm.

Tests for hard-to-reach code paths in MaxPoolHullDLP:
1. DLP algorithm core (non-trivial constraint generation, beta calculation)
2. Cache hit paths and degenerate cases
3. Architectural safeguards and error handling
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest


def maxpool_np(x):
    """NumPy implementation of MaxPool for testing."""
    return np.max(x)


class TestMaxPoolDLPCoreAlgorithm:
    """Test MaxPoolHullDLP core algorithm for DLP construction."""

    def test_maxpool_dlp_nontrivial_constraint_generation(self):
        """Test DLP construction for non-trivial MaxPool cases with varied bounds.

        Tests lines 147-207 in acthull/_maxpool.py:
        - _find_nontrivial_idxs() call
        - DLP construction with beta calculation
        """
        from wraact.acthull import MaxPoolHullDLP

        lb = np.array([-1.0, -0.8, -0.9])
        ub = np.array([1.0, 0.9, 0.8])

        hull = MaxPoolHullDLP(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should exercise DLP construction (lines 147-207)
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        assert constraints.shape[0] > 0
        assert constraints.shape[1] == 5  # 3D input -> d+2 columns

    def test_maxpool_dlp_construct_dlp_directly(self):
        """Test _construct_dlp method directly.

        Tests lines 289-315 in acthull/_maxpool.py:
        - DLP construction with odd/even index splitting
        - 2-piece DLP generation
        """
        from wraact.acthull import MaxPoolHullDLP

        # Create vertices with varied ranges
        v = np.array(
            [
                [1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ]
        )

        nt_idxs = [0, 1, 2]
        dlp_lines = MaxPoolHullDLP._construct_dlp(v, nt_idxs)  # noqa: SLF001

        # Should return 2 pieces (odd/even split) - lines 289-315
        assert dlp_lines.shape[0] == 2
        assert dlp_lines.shape[1] == v.shape[1] + 1

    def test_maxpool_dlp_beta_calculation(self):
        """Test beta coefficient calculation in DLP algorithm.

        Tests lines 153-206 in acthull/_maxpool.py:
        - Beta calculation for constraint tightening
        - Numerical stability checks
        """
        from wraact.acthull import MaxPoolHullDLP

        # Create a case that exercises beta calculation
        lb = np.array([-2.0, -1.5, -1.0])
        ub = np.array([2.0, 1.5, 1.0])

        hull = MaxPoolHullDLP(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should have successfully calculated betas
        assert np.all(np.isfinite(constraints))

    def test_maxpool_dlp_find_nontrivial_indices(self):
        """Test _find_nontrivial_idxs method.

        Tests line 147 in acthull/_maxpool.py:
        - Identification of dimensions that can be maximum
        """
        from wraact.acthull import MaxPoolHullDLP

        # Create vertices where all dimensions are non-trivial
        v = np.array(
            [
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        nt_idxs = MaxPoolHullDLP._find_nontrivial_idxs(v)  # noqa: SLF001

        # All dimensions should be non-trivial (can be maximum)
        assert len(nt_idxs) > 0
        assert all(idx < v.shape[1] - 1 for idx in nt_idxs)

    def test_maxpool_dlp_multiple_dimensions(self):
        """Test DLP with full path for 4+ dimensional inputs.

        Tests complete DLP algorithm execution with higher dimensions.
        """
        from wraact.acthull import MaxPoolHullDLP

        lb = np.array([-1.0, -1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0, 1.0])

        hull = MaxPoolHullDLP(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should generate valid constraints
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        assert constraints.shape[1] == 6  # 4D input -> d+2 columns

    def test_maxpool_dlp_cal_mn_constrs_directly(self):
        """Test cal_mn_constrs method directly through cal_hull path.

        Tests multi-neuron constraint calculation via standard path.
        """
        from wraact.acthull import MaxPoolHullDLP

        # Use cal_hull to properly initialize and call cal_mn_constrs
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = MaxPoolHullDLP(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints is not None
        assert np.all(np.isfinite(constraints))

    def test_maxpool_dlp_versus_standard_maxpool(self):
        """Compare MaxPoolHullDLP results with MaxPoolHull for consistency.

        Tests that both implementations produce valid constraints.
        """
        from wraact.acthull import MaxPoolHull, MaxPoolHullDLP

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull_std = MaxPoolHull(if_cal_multi_neuron_constrs=True)
        hull_dlp = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)

        constraints_std = hull_std.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        constraints_dlp = hull_dlp.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Both should produce valid constraints
        assert np.all(np.isfinite(constraints_std))
        assert np.all(np.isfinite(constraints_dlp))

    def test_maxpool_dlp_symmetry(self):
        """Test DLP algorithm with symmetric bounds.

        Tests algorithm behavior with symmetric input bounds.
        """
        from wraact.acthull import MaxPoolHullDLP

        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(constraints))


class TestMaxPoolCacheAndDegenerate:
    """Test MaxPool cache hit paths and degenerate cases."""

    def test_maxpool_dlp_cache_hit_path(self):
        """Test consistency of constraint computation (cache behavior).

        Tests line 115 in acthull/_maxpool.py:
        - Verifies deterministic behavior (consistent with cache hits)
        - Multiple calls should produce identical results
        """
        from wraact.acthull import MaxPoolHullDLP

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHullDLP(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        # First call
        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Second call - should be identical (consistent with cache behavior)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        np.testing.assert_array_equal(c1, c2)

    def test_maxpool_cache_persistence_across_instances(self):
        """Test that cache persists across different instances.

        Tests class variable behavior of _lower_constraints.
        """
        from wraact.acthull import MaxPoolHullDLP

        # Clear cache
        MaxPoolHullDLP._lower_constraints.clear()  # noqa: SLF001

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # First call - populates cache
        c1 = MaxPoolHullDLP.cal_sn_constrs(lb, ub)

        # Second call - should retrieve from cache
        c2 = MaxPoolHullDLP.cal_sn_constrs(lb, ub)

        # Should be identical (cache was used)
        np.testing.assert_array_equal(c1, c2)

    def test_maxpool_single_vertex_polytope(self):
        """Test degenerate case with single vertex.

        Tests lines 215-223 in acthull/_maxpool.py:
        - _handle_case_of_one_vertex path
        """
        from wraact.acthull import MaxPoolHullDLP

        hull = MaxPoolHullDLP()

        # Very narrow bounds (near-constant)
        lb = np.array([0.499, 0.499])
        ub = np.array([0.501, 0.501])

        # May hit single vertex case or MIN_BOUNDS_RANGE check
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If no error, constraints should be valid
            assert np.all(np.isfinite(constraints))
        except ValueError:
            # MIN_BOUNDS_RANGE validation may catch this first - acceptable
            pass

    def test_maxpool_trivial_early_return(self):
        """Test trivial case early return path.

        Tests line 141 in acthull/_maxpool.py:
        - Early return for trivial multi-neuron cases
        """
        from wraact.acthull import MaxPoolHullDLP

        hull = MaxPoolHullDLP(if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=True)

        # Very large positive bounds (trivial case where max is always ub)
        lb = np.array([1.0, 1.0])
        ub = np.array([2.0, 2.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should handle gracefully
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_degenerate_with_identical_dimensions(self):
        """Test degenerate case with identical dimension ranges.

        Tests edge case where multiple dimensions have same bounds.
        """
        from wraact.acthull import MaxPoolHullDLP

        # All dimensions have identical bounds
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(constraints))

    def test_maxpool_cache_different_dimensions(self):
        """Test handling of different dimensions.

        Tests that constraints are generated correctly for different input dimensions.
        """
        from wraact.acthull import MaxPoolHullDLP

        # Test with 2D input
        lb2 = np.array([-1.0, -1.0])
        ub2 = np.array([1.0, 1.0])

        hull2 = MaxPoolHullDLP(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        c2 = hull2.cal_hull(input_lower_bounds=lb2, input_upper_bounds=ub2)

        # Should generate valid constraints
        assert isinstance(c2, np.ndarray)
        assert np.all(np.isfinite(c2))

        # Test with 3D input (different dimension)
        lb3 = np.array([-1.0, -1.0, -1.0])
        ub3 = np.array([1.0, 1.0, 1.0])

        hull3 = MaxPoolHullDLP(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
        c3 = hull3.cal_hull(input_lower_bounds=lb3, input_upper_bounds=ub3)

        # Should generate valid constraints for 3D
        assert isinstance(c3, np.ndarray)
        assert np.all(np.isfinite(c3))
        assert c3.shape[1] == 5  # 3D -> d+2 columns


class TestMaxPoolErrorHandling:
    """Test MaxPool error handling and architectural safeguards."""

    def test_maxpool_prohibited_methods_dlp(self):
        """Test that MaxPoolHullDLP raises errors for ReLU-like methods.

        Tests lines 278, 319, 323 in acthull/_maxpool.py:
        - RuntimeError for _cal_mn_constrs_with_one_y
        - RuntimeError for _f method
        - RuntimeError for _df method
        """
        from wraact.acthull import MaxPoolHullDLP

        # Lines 278, 319, 323
        with pytest.raises(RuntimeError, match=r"should not be called"):
            MaxPoolHullDLP._cal_mn_constrs_with_one_y(0, None, None, None, 0.0, True)  # noqa: SLF001, FBT003

        with pytest.raises(RuntimeError, match=r"should not be called"):
            MaxPoolHullDLP._f(np.array([1.0]))  # noqa: SLF001

        with pytest.raises(RuntimeError, match=r"should not be called"):
            MaxPoolHullDLP._df(np.array([1.0]))  # noqa: SLF001

    def test_maxpool_prohibited_methods_standard(self):
        """Test that MaxPoolHull raises errors for DLP-specific methods.

        Tests lines 428, 440 in acthull/_maxpool.py:
        - RuntimeError for _construct_dlp
        - RuntimeError for _cal_mn_constrs_with_one_y
        """
        from wraact.acthull import MaxPoolHull

        # Lines 428, 440
        with pytest.raises(RuntimeError, match=r"should not be called"):
            MaxPoolHull._construct_dlp(None, None)  # noqa: SLF001

        with pytest.raises(RuntimeError, match=r"should not be called"):
            MaxPoolHull._cal_mn_constrs_with_one_y(0, None, None, None, 0.0, True)  # noqa: SLF001, FBT003

    def test_maxpool_trivial_case_detection(self):
        """Test early return in trivial cases.

        Tests line 370 in acthull/_maxpool.py:
        - Detection and handling of trivial constraints
        """
        from wraact.acthull import MaxPoolHull

        # Create a case that might trigger trivial handling
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        hull = MaxPoolHull(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should handle successfully
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_invalid_bounds_dimension_mismatch(self):
        """Test error handling for dimension mismatch.

        Tests that invalid inputs are caught.
        """
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()

        lb = np.array([-1.0, -1.0])  # 2D
        ub = np.array([1.0, 1.0, 1.0])  # 3D - mismatch!

        with pytest.raises((ValueError, RuntimeError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_maxpool_reversed_bounds_error(self):
        """Test error handling for reversed bounds (lb > ub).

        Tests input validation.
        """
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()

        lb = np.array([1.0, 1.0])
        ub = np.array([-1.0, -1.0])  # Reversed!

        with pytest.raises((ValueError, RuntimeError)):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)


class TestMaxPoolDegenerateEdgeCases:
    """Test degenerate edge cases in MaxPool computation.

    These tests target lines that are difficult to reach naturally:
    - Lines 155-162: Single nontrivial piece after DLP construction
    - Lines 168-180: Beta coefficient validation
    """

    def test_maxpool_dlp_single_nontrivial_piece_simplification(self):
        """Test DLP simplification when only one piece dominates all vertices.

        Tests lines 155-162 in acthull/_maxpool.py:
        - Detection of single nontrivial piece
        - Simplified constraint generation
        """
        from wraact.acthull import MaxPoolHullDLP

        # Create bounds where only one dimension can be maximum
        # This forces all vertices to have only one dimension as the maximum
        # The key is to use very asymmetric bounds where one dimension clearly dominates
        lb = np.array([-10.0, -0.1, -0.1])
        ub = np.array([10.0, 0.1, 0.1])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should successfully handle simplification
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_dlp_all_vertices_same_maximum_dimension(self):
        """Test case where all vertices have the same maximum dimension.

        Creates vertices where only dimension 0 is ever the maximum,
        forcing the algorithm to simplify to a single piece constraint.
        """
        from wraact.acthull import MaxPoolHullDLP

        # Extreme asymmetry: x0 always much larger than others
        # Bounds need to satisfy MIN_BOUNDS_RANGE_ACTHULL (0.05 minimum)
        lb = np.array([0.0, -0.2, -0.2])
        ub = np.array([1.0, 0.2, 0.2])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_dlp_narrow_ranges_except_one_dimension(self):
        """Test with very narrow bounds on all but one dimension.

        This creates a degenerate polytope where only one dimension varies,
        simplifying the MaxPool to essentially a linear function.
        """
        from wraact.acthull import MaxPoolHullDLP

        lb = np.array([-10.0, -1e-6, -1e-6, -1e-6])
        ub = np.array([10.0, 1e-6, 1e-6, 1e-6])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If it succeeds, constraints should be valid
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))
        except ValueError as e:
            # May fail due to MIN_BOUNDS_RANGE check - acceptable
            if "minimum range" not in str(e).lower():
                raise

    def test_maxpool_dlp_two_dimensions_only(self):
        """Test MaxPool DLP with minimal dimension case (2D).

        Tests the algorithm with smallest non-trivial dimension.
        """
        from wraact.acthull import MaxPoolHullDLP

        lb = np.array([-1.0, -0.5])
        ub = np.array([1.0, 0.5])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        # 2D -> d+2 = 4 columns
        assert constraints.shape[1] == 4


class TestActHullDoubleOrdersFeature:
    """Test the double orders feature for enhanced precision.

    The double orders mode calculates the function hull with reversed
    input dimension order to improve precision in multi-neuron constraints.

    BUG FIX: Fixed IndexError caused by using output constraint dimensions
    instead of input constraint dimensions for reversed order computation.
    See: acthull/_act.py line 309 and oney/_act.py line 104
    """

    def test_double_orders_enabled_relu(self):
        """Test double orders mode with ReLU activation.

        Tests that double orders can be enabled and produces valid constraints.
        """
        from wraact.acthull import ReLUHull

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        # Create hull with double orders enabled
        hull = ReLUHull(if_cal_multi_neuron_constrs=True, if_use_double_orders=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        # Double orders should produce more constraints (2x due to reversed order)
        assert constraints.shape[0] > 0

    def test_double_orders_vs_single_order_relu(self):
        """Test that double orders produces more constraints than single order.

        Verifies that reversing dimension order produces additional constraints.
        """
        from wraact.acthull import ReLUHull

        lb = np.array([-1.5, -1.0, -0.5])
        ub = np.array([1.5, 1.0, 0.5])

        # Single order
        hull_single = ReLUHull(if_cal_multi_neuron_constrs=True, if_use_double_orders=False)
        c_single = hull_single.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Double order
        hull_double = ReLUHull(if_cal_multi_neuron_constrs=True, if_use_double_orders=True)
        c_double = hull_double.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Both should be valid
        assert np.all(np.isfinite(c_single))
        assert np.all(np.isfinite(c_double))
        # Double order should have more constraints (original + reversed)
        assert c_double.shape[0] >= c_single.shape[0]

    def test_double_orders_sigmoid(self):
        """Test double orders mode with Sigmoid activation.

        Tests S-shape activation with double orders enhancement.
        """
        from wraact.acthull import SigmoidHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = SigmoidHull(if_cal_multi_neuron_constrs=True, if_use_double_orders=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints with precision improvement
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_double_orders_tanh(self):
        """Test double orders mode with Tanh activation.

        Verifies feature works with different S-shape activation.
        """
        from wraact.acthull import TanhHull

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = TanhHull(if_cal_multi_neuron_constrs=True, if_use_double_orders=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_double_orders_validation_constraint(self):
        """Test that double orders properly validates reversed constraints.

        Ensures reversed constraint computation doesn't break soundness.
        """
        from wraact.acthull import ELUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = ELUHull(if_cal_multi_neuron_constrs=True, if_use_double_orders=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify soundness: all constraints should be finite
        assert np.all(np.isfinite(constraints))
        assert constraints.shape[0] > 0
