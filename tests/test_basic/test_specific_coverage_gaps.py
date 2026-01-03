"""Explicitly test remaining coverage gaps not found by fuzzing."""

__docformat__ = "restructuredtext"

import numpy as np

from wraact import _constants
from wraact.acthull import MaxPoolHullDLP, ReLUHull


class TestMaxPoolCacheHit:
    """Test MaxPool cache retrieval (line 115 in _maxpool.py)."""

    def test_cache_hit_on_second_call(self):
        """Second call should retrieve from cache."""
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        # First call - populates cache
        c1 = MaxPoolHullDLP.cal_sn_constrs(lb, ub)

        # Second call - retrieves from cache (line 115)
        c2 = MaxPoolHullDLP.cal_sn_constrs(lb, ub)

        np.testing.assert_array_equal(c1, c2)

    def test_cache_hit_multiple_dimensions(self):
        """Test cache with different dimensions."""
        # Test dimension 2
        lb2 = np.array([-1.0, -1.0])
        ub2 = np.array([1.0, 1.0])
        c2_first = MaxPoolHullDLP.cal_sn_constrs(lb2, ub2)
        c2_second = MaxPoolHullDLP.cal_sn_constrs(lb2, ub2)
        np.testing.assert_array_equal(c2_first, c2_second)

        # Test dimension 4
        lb4 = np.array([-1.0, -1.0, -1.0, -1.0])
        ub4 = np.array([1.0, 1.0, 1.0, 1.0])
        c4_first = MaxPoolHullDLP.cal_sn_constrs(lb4, ub4)
        c4_second = MaxPoolHullDLP.cal_sn_constrs(lb4, ub4)
        np.testing.assert_array_equal(c4_first, c4_second)


class TestDEBUGMode:
    """Test DEBUG mode paths (lines 345-347, 384-385 in _act.py)."""

    def test_debug_mode_direct_computation(self):
        """With DEBUG=True, no exception handling wrapper."""
        original_debug = _constants.DEBUG

        try:
            _constants.DEBUG = True

            lb = np.array([-1.0, -1.0])
            ub = np.array([1.0, 1.0])

            hull = ReLUHull()
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            # Should execute lines 345-347, 384-385
            assert np.all(np.isfinite(constraints))
            assert constraints.shape[0] > 0

        finally:
            _constants.DEBUG = original_debug

    def test_debug_mode_multiple_activations(self):
        """Test DEBUG mode with different activation functions."""
        from wraact.acthull import ELUHull, LeakyReLUHull

        original_debug = _constants.DEBUG

        try:
            _constants.DEBUG = True

            lb = np.array([-1.0, -1.0])
            ub = np.array([1.0, 1.0])

            # Test ELU
            hull_elu = ELUHull()
            c_elu = hull_elu.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(c_elu))

            # Test LeakyReLU
            hull_lrelu = LeakyReLUHull()
            c_lrelu = hull_lrelu.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(c_lrelu))

        finally:
            _constants.DEBUG = original_debug

    def test_debug_mode_restored_after_test(self):
        """Test that DEBUG state is properly restored."""
        original_debug = _constants.DEBUG

        try:
            # Change DEBUG state
            _constants.DEBUG = not original_debug
            assert (not original_debug) == _constants.DEBUG

        finally:
            # Restore original
            _constants.DEBUG = original_debug

        # Should be restored
        assert original_debug == _constants.DEBUG


class TestMaxPoolSingleVertex:
    """Test MaxPool with single vertex (constant function)."""

    def test_single_vertex_case(self):
        """Test MaxPool with single vertex."""
        # Single point = single vertex
        lb = ub = np.array([0.5, 0.5])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should produce valid constraints for constant function
        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] >= 2  # At least upper and lower bounds
        assert np.all(np.isfinite(constraints))

    def test_single_vertex_3d(self):
        """Test 3D single vertex case."""
        lb = ub = np.array([1.0, 2.0, 3.0])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestMaxPoolSinglePiece:
    """Test MaxPool with single dominant piece."""

    def test_single_nontrivial_piece_simplification(self):
        """Test DLP where only one piece is ever maximum."""
        # Extreme asymmetry: x0 always dominates
        lb = np.array([10.0, -0.1, -0.1])
        ub = np.array([20.0, 0.1, 0.1])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should successfully simplify to single piece
        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_single_piece_4d(self):
        """Test 4D case with one dimension clearly dominant."""
        # First dimension always largest
        lb = np.array([100.0, -1.0, -1.0, -1.0])
        ub = np.array([200.0, 1.0, 1.0, 1.0])

        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
