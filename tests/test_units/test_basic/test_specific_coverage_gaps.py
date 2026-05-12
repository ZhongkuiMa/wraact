"""Explicitly test remaining coverage gaps not found by fuzzing."""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact import _constants
from wraact.acthull import ELUHull, LeakyReLUHull, MaxPoolHullDLP, ReLUHull


class TestMaxPoolCacheHit:
    """Test MaxPool cache retrieval (line 115 in _maxpool.py)."""

    @pytest.mark.parametrize(
        ("lb", "ub"),
        [
            pytest.param(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]), id="3d"),
            pytest.param(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), id="2d"),
            pytest.param(
                np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), id="4d"
            ),
        ],
    )
    def test_cache_hit(self, lb, ub):
        """Test cache retrieval on second call with same dimension."""
        c1 = MaxPoolHullDLP.cal_sn_constrs(lb, ub)
        c2 = MaxPoolHullDLP.cal_sn_constrs(lb, ub)

        np.testing.assert_array_equal(c1, c2)


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

    @pytest.mark.parametrize(
        ("lb", "ub", "scenario"),
        [
            pytest.param(np.array([0.5, 0.5]), np.array([0.5, 0.5]), "2d", id="2d"),
            pytest.param(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "3d", id="3d"),
        ],
    )
    def test_single_vertex(self, lb, ub, scenario):
        """Test MaxPool with single vertex (constant function)."""
        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[0] >= 2  # At least upper and lower bounds
        assert np.all(np.isfinite(constraints))


class TestMaxPoolSinglePiece:
    """Test MaxPool with single dominant piece."""

    @pytest.mark.parametrize(
        ("lb", "ub", "scenario"),
        [
            pytest.param(np.array([10.0, -0.1, -0.1]), np.array([20.0, 0.1, 0.1]), "3d", id="3d"),
            pytest.param(
                np.array([100.0, -1.0, -1.0, -1.0]), np.array([200.0, 1.0, 1.0, 1.0]), "4d", id="4d"
            ),
        ],
    )
    def test_single_piece(self, lb, ub, scenario):
        """Test DLP where only one piece is ever maximum."""
        hull = MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
