"""Shared fixtures for test_basic test suite."""

__docformat__ = "restructuredtext"

import pytest

from wraact.acthull import MaxPoolHullDLP


@pytest.fixture
def max_pool_hull_dlp():
    """MaxPoolHullDLP instance with multi-neuron constraints enabled."""
    return MaxPoolHullDLP(if_cal_multi_neuron_constrs=True)
