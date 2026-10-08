"""Strict graph-containment checks for every public activation-hull surface."""

__docformat__ = "restructuredtext"

from itertools import product

import numpy as np
import pytest

from wraact import (
    ELUHull,
    ELUHullWithOneY,
    LeakyReLUHull,
    LeakyReLUHullWithOneY,
    MaxPoolHull,
    MaxPoolHullDLP,
    MaxPoolHullDLPWithOneY,
    MaxPoolHullWithOneY,
    ReLUHull,
    ReLUHullWithOneY,
    SigmoidHull,
    SigmoidHullWithOneY,
    TanhHull,
    TanhHullWithOneY,
)
from wraact._functions import elu_np, leakyrelu_np, relu_np, sigmoid_np, tanh_np

_ELEMENTWISE_HULLS = [
    pytest.param(ReLUHull, relu_np, id="relu"),
    pytest.param(LeakyReLUHull, leakyrelu_np, id="leakyrelu"),
    pytest.param(ELUHull, elu_np, id="elu"),
    pytest.param(SigmoidHull, sigmoid_np, id="sigmoid"),
    pytest.param(TanhHull, tanh_np, id="tanh"),
]

_ONE_Y_HULLS = [
    pytest.param(ReLUHullWithOneY, relu_np, id="relu"),
    pytest.param(LeakyReLUHullWithOneY, leakyrelu_np, id="leakyrelu"),
    pytest.param(ELUHullWithOneY, elu_np, id="elu"),
    pytest.param(SigmoidHullWithOneY, sigmoid_np, id="sigmoid"),
    pytest.param(TanhHullWithOneY, tanh_np, id="tanh"),
]


def _sample_box(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Return deterministic interior samples plus every box corner."""
    rng = np.random.default_rng(20261007)
    random_points = rng.uniform(lb, ub, size=(1024, lb.size))
    corners = np.asarray(list(product(*zip(lb, ub, strict=True))), dtype=np.float64)
    return np.vstack((random_points, corners, lb, ub, (lb + ub) / 2.0))


def _assert_contains(constraints: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Assert every concrete graph point satisfies every returned constraint."""
    points = np.hstack((x, y))
    margins = constraints[:, :1] + constraints[:, 1:] @ points.T
    assert np.all(np.isfinite(constraints))
    assert np.min(margins) >= -1e-8


@pytest.mark.parametrize(("hull_class", "activation"), _ELEMENTWISE_HULLS)
@pytest.mark.parametrize("mode", ["multi", "single", "double"])
def test_elementwise_public_hulls_contain_asymmetric_graph(hull_class, activation, mode):
    """Check all full-output modes on a non-symmetric, mixed-sign box."""
    lb = np.array([-2.0, -0.5, -0.1])
    ub = np.array([0.3, 1.7, 2.5])
    kwargs = {
        "if_cal_single_neuron_constrs": mode == "single",
        "if_cal_multi_neuron_constrs": mode != "single",
        "if_use_double_orders": mode == "double",
    }
    constraints = hull_class(**kwargs).cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
    x = _sample_box(lb, ub)

    _assert_contains(constraints, x, activation(x))


@pytest.mark.parametrize(("hull_class", "activation"), _ONE_Y_HULLS)
def test_one_y_public_hulls_contain_asymmetric_graph(hull_class, activation):
    """Check single-output hulls against the first activation coordinate."""
    lb = np.array([-2.0, -0.5, 0.2])
    ub = np.array([0.3, 1.7, 2.5])
    constraints = hull_class().cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
    x = _sample_box(lb, ub)

    _assert_contains(constraints, x, activation(x[:, :1]))


@pytest.mark.parametrize(
    "hull",
    [
        pytest.param(ELUHull(), id="full"),
        pytest.param(ELUHull(if_use_double_orders=True), id="double"),
        pytest.param(ELUHullWithOneY(), id="one-y"),
    ],
)
def test_elu_hulls_contain_wide_crossing_domain(hull):
    """Keep the wide-domain ELU DLP anchored to the activation graph."""
    lb = np.array([-3.0])
    ub = np.array([5.0])
    x = np.vstack((_sample_box(lb, ub), np.zeros((1, 1))))

    constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    _assert_contains(constraints, x, elu_np(x))


@pytest.mark.parametrize(
    "hull_class",
    [MaxPoolHull, MaxPoolHullDLP, MaxPoolHullWithOneY, MaxPoolHullDLPWithOneY],
)
def test_maxpool_public_hulls_contain_asymmetric_graph(hull_class):
    """Check exact and compatibility MaxPool surfaces on an asymmetric box."""
    lb = np.array([-3.0, -0.5, 0.2])
    ub = np.array([0.3, 1.7, 2.5])
    constraints = hull_class().cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
    x = _sample_box(lb, ub)

    _assert_contains(constraints, x, np.max(x, axis=1, keepdims=True))
