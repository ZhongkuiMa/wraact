__docformat__ = "restructuredtext"
__all__ = [
    "get_parallel_tangent_line_sigmoid_np",
    "get_parallel_tangent_line_tanh_np",
    "get_second_tangent_line_sigmoid_np",
    "get_second_tangent_line_tanh_np",
]


import logging

import numpy as np
from numba import njit
from numpy import ndarray

from wraact._exceptions import NotConvergedError

_LOG_MIN = 1e-6
_MAX_ITER = 100
_CONVERGE_TOL = 1e-4

# Disable the logging of Numba, which may be conflict with our logging.
logging.getLogger("numba").setLevel(logging.CRITICAL)


@njit
def get_parallel_tangent_line_sigmoid_np(
    k: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    sign = 1.0 if get_big else -1.0

    temp = np.maximum(1.0 - 4.0 * k, 0.0)  # Avoid minimal negative value
    sigma = 2.0 * np.reciprocal(1.0 + sign * np.sqrt(temp))
    temp = np.maximum(sigma - 1.0, _LOG_MIN)
    x = -np.log(temp)
    # b = sigmoid(x) - k * x
    b = np.reciprocal(1.0 + np.exp(-x)) - k * x

    return b, k, x


rng = np.random.Generator(np.random.PCG64())
k_np = rng.random(10) / 2
get_big_np = True
get_parallel_tangent_line_sigmoid_np(k_np, get_big_np)


@njit
def get_parallel_tangent_line_tanh_np(
    k: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    sign = 1.0 if get_big else -1.0
    temp = np.maximum(1.0 - k, 0.0)  # Avoid minimal negative value
    sigma = sign * np.sqrt(temp)
    x = np.log((1.0 + sigma) / (1.0 - sigma)) * 0.5
    b = np.tanh(x) - k * x

    return b, k, x


get_parallel_tangent_line_tanh_np(k_np, get_big_np)


def get_second_tangent_line_sigmoid_np(
    x1: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    x2 = np.zeros_like(x1, dtype=np.float64)
    y1 = np.reciprocal(1.0 + np.exp(-x1))

    for _ in range(_MAX_ITER):
        y2 = np.reciprocal(1.0 + np.exp(-x2))
        k = (y2 - y1) / (x2 - x1)
        b, k, x_new = get_parallel_tangent_line_sigmoid_np(k, get_big)

        if np.all(np.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new

        x2 = x_new

    raise NotConvergedError


def get_second_tangent_line_tanh_np(
    x1: ndarray | float, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    x2 = np.zeros_like(x1, dtype=np.float64)
    y1 = np.tanh(x1)

    for _ in range(_MAX_ITER):
        y2 = np.tanh(x2)
        k = (y2 - y1) / (x2 - x1)
        b, k, x_new = get_parallel_tangent_line_tanh_np(k, get_big)

        if np.all(np.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new

        x2 = x_new

    raise NotConvergedError
