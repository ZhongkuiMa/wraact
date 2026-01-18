"""Tangent line computation for S-shaped activation functions.

This module provides functions to compute tangent lines for sigmoid and tanh
functions, used in convex hull construction for neural network verification.
"""

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
    """Find tangent line to sigmoid with given slope.

    Computes the tangent line y = k*x + b where the slope k is given,
    and the line is tangent to the sigmoid curve.

    :param k: Slope values. Shape: (n,).
    :param get_big: If True, return upper tangent; else lower tangent.
    :return: Tuple of (b, k, x) where b is intercept, k is slope, x is
        tangent point. Each has shape (n,).
    """
    sign = 1.0 if get_big else -1.0

    temp = np.maximum(1.0 - 4.0 * k, 0.0)  # Avoid minimal negative value
    sigma = 2.0 * np.reciprocal(1.0 + sign * np.sqrt(temp))
    temp = np.maximum(sigma - 1.0, _LOG_MIN)
    x = -np.log(temp)
    # b = sigmoid(x) - k * x
    b = np.reciprocal(1.0 + np.exp(-x)) - k * x

    return b, k, x


@njit
def get_parallel_tangent_line_tanh_np(
    k: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    """Find tangent line to tanh with given slope.

    Computes the tangent line y = k*x + b where the slope k is given,
    and the line is tangent to the tanh curve.

    :param k: Slope values. Shape: (n,).
    :param get_big: If True, return upper tangent; else lower tangent.
    :return: Tuple of (b, k, x) where b is intercept, k is slope, x is
        tangent point. Each has shape (n,).
    """
    sign = 1.0 if get_big else -1.0
    temp = np.maximum(1.0 - k, 0.0)  # Avoid minimal negative value
    sigma = sign * np.sqrt(temp)
    x = np.log((1.0 + sigma) / (1.0 - sigma)) * 0.5
    b = np.tanh(x) - k * x

    return b, k, x


def _warmup_jit_functions() -> None:
    """Compile Numba JIT functions on first call.

    This function triggers JIT compilation of tangent line functions
    to reduce latency on first use. It is called automatically at module
    import time to ensure functions are compiled before use.

    This is necessary for Numba-compiled functions to achieve good
    performance after initial compilation overhead.
    """
    rng = np.random.Generator(np.random.PCG64())
    k_np = rng.random(10) / 2
    get_big_np = True
    # Trigger JIT compilation for sigmoid tangent line
    get_parallel_tangent_line_sigmoid_np(k_np, get_big_np)
    # Trigger JIT compilation for tanh tangent line
    get_parallel_tangent_line_tanh_np(k_np, get_big_np)


# Warm up JIT functions at module import time
_warmup_jit_functions()


def get_second_tangent_line_sigmoid_np(
    x1: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    """Find second tangent line to sigmoid passing through point x1.

    Uses iterative method to find a tangent line that passes through
    the point (x1, sigmoid(x1)) on the sigmoid curve.

    :param x1: First tangent point x-coordinates. Shape: (n,).
    :param get_big: If True, return upper tangent; else lower tangent.
    :return: Tuple of (b, k, x2) where b is intercept, k is slope, x2 is
        second tangent point. Each has shape (n,).
    :raises NotConvergedError: If iteration does not converge.
    """
    x2 = np.where(x1 == 0.0, 0.5, 0.0)  # Initialize x2 away from x1 to avoid division by zero
    y1 = np.reciprocal(1.0 + np.exp(-x1))

    for _ in range(_MAX_ITER):
        y2 = np.reciprocal(1.0 + np.exp(-x2))
        with np.errstate(divide="ignore", invalid="ignore"):
            k = (y2 - y1) / (x2 - x1)
        # Handle any NaN values from division by zero
        k = np.where(np.isnan(k), 0.1, k)
        b, k, x_new = get_parallel_tangent_line_sigmoid_np(k, get_big)

        if np.all(np.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new

        x2 = x_new

    raise NotConvergedError


def get_second_tangent_line_tanh_np(
    x1: ndarray | float, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    """Find second tangent line to tanh passing through point x1.

    Uses iterative method to find a tangent line that passes through
    the point (x1, tanh(x1)) on the tanh curve.

    :param x1: First tangent point x-coordinates. Shape: (n,) or scalar.
    :param get_big: If True, return upper tangent; else lower tangent.
    :return: Tuple of (b, k, x2) where b is intercept, k is slope, x2 is
        second tangent point. Each has shape (n,).
    :raises NotConvergedError: If iteration does not converge.
    """
    # Initialize x2 away from x1 to avoid division by zero on first iteration
    if isinstance(x1, (int, float)):
        x2 = 0.5 if x1 == 0.0 else 0.0
    else:
        x2 = np.where(x1 == 0.0, 0.5, 0.0)  # type: ignore[assignment]
    y1 = np.tanh(x1)

    for _ in range(_MAX_ITER):
        y2 = np.tanh(x2)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = (y2 - y1) / (x2 - x1)
        # Handle any NaN values from division by zero
        if isinstance(k, (int, float)):
            k = 0.1 if np.isnan(k) else k
        else:
            k = np.where(np.isnan(k), 0.1, k)
        b, k, x_new = get_parallel_tangent_line_tanh_np(k, get_big)

        if np.all(np.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new

        x2 = x_new

    raise NotConvergedError
