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

_LOG_MIN: float = 1e-6
_MAX_ITER: int = 100
_CONVERGE_TOL: float = 1e-4

# Disable the logging of Numba, which may be conflict with our logging.
logging.getLogger("numba").setLevel(logging.CRITICAL)


@njit(cache=True)
def get_parallel_tangent_line_sigmoid_np(
    k: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    """Find tangent line to sigmoid with given slope.

    Computes the tangent line y = k*x + b where the slope k is given,
    and the line is tangent to the sigmoid curve.

    :param k: Slope values. Shape: ``n,``.
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


@njit(cache=True)
def get_parallel_tangent_line_tanh_np(
    k: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    """Find tangent line to tanh with given slope.

    Computes the tangent line y = k*x + b where the slope k is given,
    and the line is tangent to the tanh curve.

    :param k: Slope values. Shape: ``n,``.
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


@njit(cache=True)
def _get_second_tangent_sigmoid_jit(
    x1: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray, bool]:
    """JIT convergence loop for sigmoid second tangent line (array-only).

    :param x1: First tangent x-coordinates. Shape: ``n,``.
    :param get_big: If True, upper tangent; else lower.
    :return: (b, k, x2, converged).
    """
    x2 = np.where(x1 == 0.0, 0.5, 0.0)
    y1 = np.reciprocal(1.0 + np.exp(-x1))
    b = np.zeros_like(x1)
    k = np.zeros_like(x1)
    x_new = np.zeros_like(x1)
    for _ in range(_MAX_ITER):
        y2 = np.reciprocal(1.0 + np.exp(-x2))
        k = (y2 - y1) / (x2 - x1)
        k = np.where(np.isnan(k), 0.1, k)
        b, k, x_new = get_parallel_tangent_line_sigmoid_np(k, get_big)
        if np.all(np.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new, True
        x2 = x_new
    return b, k, x_new, False


@njit(cache=True)
def _get_second_tangent_tanh_jit(
    x1: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray, bool]:
    """JIT convergence loop for tanh second tangent line (array-only).

    :param x1: First tangent x-coordinates. Shape: ``n,``.
    :param get_big: If True, upper tangent; else lower.
    :return: (b, k, x2, converged).
    """
    x2 = np.where(x1 == 0.0, 0.5, 0.0)
    y1 = np.tanh(x1)
    b = np.zeros_like(x1)
    k = np.zeros_like(x1)
    x_new = np.zeros_like(x1)
    for _ in range(_MAX_ITER):
        y2 = np.tanh(x2)
        k = (y2 - y1) / (x2 - x1)
        k = np.where(np.isnan(k), 0.1, k)
        b, k, x_new = get_parallel_tangent_line_tanh_np(k, get_big)
        if np.all(np.abs(x2 - x_new) < _CONVERGE_TOL):
            return b, k, x_new, True
        x2 = x_new
    return b, k, x_new, False


def get_second_tangent_line_sigmoid_np(
    x1: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    """Find second tangent line to sigmoid passing through point x1.

    Uses iterative method to find a tangent line that passes through
    the point (x1, sigmoid(x1)) on the sigmoid curve.

    :param x1: First tangent point x-coordinates. Shape: ``n,``.
    :param get_big: If True, return upper tangent; else lower tangent.
    :return: Tuple of (b, k, x2) where b is intercept, k is slope, x2 is
        second tangent point. Each has shape (n,).
    :raises NotConvergedError: If iteration does not converge.
    """
    b, k, x_new, converged = _get_second_tangent_sigmoid_jit(x1, get_big)
    if not converged:
        raise NotConvergedError
    return b, k, x_new


def get_second_tangent_line_tanh_np(
    x1: ndarray | float, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    """Find second tangent line to tanh passing through point x1.

    Uses iterative method to find a tangent line that passes through
    the point (x1, tanh(x1)) on the tanh curve.

    :param x1: First tangent point x-coordinates. Shape: ``n,`` or scalar.
    :param get_big: If True, return upper tangent; else lower tangent.
    :return: Tuple of (b, k, x2) where b is intercept, k is slope, x2 is
        second tangent point. Each has shape (n,).
    :raises NotConvergedError: If iteration does not converge.
    """
    if isinstance(x1, (int, float)):
        x1_arr = np.array([float(x1)], dtype=np.float64)
        b, k, x_new, converged = _get_second_tangent_tanh_jit(x1_arr, get_big)
        if not converged:
            raise NotConvergedError
        return float(b[0]), float(k[0]), float(x_new[0])  # type: ignore[return-value]
    b, k, x_new, converged = _get_second_tangent_tanh_jit(x1, get_big)
    if not converged:
        raise NotConvergedError
    return b, k, x_new
