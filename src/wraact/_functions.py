"""Activation functions and their derivatives for neural network verification.

This module provides NumPy implementations of common activation functions
and their first/second derivatives, used for computing convex hulls.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "delu_np",
    "dleakyrelu_np",
    "drelu_np",
    "dsigmoid_np",
    "dtanh_np",
    "elu_np",
    "leakyrelu_np",
    "relu_np",
    "sigmoid_np",
    "tanh_np",
]

from typing import cast, overload

import numpy as np
from numpy import ndarray


def relu_np(x: ndarray | float) -> ndarray | float:
    """Compute ReLU activation: max(0, x).

    :param x: Input array or scalar.
    :return: Output with same shape as input.
    """
    return np.maximum(x, 0.0)


def drelu_np(x: ndarray | float) -> ndarray | float:
    """Compute derivative of ReLU: 1 if x > 0, else 0.

    :param x: Input array or scalar.
    :return: Output with same shape as input.
    """
    return np.where(x > 0, 1.0, 0.0)


@overload
def sigmoid_np(x: ndarray) -> ndarray: ...


@overload
def sigmoid_np(x: float) -> float: ...  # type: ignore[misc]


def sigmoid_np(x: ndarray | float) -> ndarray | float:
    """Compute sigmoid activation: 1 / (1 + exp(-x)).

    :param x: Input array or scalar.
    :return: Output with same shape as input, values in (0, 1).
    """
    return cast(ndarray | float, np.reciprocal(1.0 + np.exp(-x)))


@overload
def dsigmoid_np(x: ndarray) -> ndarray: ...


@overload
def dsigmoid_np(x: float) -> float: ...  # type: ignore[misc]


def dsigmoid_np(x: ndarray | float) -> ndarray | float:
    """Compute first derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x)).

    :param x: Input array or scalar.
    :return: Output with same shape as input.
    """
    s = sigmoid_np(x)
    return s * (1.0 - s)


def ddsigmoid_np(x: ndarray | float) -> ndarray | float:
    """Compute second derivative of sigmoid.

    :param x: Input array or scalar.
    :return: Output with same shape as input.
    """
    s = sigmoid_np(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


@overload
def tanh_np(x: ndarray) -> ndarray: ...


@overload
def tanh_np(x: float) -> float: ...  # type: ignore[misc]


def tanh_np(x: ndarray | float) -> ndarray | float:
    """Compute tanh activation.

    :param x: Input array or scalar.
    :return: Output with same shape as input, values in (-1, 1).
    """
    return np.tanh(x)


@overload
def dtanh_np(x: ndarray) -> ndarray: ...


@overload
def dtanh_np(x: float) -> float: ...  # type: ignore[misc]


def dtanh_np(x: ndarray | float) -> ndarray | float:
    """Compute first derivative of tanh: 1 - tanh(x)^2.

    :param x: Input array or scalar.
    :return: Output with same shape as input.
    """
    return 1.0 - np.tanh(x) ** 2


def ddtanh_np(x: ndarray | float) -> ndarray | float:
    """Compute second derivative of tanh: -2 * tanh(x) * (1 - tanh(x)^2).

    :param x: Input array or scalar.
    :return: Output with same shape as input.
    """
    return -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2)


def elu_np(x: ndarray | float) -> ndarray | float:
    """Compute ELU activation: x if x > 0, else exp(x) - 1.

    :param x: Input array or scalar.
    :return: Output with same shape as input.
    """
    if isinstance(x, float):
        return x if x > 0 else np.exp(x) - 1.0
    # For arrays: avoid computing exp for positive values to prevent overflow
    result = np.empty_like(x, dtype=np.float64)
    mask_pos = x > 0
    result[mask_pos] = x[mask_pos]
    result[~mask_pos] = np.exp(x[~mask_pos]) - 1.0
    return result


def delu_np(x: ndarray | float) -> ndarray | float:
    """Compute derivative of ELU: 1 if x > 0, else exp(x).

    :param x: Input array or scalar.
    :return: Output with same shape as input.
    """
    if isinstance(x, float):
        return 1.0 if x > 0 else np.exp(x)
    # For arrays: avoid computing exp for positive values to prevent overflow
    result = np.empty_like(x, dtype=np.float64)
    mask_pos = x > 0
    result[mask_pos] = 1.0
    result[~mask_pos] = np.exp(x[~mask_pos])
    return result


def leakyrelu_np(x: ndarray | float, negative_slope: ndarray | float = 0.01) -> ndarray | float:
    """Compute Leaky ReLU: x if x > 0, else negative_slope * x.

    :param x: Input array or scalar.
    :param negative_slope: Slope for negative values. Default: 0.01.
    :return: Output with same shape as input.
    """
    return np.where(x > 0, x, negative_slope * x)


def dleakyrelu_np(x: ndarray | float, negative_slope: ndarray | float = 0.01) -> ndarray | float:
    """Compute derivative of Leaky ReLU: 1 if x > 0, else negative_slope.

    :param x: Input array or scalar.
    :param negative_slope: Slope for negative values. Default: 0.01.
    :return: Output with same shape as input.
    """
    return np.where(x > 0, 1.0, negative_slope)
