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
    return np.maximum(x, 0.0)


def drelu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, 0.0)


@overload
def sigmoid_np(x: ndarray) -> ndarray: ...


@overload
def sigmoid_np(x: float) -> float: ...  # type: ignore[misc]


def sigmoid_np(x: ndarray | float) -> ndarray | float:
    return cast(ndarray | float, np.reciprocal(1.0 + np.exp(-x)))


@overload
def dsigmoid_np(x: ndarray) -> ndarray: ...


@overload
def dsigmoid_np(x: float) -> float: ...  # type: ignore[misc]


def dsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s)


def ddsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


@overload
def tanh_np(x: ndarray) -> ndarray: ...


@overload
def tanh_np(x: float) -> float: ...  # type: ignore[misc]


def tanh_np(x: ndarray | float) -> ndarray | float:
    return np.tanh(x)


@overload
def dtanh_np(x: ndarray) -> ndarray: ...


@overload
def dtanh_np(x: float) -> float: ...  # type: ignore[misc]


def dtanh_np(x: ndarray | float) -> ndarray | float:
    return 1.0 - np.tanh(x) ** 2


def ddtanh_np(x: ndarray | float) -> ndarray | float:
    return -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2)


def elu_np(x: ndarray | float) -> ndarray | float:
    if isinstance(x, float):
        return x if x > 0 else np.exp(x) - 1.0
    # For arrays: avoid computing exp for positive values to prevent overflow
    result = np.empty_like(x, dtype=np.float64)
    mask_pos = x > 0
    result[mask_pos] = x[mask_pos]
    result[~mask_pos] = np.exp(x[~mask_pos]) - 1.0
    return result


def delu_np(x: ndarray | float) -> ndarray | float:
    if isinstance(x, float):
        return 1.0 if x > 0 else np.exp(x)
    # For arrays: avoid computing exp for positive values to prevent overflow
    result = np.empty_like(x, dtype=np.float64)
    mask_pos = x > 0
    result[mask_pos] = 1.0
    result[~mask_pos] = np.exp(x[~mask_pos])
    return result


def leakyrelu_np(x: ndarray | float, negative_slope: ndarray | float = 0.01) -> ndarray | float:
    return np.where(x > 0, x, negative_slope * x)


def dleakyrelu_np(x: ndarray | float, negative_slope: ndarray | float = 0.01) -> ndarray | float:
    return np.where(x > 0, 1.0, negative_slope)
