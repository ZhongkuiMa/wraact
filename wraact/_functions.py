__docformat__ = "restructuredtext"
__all__ = [
    "relu_np",
    "drelu_np",
    "sigmoid_np",
    "dsigmoid_np",
    "tanh_np",
    "dtanh_np",
    "elu_np",
    "delu_np",
    "leakyrelu_np",
    "dleakyrelu_np",
    "silu_np",
    "dsilu_np",
]

import numpy as np
from numpy import ndarray


def relu_np(x: ndarray | float) -> ndarray | float:
    return np.maximum(x, 0.0)


def drelu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, 0.0)


def sigmoid_np(x: ndarray | float) -> ndarray | float:
    return np.reciprocal(1.0 + np.exp(-x))


def dsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s)


def ddsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


def tanh_np(x: ndarray | float) -> ndarray | float:
    return np.tanh(x)


def dtanh_np(x: ndarray | float) -> ndarray | float:
    return 1.0 - np.tanh(x) ** 2


def ddtanh_np(x: ndarray | float) -> ndarray | float:
    return -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2)


def elu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, x, np.exp(x) - 1.0)


def delu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, np.exp(x))


def leakyrelu_np(
    x: ndarray | float, negative_slope: ndarray | float = 0.01
) -> ndarray | float:

    return np.where(x > 0, x, negative_slope * x)


def dleakyrelu_np(
    x: ndarray | float, negative_slope: ndarray | float = 0.01
) -> ndarray | float:
    return np.where(x > 0, 1.0, negative_slope)


def silu_np(x: ndarray | float) -> ndarray | float:
    return np.reciprocal(1.0 + np.exp(-x)) * x


def dsilu_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s + x * s * (1.0 - s)
