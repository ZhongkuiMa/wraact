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
def sigmoid_np(x: float) -> float: ...

def sigmoid_np(x: ndarray | float) -> ndarray | float:
    return cast(ndarray | float, np.reciprocal(1.0 + np.exp(-x)))


@overload
def dsigmoid_np(x: ndarray) -> ndarray: ...

@overload
def dsigmoid_np(x: float) -> float: ...

def dsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s)


def ddsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


@overload
def tanh_np(x: ndarray) -> ndarray: ...

@overload
def tanh_np(x: float) -> float: ...

def tanh_np(x: ndarray | float) -> ndarray | float:
    return np.tanh(x)


@overload
def dtanh_np(x: ndarray) -> ndarray: ...

@overload
def dtanh_np(x: float) -> float: ...

def dtanh_np(x: ndarray | float) -> ndarray | float:
    return 1.0 - np.tanh(x) ** 2


def ddtanh_np(x: ndarray | float) -> ndarray | float:
    return -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2)


def elu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, x, np.exp(x) - 1.0)


def delu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, np.exp(x))


def leakyrelu_np(x: ndarray | float, negative_slope: ndarray | float = 0.01) -> ndarray | float:
    return np.where(x > 0, x, negative_slope * x)


def dleakyrelu_np(x: ndarray | float, negative_slope: ndarray | float = 0.01) -> ndarray | float:
    return np.where(x > 0, 1.0, negative_slope)


@overload
def silu_np(x: ndarray) -> ndarray: ...

@overload
def silu_np(x: float) -> float: ...

def silu_np(x: ndarray | float) -> ndarray | float:
    return cast(ndarray | float, np.reciprocal(1.0 + np.exp(-x)) * x)


@overload
def dsilu_np(x: ndarray) -> ndarray: ...

@overload
def dsilu_np(x: float) -> float: ...

def dsilu_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s + x * s * (1.0 - s)
