"""Sigmoid activation hull computation.

This module provides the SigmoidHull class for computing convex hulls
of the sigmoid activation function.
"""

__docformat__ = "restructuredtext"
__all__ = ["SigmoidHull"]

from numpy import ndarray

from wraact._functions import dsigmoid_np, sigmoid_np
from wraact.acthull._sshape import SShapeHull


class SigmoidHull(SShapeHull):
    """Compute convex hull for sigmoid activation function.

    Sigmoid is defined as f(x) = 1 / (1 + exp(-x)). This class computes
    tight convex hull constraints for neural network verification.
    See :class:`SShapeHull` for inherited methods.
    """

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        """Find second tangent line to sigmoid passing through x1.

        :param x1: First tangent point x-coordinates.
        :param get_big: If True, return upper tangent; else lower.
        :return: Tuple of (intercept, slope, tangent_point).
        """
        from wraact._tangent_lines import get_second_tangent_line_sigmoid_np

        return get_second_tangent_line_sigmoid_np(x1, get_big)  # type: ignore[arg-type,return-value]

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        """Find tangent line to sigmoid with given slope.

        :param k: Slope values.
        :param get_big: If True, return upper tangent; else lower.
        :return: Tuple of (intercept, slope, tangent_point).
        """
        from wraact._tangent_lines import get_parallel_tangent_line_sigmoid_np

        return get_parallel_tangent_line_sigmoid_np(k, get_big)  # type: ignore[arg-type,return-value]

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        """Evaluate sigmoid function.

        :param x: Input values.
        :return: Sigmoid output.
        """
        return sigmoid_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        """Evaluate sigmoid derivative.

        :param x: Input values.
        :return: Sigmoid derivative output.
        """
        return dsigmoid_np(x)
