__docformat__ = "restructuredtext"
__all__ = ["SigmoidHull"]

from numpy import ndarray

from wraact._functions import dsigmoid_np, sigmoid_np
from wraact.acthull._sshape import SShapeHull


class SigmoidHull(SShapeHull):
    """
    This is to calculate the function hull for the sigmoid activation function.

    Please refer to the :class:`SShapeHull` for more details.
    """

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        from wraact._tangent_lines import get_second_tangent_line_sigmoid_np

        return get_second_tangent_line_sigmoid_np(x1, get_big)  # type: ignore[arg-type,return-value]

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        from wraact._tangent_lines import get_parallel_tangent_line_sigmoid_np

        return get_parallel_tangent_line_sigmoid_np(k, get_big)  # type: ignore[arg-type,return-value]

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return sigmoid_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dsigmoid_np(x)
