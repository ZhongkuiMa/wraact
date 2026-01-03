__docformat__ = "restructuredtext"
__all__ = ["TanhHull"]

from numpy import ndarray

from wraact._functions import dtanh_np, tanh_np
from wraact.acthull._sshape import SShapeHull


class TanhHull(SShapeHull):
    """
    This is to calculate the function hull for the hyperbolic tangent activation function.

    Please refer to the :class:`SShapeHull` for more details.
    """

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        from wraact._tangent_lines import get_second_tangent_line_tanh_np

        return get_second_tangent_line_tanh_np(x1, get_big)  # type: ignore[arg-type,return-value]

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        from wraact._tangent_lines import get_parallel_tangent_line_tanh_np

        return get_parallel_tangent_line_tanh_np(k, get_big)  # type: ignore[arg-type,return-value]

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return tanh_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dtanh_np(x)
