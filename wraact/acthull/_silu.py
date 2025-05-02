__docformat__ = ["restructuredtext"]
__all__ = ["ReLULikeHull"]

import numpy as np
from numpy import ndarray

from ._relulike import ReLULikeHull
from .._functions import *


class SiLUHull(ReLULikeHull):
    """
    This is to calculate the function hull for the sigmoid linear unit (SiLU)

    .. warning::
        This class is not implemented yet.

    """

    def __int__(self):
        raise NotImplementedError

    @classmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (_, 1+2*d)
        raise NotImplementedError

    @classmethod
    def _construct_dlp(
        cls, idx: int, dim: int, l: float, u: float
    ) -> tuple[ndarray, float]:  # (2, 1+{dim}+{idx}+1)
        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)
        kp1 = cls._f(l) / l
        kp2 = 1.0
        aux_lines = np.asarray(
            [
                [0.0] + temp1 + [kp1] + temp2 + [-1.0],
                [0.0] + temp1 + [kp2] + temp2 + [-1.0],
            ],
            dtype=np.float64,
        )
        aux_point = 0.0
        return aux_lines, aux_point

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return silu_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dsilu_np(x)
