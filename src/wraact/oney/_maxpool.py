__docformat__ = "restructuredtext"
__all__ = ["MaxPoolHullDLPWithOneY", "MaxPoolHullWithOneY"]

from typing import Literal

import numpy as np
from numpy import ndarray

from wraact.acthull import MaxPoolHull, MaxPoolHullDLP
from wraact.oney._relulike import ReLULikeHullWithOneY


class MaxPoolHullDLPWithOneY(ReLULikeHullWithOneY, MaxPoolHullDLP):
    """
    The class to calculate the function hull for the max pooling layer with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`MaxPoolHullDLP` for more details.
    """

    def cal_constrs(
        self,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, d+1)
        return ReLULikeHullWithOneY.cal_constrs(self, c, v, lb, ub, dtype_cdd)

    @classmethod
    def cal_sn_constrs(
        cls,
        lb: ndarray,  # (d,)
        ub: ndarray,  # (d,)
    ) -> ndarray:  # (1, d+2)
        d = lb.shape[0]

        # Upper bounds
        # Reference: Formal Verification of Piece-Wise Linear Feed-Forward Neural
        # Networks, https://arxiv.org/pdf/1705.01320
        # y <= sum(x_i - l_i) + l_max
        c_u = np.zeros((1, d + 2), dtype=np.float64)

        # Here, we do not use Ehler's method because it is not very precise.
        # Just use the maximum value of upper bounds.
        # l_sum = np.sum(lb)
        # l_max = np.max(lb)
        # c_u[-1, 0] = l_max - l_sum
        c_u[-1, 0] = np.max(ub)
        c_u[-1, -1] = -1.0

        return c_u

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        c = MaxPoolHullDLP.cal_mn_constrs(c, v, lb, ub)
        c = cls._get_topk_constrs(c, n_output_constrs)
        return c


class MaxPoolHullWithOneY(MaxPoolHullDLPWithOneY, MaxPoolHull):
    """
    The class to calculate the function hull for the max pooling layer with only one output dimension.

    Please refer to the :class:`MaxPoolHullDLPWithOneY` and :class:`MaxPoolHull` for more details.
    """

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        c = MaxPoolHull.cal_mn_constrs(c, v, lb, ub)
        c = cls._get_topk_constrs(c, n_output_constrs)
        return c
