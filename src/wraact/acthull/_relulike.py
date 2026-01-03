__docformat__ = "restructuredtext"
__all__ = ["ReLULikeHull"]

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy import ndarray

from wraact.acthull._act import ActHull
from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp


class ReLULikeHull(ActHull, ABC):
    """This is the base class for the ReLU-like activation functions to calculate the function hull."""

    def cal_constrs(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        lb: ndarray | None,  # (d-1,)
        ub: ndarray | None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, 2*d-1)
        d = c.shape[1] - 1
        c = np.array(c, dtype=np.float64)
        lb = np.array(lb, dtype=np.float64)
        ub = np.array(ub, dtype=np.float64)
        cc = np.empty((0, 1 + 2 * d), dtype=np.float64)

        if self._add_sn_constrs:
            c1 = self.cal_sn_constrs(lb, ub)
            cc = np.vstack((cc, c1))

        if self._add_mn_constrs:
            c2 = self.cal_mn_constrs(c, v, lb, ub)
            cc = np.vstack((cc, c2))

        return cc, dtype_cdd

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        lb: ndarray | None,  # (d-1,)
        ub: ndarray | None,  # (d-1,)
    ) -> ndarray:  # (_, 2*d-1)
        d = c.shape[1] - 1
        # Type assertion: lb and ub are expected to be ndarrays if this code path is reached
        lb_arr: ndarray = lb  # type: ignore[assignment]
        ub_arr: ndarray = ub  # type: ignore[assignment]

        for i in range(d):
            lines, point = cls._construct_dlp(i, d, lb_arr[i], ub_arr[i])
            c, v = cls._cal_mn_constrs_with_one_y(i, c, v, lines, point, is_convex=True)

        return c

    @classmethod
    def _cal_mn_constrs_with_one_y(
        cls,
        idx: int,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        dlp_lines: ndarray,  # (2, d+1)
        dlp_point: float,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:  # (n, d+1) , (m, d+1)
        return cal_mn_constrs_with_one_y_dlp(idx, c, v, dlp_lines, dlp_point, is_convex=is_convex)

    @classmethod
    @abstractmethod
    def _construct_dlp(cls, *args, **kwargs):
        pass
