__docformat__ = "restructuredtext"
__all__ = ["ReLULikeHullWithOneY"]

from abc import ABC
from typing import Literal

import numpy as np
from numpy import ndarray

from wraact._constants import MIN_BOUNDS_RANGE_ONEY
from wraact.acthull import ReLULikeHull
from wraact.oney._act import ActHullWithOneY


class ReLULikeHullWithOneY(ActHullWithOneY, ReLULikeHull, ABC):
    """
    The base class for the ReLU like activation functions to calculate the function hull with only one output dimension.

    Please refer to the :class:`ActHullWithOneY` and :class:`ReLULikeHull` for more details.
    """

    def cal_constrs(
        self,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, d+1)
        c = np.array(c, dtype=np.float64)

        # Type narrowing for bounds
        lb_arr: ndarray = lb  # type: ignore[assignment]
        ub_arr: ndarray = ub  # type: ignore[assignment]

        if np.min(np.abs(ub_arr - lb_arr)) < MIN_BOUNDS_RANGE_ONEY and len(v) > 2:
            # The input polytope is too small, and we only return the single-neuron
            # constraints.
            # We do not want to remove the trivial cases of MaxPool function (one vertex
            # and one piece).
            min_range = np.min(np.abs(ub_arr - lb_arr))
            raise ValueError(
                f"Polytope too small: minimum range {min_range:.6f} < "
                f"threshold {MIN_BOUNDS_RANGE_ONEY}. Cannot compute reliable constraints."
            )
        c_m = self.cal_mn_constrs(c, v, lb_arr, ub_arr, self._n_output_constrs)

        # Fill c_m with c_s if constraints number is smaller than n_output_constrs
        if c_m.shape[0] < self._n_output_constrs:
            c_s = self.cal_sn_constrs(lb_arr, ub_arr)
            c_su = c_s[c_s[:, -1] < 0]
            n_fill = self._n_output_constrs - c_m.shape[0]
            # Repeat c_s to fill the rest with the constraints
            temp = np.tile(c_su, (n_fill // c_su.shape[0], 1))
            c_m = np.vstack((c_m, temp))

        return c_m, dtype_cdd

    @classmethod
    def cal_sn_constrs(
        cls,
        lb: ndarray,  # (d,)
        ub: ndarray,  # (d,)
    ) -> ndarray:  # (_, d+2)
        dim = lb.shape[0]
        c = np.zeros((1, dim + 2), dtype=np.float64)

        # Here, we only need one upper bound of the ReLU-like function for the first
        # dimension. There is only one upper constraints because it is a convex
        # function.
        l0, u0 = lb[0], ub[0]
        yl0, yu0 = cls._f(l0), cls._f(u0)
        k = (yu0 - yl0) / (u0 - l0)
        c[0, 0] = -u0 * k + yu0
        c[0, 1] = k
        c[0, -1] = -1.0

        return c

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        d = c.shape[1] - 1

        # Type assertion: l and u are expected to be ndarrays if this code path is reached
        lb_arr: ndarray = lb  # type: ignore[assignment]
        ub_arr: ndarray = ub  # type: ignore[assignment]
        aux_lines, aux_point = cls._construct_dlp(0, d, lb_arr[0], ub_arr[0])
        c, v = cls._cal_mn_constrs_with_one_y(0, c, v, aux_lines, aux_point, is_convex=True)
        c = cls._get_topk_constrs(c, n_output_constrs)

        return c
