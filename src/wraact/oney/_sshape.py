__docformat__ = "restructuredtext"
__all__ = ["SShapeHullWithOneY"]

from abc import ABC
from typing import Literal

import numpy as np
from numpy import ndarray

from wraact.acthull import SShapeHull
from wraact.oney._act import ActHullWithOneY


class SShapeHullWithOneY(ActHullWithOneY, SShapeHull, ABC):
    """
    The base class for the S-shape activation functions to calculate the function hull with only one output dimension.

    Please refer to the :class:`ActHullWithOneY` and :class:`SShapeHull` for more details.
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

        c_mn = self.cal_mn_constrs(c, v, lb, ub, self._n_output_constrs)

        return c_mn, dtype_cdd

    def cal_mn_constrs(  # type: ignore[override]
        self,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        if lb is None and ub is None:
            raise ValueError(
                "The lower and upper bounds should be provided for the S-shape activation function."
            )

        d = c.shape[1] - 1

        # The single-neuron constraints
        cc_s = np.empty((0, 1 + d), dtype=np.float64)
        # The multi-neuron constraints providing lower/upper output bounds
        cc_ml, cc_mu = c, c.copy()

        vl, vu = v, v.copy()

        # Type assertion: l and u are expected to be ndarrays if this code path is reached
        lb_arr: ndarray = lb  # type: ignore[assignment]
        ub_arr: ndarray = ub  # type: ignore[assignment]
        f, df = self._f, self._df
        xl, xu = lb_arr[0], ub_arr[0]
        yl, yu, kl, ku = f(xl), f(xu), df(xl), df(xu)
        with np.errstate(divide="ignore", invalid="ignore"):
            klu = (yu - yl) / (xu - xl)

        args = (d, xl, xu, yl, yu, kl, ku, klu, cc_s)
        dlp_line_l, dlp_line_u, dlp_point_l, dlp_point_u, cc_s = self._construct_dlp(
            0, *args, self._add_sn_constrs
        )

        cc_ml, vl = self._cal_mn_constrs_with_one_y(
            0, cc_ml, vl, dlp_line_l, dlp_point_l, is_convex=False
        )
        cc_mu, vu = self._cal_mn_constrs_with_one_y(
            0, cc_mu, vu, dlp_line_u, dlp_point_u, is_convex=True
        )

        # Fill c_mn with c_sn if constraints number is smaller than n_output_constrs
        cc_mu = self._get_topk_constrs(cc_mu, n_output_constrs, is_min=True)
        cc_ml = self._get_topk_constrs(cc_ml, n_output_constrs, is_min=False)

        if cc_mu.shape[0] < n_output_constrs:
            cc_su = cc_s[cc_s[:, -1] > 0]
            n_fill = n_output_constrs - cc_mu.shape[0]
            temp = np.tile(cc_su, (n_fill // cc_mu.shape[0], 1))
            cc_mu = np.vstack((cc_mu, temp))

        if cc_ml.shape[0] < n_output_constrs:
            cc_sl = cc_s[cc_s[:, -1] < 0]
            n_fill = n_output_constrs - cc_ml.shape[0]
            temp = np.tile(cc_sl, (n_fill // cc_ml.shape[0], 1))
            cc_ml = np.vstack((cc_ml, temp))

        cc = np.vstack((cc_mu, cc_ml))

        return cc
