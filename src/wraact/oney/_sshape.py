"""Base class for single-output S-shaped activation hull computation."""

__docformat__ = "restructuredtext"
__all__ = ["SShapeHullWithOneY"]

from abc import ABC
from math import ceil
from typing import Literal

import numpy as np
from numpy import ndarray

from wraact._enums import TopKSelector
from wraact.acthull import SShapeHull
from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp
from wraact.oney._act import ActHullWithOneY


class SShapeHullWithOneY(ActHullWithOneY, SShapeHull, ABC):
    """
    The base class for the S-shape activation functions to calculate the function hull with only one output dimension.

    Please refer to the :class:`ActHullWithOneY` and :class:`SShapeHull` for more details.
    """

    def cal_constrs(
        self,
        c: ndarray,
        v: ndarray,
        lb: ndarray | None = None,
        ub: ndarray | None = None,
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:
        """Compute single-output S-shaped hull constraints.

        :param c: Input constraints. Shape: ``_, d``.
        :param v: Vertices. Shape: ``_, d``.
        :param lb: Lower bounds per dimension.
        :param ub: Upper bounds per dimension.
        :param dtype_cdd: Data type for pycddlib. Default: "float".
        :return: Tuple of (constraints, dtype_cdd).
        """
        c = np.array(c, dtype=np.float64)

        c_mn = self.cal_mn_constrs(
            c, v, lb, ub, self._n_output_constrs, topk_selector=self._topk_selector
        )

        if not np.all(np.isfinite(c_mn)):
            c_mn = self._get_one_y_output_range_constrs(c.shape[1] - 1)

        return c_mn, dtype_cdd

    @classmethod
    def _get_one_y_output_range_constrs(cls, dim: int) -> ndarray:
        """Return conservative single-output constraints from the activation codomain."""
        constraints = np.zeros((2, dim + 2), dtype=np.float64)
        constraints[0, 0] = -cls._OUTPUT_LOWER_BOUND
        constraints[0, -1] = 1.0
        constraints[1, 0] = cls._OUTPUT_UPPER_BOUND
        constraints[1, -1] = -1.0
        return constraints

    @classmethod
    def _get_one_y_interval_constrs(cls, dim: int, lb: ndarray, ub: ndarray) -> ndarray:
        """Return conservative interval constraints for the first output."""
        full = cls._get_interval_output_constrs(lb[:1], ub[:1])
        constraints = np.zeros((2, dim + 2), dtype=np.float64)
        constraints[:, 0] = full[:, 0]
        constraints[:, -1] = full[:, -1]
        return constraints

    def cal_mn_constrs(  # type: ignore[override]
        self,
        c: ndarray,
        v: ndarray,
        lb: ndarray | None = None,
        ub: ndarray | None = None,
        n_output_constrs: int = 1,
        topk_selector: TopKSelector = TopKSelector.BETA_MIN,
    ) -> ndarray:
        """Compute multi-neuron constraints for single-output S-shaped activation.

        Constructs lower and upper DLP bounds, then selects the top-k
        constraints from each side.

        :param c: Input constraints. Shape: ``_, d``.
        :param v: Vertices. Shape: ``_, d``.
        :param lb: Lower bounds per dimension.
        :param ub: Upper bounds per dimension.
        :param n_output_constrs: Number of output constraints per side.
        :return: Combined upper and lower multi-neuron constraints.
        :raises ValueError: If bounds are not provided.
        """
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
        dlp_line_l, dlp_line_u, dlp_point_l, dlp_point_u, _ = self._construct_dlp(
            0, *args, return_single_neuron_constrs=False
        )
        cc_s = self._get_one_y_interval_constrs(d, lb_arr, ub_arr)

        cc_ml, vl = cal_mn_constrs_with_one_y_dlp(
            0, cc_ml, vl, dlp_line_l, dlp_point_l, is_convex=False
        )
        cc_mu, vu = cal_mn_constrs_with_one_y_dlp(
            0, cc_mu, vu, dlp_line_u, dlp_point_u, is_convex=True
        )

        # Fill c_mn with c_sn if constraints number is smaller than n_output_constrs
        cc_mu = self._get_topk_constrs(cc_mu, n_output_constrs, is_min=True, selector=topk_selector)
        cc_ml = self._get_topk_constrs(
            cc_ml, n_output_constrs, is_min=False, selector=topk_selector
        )

        if cc_mu.shape[0] < n_output_constrs:
            cc_su = cc_s[cc_s[:, -1] > 0]
            n_fill = n_output_constrs - cc_mu.shape[0]
            if cc_su.shape[0] > 0:
                reps = ceil(n_fill / cc_su.shape[0])
                temp = np.tile(cc_su, (reps, 1))[:n_fill]
                cc_mu = np.vstack((cc_mu, temp))

        if cc_ml.shape[0] < n_output_constrs:
            cc_sl = cc_s[cc_s[:, -1] < 0]
            n_fill = n_output_constrs - cc_ml.shape[0]
            if cc_sl.shape[0] > 0:
                reps = ceil(n_fill / cc_sl.shape[0])
                temp = np.tile(cc_sl, (reps, 1))[:n_fill]
                cc_ml = np.vstack((cc_ml, temp))

        cc = np.vstack((cc_mu, cc_ml))

        if not np.all(np.isfinite(cc)):
            cc = self._get_one_y_output_range_constrs(d)

        return cc
