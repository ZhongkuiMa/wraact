"""Base class for ReLU-like activation hull computation.

This module provides the base class for computing convex hulls of piecewise
linear activation functions like ReLU, LeakyReLU, and ELU.
"""

__docformat__ = "restructuredtext"
__all__ = ["ReLULikeHull"]

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy import ndarray

from wraact.acthull._act import ActHull
from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp


class ReLULikeHull(ActHull, ABC):
    """Base class for ReLU-like activation function hull computation.

    ReLU-like functions are piecewise linear with a kink at zero. This class
    provides methods for computing tight convex hull constraints using
    Double Linear Programming (DLP) techniques.
    """

    def cal_constrs(
        self,
        c: ndarray,
        v: ndarray,
        lb: ndarray | None,
        ub: ndarray | None,
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:
        """Compute hull constraints combining single and multi-neuron constraints.

        :param c: Input constraints in H-representation. Shape: ``n, d``.
        :param v: Vertices of input polytope. Shape: ``m, d``.
        :param lb: Lower bounds per dimension. Shape: ``d-1,``.
        :param ub: Upper bounds per dimension. Shape: ``d-1,``.
        :param dtype_cdd: Data type for CDD library. Default: "float".
        :return: Tuple of (constraints, dtype) where constraints has
            shape (_, 2*d-1).
        """
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
        c: ndarray,
        v: ndarray,
        lb: ndarray | None,
        ub: ndarray | None,
    ) -> ndarray:
        """Compute multi-neuron constraints using DLP.

        Iteratively applies DLP to each output dimension to generate
        tight multi-neuron constraints.

        :param c: Input constraints in H-representation. Shape: ``n, d``.
        :param v: Vertices of input polytope. Shape: ``m, d``.
        :param lb: Lower bounds per dimension. Shape: ``d-1,``.
        :param ub: Upper bounds per dimension. Shape: ``d-1,``.
        :return: Multi-neuron constraints. Shape: ``_, 2*d-1``.
        """
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
        c: ndarray,
        v: ndarray,
        dlp_lines: ndarray,
        dlp_point: float,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:
        """Compute multi-neuron constraints for one output dimension.

        :param idx: Index of the output dimension to process.
        :param c: Current constraints. Shape: ``n, d``.
        :param v: Current vertices. Shape: ``m, d``.
        :param dlp_lines: DLP line parameters. Shape: ``2, d+1``.
        :param dlp_point: DLP auxiliary point.
        :param is_convex: True if activation is convex in this region.
        :return: Tuple of (updated_constraints, updated_vertices).
        """
        return cal_mn_constrs_with_one_y_dlp(idx, c, v, dlp_lines, dlp_point, is_convex=is_convex)

    @classmethod
    @abstractmethod
    def _construct_dlp(cls, *args, **kwargs):
        pass
