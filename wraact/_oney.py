__docformat__ = "restructuredtext"
__all__ = [
    "ActHullWithOneY",
    "ReLULikeHullWithOneY",
    "ReLUHullWithOneY",
    "LeakyReLUHullWithOneY",
    "ELUHullWithOneY",
    "MaxPoolHullDLPWithOneY",
    "MaxPoolHullWithOneY",
    "SShapeHullWithOneY",
    "SigmoidHullWithOneY",
    "TanhHullWithOneY",
]

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy import ndarray

from ._acthull import *
from ._exceptions import *

_TOL = 1e-4
_MIN_BOUNDS_RANGE = 0.04


class ActHullWithOneY(ActHull, ABC):
    """
    An object used to calculate the convex hull of the activation
    function with only extending one output dimension.

    We only need several output constraints which have big beta values, which provide
    the important multi-neuron constraints.

    :param dtype_cdd: The data type used in pycddlib library.
    :param n_output_constraints: The number of output constraints.
    """

    __slots__ = ActHull.__slots__ + ["_n_output_constrs"]

    def __init__(
        self,
        dtype_cdd: Literal["fraction", "float"] = "float",
        n_output_constraints: int = 1,
        if_return_input_bounds_by_vertices: bool = False,
    ):
        super().__init__(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=True,
            if_use_double_orders=False,
            if_return_input_bounds_by_vertices=if_return_input_bounds_by_vertices,
            dtype_cdd=dtype_cdd,
        )

        self._n_output_constrs = n_output_constraints

    def _cal_hull_with_mn_constrs(
        self,
        c: ndarray,  # (n, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
    ) -> (
        ndarray  # (_, 2*d-1) | (_, d+1)
        | None
        | tuple[
            ndarray | None,  # (_, 2*d-1) | (_, d+1)
            ndarray,  # (d-1,)
            ndarray,  # (d-1,)
        ]
    ):
        if c is None:
            raise ValueError("The input constraints should be provided.")

        try:
            """
            The bounds need update if we use update scalar bounds per layer of
            DeepPoly. This will cause degenerated input polytope.

            There are two cases:
            (1) One of the input dimension has the same lower and upper bounds, which
            will throw a Degenerated exception.
            (2) The number of vertices is fewer than the dimension, which will call
            a Degenerated exception.

            We will first recalculate the vertices with the fractional number if there
            is an exception. If there is still an exception, we will accept the
            degenerated input polytope.
            """
            v, dtype_cdd = self._cal_vertices_with_exception(c, l, u, self.dtype_cdd)
            new_l = np.min(v, axis=0)[1:]
            new_u = np.max(v, axis=0)[1:]
            self._check_degenerated_input_polytope(v, new_l, new_u)
            l = new_l
            u = new_u
        except Degenerated:
            v, dtype_cdd = self.cal_vertices(c, "fraction")
            l = np.min(v, axis=0)[1:]
            u = np.max(v, axis=0)[1:]
        except Exception as e:
            raise e

        # Update input bounds constraints
        d = l.shape[0]
        c[-2 * d : -d, 0] = -l  # noqa: E203
        c[-d:, 0] = u

        cc, dtype_cdd = self._cal_constrs_with_exception(c, v, l, u, dtype_cdd)

        # ====================CHECK====================
        # Check if all vertices satisfy the constraints.
        # v_y = self._f(v[:, 1:])
        # vertices = np.hstack((v, v_y))
        # check = np.matmul(cc, vertices.T)
        # if not np.all(check >= -_TOL):
        #     raise RuntimeError("Not all vertices satisfy the constraints.")

        if self._use_double_orders:
            # Here we reverse the order of output dimensions to calculate the function
            # hull because our algorithm is a progressive algorithm that calculates the
            # function hull of the output dimensions one by one.
            o_r = ActHull._get_reversed_order(cc.shape[1] - 1)
            c_r = c.copy()  # Reversed constraints
            c_r = c_r[:, o_r]
            cc_r, dtype_cdd = self._cal_constrs_with_exception(c_r, v, l, u, dtype_cdd)
            cc_r = cc_r[:, o_r]
            cc = np.vstack((cc, cc_r))

        return cc

    @classmethod
    @abstractmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        pass

    @staticmethod
    def _get_topk_constrs(
        c: ndarray,  # (_, d)
        topk: int,
        is_min: bool = True,
    ) -> ndarray:  # (_, d)
        # Choose the constraints with non-zero beta values, which is the last column
        # of the constraints.
        c = c[(c[:, -1] < -_TOL) | (c[:, -1] > _TOL)]

        # c = c[np.argsort(-np.abs(c[:, 0] / c[:, -1]))]
        c = c[np.argsort(c[:, -1])]

        # Get the topk maximum or minimum beta values.
        if is_min:
            c = c[:topk]
        else:
            c = c[-topk:]

        return c


class ReLULikeHullWithOneY(ActHullWithOneY, ReLULikeHull, ABC):
    """
    The base class for the ReLU like activation functions to calculate the function hull
    with only one output dimension.

    Please refer to the :class:`ActHullWithOneY` and :class:`ReLULikeHull` for more
    details.
    """

    def cal_constrs(
        self,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, d+1)
        c = np.array(c, dtype=np.float64)

        if np.min(np.abs(u - l)) < _MIN_BOUNDS_RANGE and len(v) > 2:
            # The input polytope is too small, and we only return the single-neuron
            # constraints.
            # We do not want to remove the trivial cases of MaxPool function (one vertex
            # and one piece).
            c_m = np.empty((0, c.shape[1] + 1), dtype=np.float64)
        else:
            c_m = self.cal_mn_constrs(c, v, l, u, self._n_output_constrs)

        # Fill c_m with c_s if constraints number is smaller than n_output_constrs
        if c_m.shape[0] < self._n_output_constrs:
            c_s = self.cal_sn_constrs(l, u)
            c_su = c_s[c_s[:, -1] < 0]
            n_fill = self._n_output_constrs - c_m.shape[0]
            # Repeat c_s to fill the rest with the constraints
            temp = np.tile(c_su, (n_fill // c_su.shape[0], 1))
            c_m = np.vstack((c_m, temp))

        return c_m, dtype_cdd

    @classmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (_, d+2)

        dim = l.shape[0]
        c = np.zeros((1, dim + 2), dtype=np.float64)

        # Here, we only need one upper bound of the ReLU-like function for the first
        # dimension. There is only one upper constraints because it is a convex
        # function.
        l0, u0 = l[0], u[0]
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
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        d = c.shape[1] - 1

        aux_lines, aux_point = cls._construct_dlp(0, d, l[0], u[0])
        c, v = cls._cal_mn_constrs_with_one_y(0, c, v, aux_lines, aux_point, True)
        c = cls._get_topk_constrs(c, n_output_constrs)

        return c


class ReLUHullWithOneY(ReLULikeHullWithOneY, ReLUHull):
    """
    The class to calculate the function hull for the rectified linear unit (ReLU)
    activation function with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`ReLUHull` for more
    details.

    """

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        v = np.transpose(v)
        mask_xp, mask_xn = (v > _TOL), (v < -_TOL)
        if not np.any(mask_xp) and not np.any(mask_xn):
            raise ArithmeticError(
                "The vertices should not all positive or all negative."
            )

        cv = np.matmul(c, v)

        s = (c.shape[0], 1)
        beta1 = np.zeros(s, dtype=np.float64)
        beta2 = np.zeros(s, dtype=np.float64)

        xi_p, xi_n = mask_xp[1], mask_xn[1]

        if np.any(xi_p):
            temp = cv[:, xi_p] / v[1, xi_p]
            beta1[:, 0] = -np.min(temp, axis=1)

        if np.any(xi_n):
            temp = cv[:, xi_n] / v[1, xi_n]
            beta2[:, 0] = np.max(temp, axis=1)

        # Eliminate tiny positive values
        beta1[beta1 > 0] = 0.0
        beta2[beta2 > 0] = 0.0

        c = np.hstack((c, beta1 + beta2))
        c[:, [1]] -= beta2

        c = cls._get_topk_constrs(c, n_output_constrs)

        return c


class LeakyReLUHullWithOneY(ReLULikeHullWithOneY, LeakyReLUHull):
    """
    The class to calculate the function hull for the leaky rectified linear unit (Leaky
    ReLU) activation function with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`LeakyReLUHull` for
    more details.
    """

    pass


class ELUHullWithOneY(ReLULikeHullWithOneY, ELUHull):
    """
    The class to calculate the function hull for the exponential linear unit (ELU)
    activation function with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`ELUHull` for more
    details
    """

    pass


class MaxPoolHullDLPWithOneY(ReLULikeHullWithOneY, MaxPoolHullDLP):
    """
    The class to calculate the function hull for the max pooling layer with only one
    output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`MaxPoolHullDLP` for
    more details.
    """

    def cal_constrs(
        self,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, d+1)
        return ReLULikeHullWithOneY.cal_constrs(self, c, v, l, u, dtype_cdd)

    @classmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (1, d+2)

        d = l.shape[0]

        # Upper bounds
        # Reference: Formal Verification of Piece-Wise Linear Feed-Forward Neural
        # Networks, https://arxiv.org/pdf/1705.01320
        # y <= sum(x_i - l_i) + l_max
        c_u = np.zeros((1, d + 2), dtype=np.float64)

        # Here, we do not use Ehler's method because it is not very precise.
        # Just use the maximum value of upper bounds.
        # l_sum = np.sum(l)
        # l_max = np.max(l)
        # c_u[-1, 0] = l_max - l_sum
        c_u[-1, 0] = np.max(u)
        c_u[-1, -1] = -1.0

        return c_u

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        c = MaxPoolHullDLP.cal_mn_constrs(c, v, l, u)
        c = cls._get_topk_constrs(c, n_output_constrs)
        return c


class MaxPoolHullWithOneY(MaxPoolHullDLPWithOneY, MaxPoolHull):
    """
    The class to calculate the function hull for the max pooling layer with only one
    output dimension.

    Please refer to the :class:`MaxPoolHullDLPWithOneY` and :class:`MaxPoolHull` for
    more details
    """

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        c = MaxPoolHull.cal_mn_constrs(c, v, l, u)
        c = cls._get_topk_constrs(c, n_output_constrs)
        return c


class SShapeHullWithOneY(ActHullWithOneY, SShapeHull, ABC):
    """
    The base class for the S-shape activation functions to calculate the function hull
    with only one output dimension.

    Please refer to the :class:`ActHullWithOneY` and :class:`SShapeHull` for more
    details.
    """

    def cal_constrs(
        self,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, d+1)
        raise NotImplementedError
        # c = np.array(c, dtype=np.float64)
        #
        # c_mn = self.cal_mn_constrs(c, v, l, u, self._n_output_constrs)
        #
        # return c_mn, dtype_cdd

    def cal_mn_constrs(
        self,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        n_output_constrs: int = 1,
    ) -> ndarray:  # (_, d+1)
        if l is None and u is None:
            raise ValueError(
                "The lower and upper bounds should be provided for the S-shape "
                "activation function."
            )

        d = c.shape[1] - 1

        # The single-neuron constraints
        cc_s = np.empty((0, 1 + d), dtype=np.float64)
        # The multi-neuron constraints providing lower/upper output bounds
        cc_ml, cc_mu = c, c.copy()

        vl, vu = v, v.copy()

        f, df = self._f, self._df
        xl, xu = l[0], u[0]
        yl, yu, kl, ku = f(xl), f(xu), df(xl), df(xu)
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
        cc_mu = self._get_topk_constrs(cc_mu, n_output_constrs, True)
        cc_ml = self._get_topk_constrs(cc_ml, n_output_constrs, False)

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


class SigmoidHullWithOneY(SShapeHullWithOneY, SigmoidHull):
    """
    The class used to calculate the convex hull of the sigmoid activation function with
    only extending one output dimension.

    Please refer to the :class:`SShapeHullWithOneY` and :class:`SigmoidHull` for more
    details.
    """

    pass


class TanhHullWithOneY(SShapeHullWithOneY, TanhHull):
    """
    The class used to calculate the convex hull of the hyperbolic tangent (tanh)
    activation function with only extending one output dimension.

    Please refer to the :class:`SShapeHullWithOneY` and :class:`TanhHull` for more
    details.
    """

    pass
