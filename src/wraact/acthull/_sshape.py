__docformat__ = "restructuredtext"
__all__ = ["SShapeHull"]

from abc import ABC, abstractmethod
from typing import Literal, cast

import numpy as np
from numpy import ndarray

from wraact._constants import MIN_DLP_ANGLE
from wraact.acthull._act import ActHull
from wraact.acthull._utils import cal_mn_constrs_with_one_y_dlp


class SShapeHull(ActHull, ABC):
    """
    This is the base class for the S-shaped activation functions to calculate the function hull.

    The S-shaped activation functions include the sigmoid, hyperbolic tangent, etc.

    .. tip::
        Overall, to calculate the function hull of the S-shaped activation functions, we
        construct two *double-linear-piece* (DLP) functions as the upper and lower
        bounds of the activation function. We take the upper constraints of the upper
        DLP function and the lower constraints of the lower DLP function as the
        multi-neuron constraints.

    .. tip::
        The constraints construction of the S-shaped activation functions is based on
        some tangent lines of the activation function. The tangent lines are calculated
        in an iterative way, resulting it is slower than the ReLU-like activation
        functions.

        Refer to the paper:
        `Efficient Neural Network Verification via Adaptive Refinement and Adversarial
        Search
        <https://ecai2020.eu/papers/384_paper.pdf>`__
        :cite:`henriksen_efficient_2020`
        for numerically calculating the tangent lines of Sigmoid and Tanh functions.
    """

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
        cc = np.empty((0, 2 * d + 1), dtype=np.float64)

        if self._add_sn_constrs and not self._add_mn_constrs:
            c_s = self.cal_sn_constrs(lb, ub)
            cc = np.vstack((cc, c_s))

        if self._add_mn_constrs:
            c_m = self.cal_mn_constrs(c, v, lb, ub)
            cc = np.vstack((cc, c_m))

        return cc, dtype_cdd

    def cal_sn_constrs(  # type: ignore[override]
        self,
        lb: ndarray,  # (d,)
        ub: ndarray,  # (d,)
    ) -> ndarray:  # (_, 1+2*d)
        d = lb.shape[0]
        cc = np.empty((0, 1 + d), dtype=np.float64)

        f, df = self._f, self._df
        xl, xu = lb, ub
        yl: ndarray
        yu: ndarray
        kl: ndarray
        ku: ndarray
        yl, yu, kl, ku = f(xl), f(xu), df(xl), df(xu)  # type: ignore[assignment]
        with np.errstate(divide="ignore", invalid="ignore"):
            klu = (yu - yl) / (xu - xl)

        for i in range(d):
            args = (i, d, xl[i], xu[i], yl[i], yu[i], kl[i], ku[i], klu[i], cc)
            _, _, _, _, cc = self._construct_dlp(*args, self._add_sn_constrs)

        return cc

    def cal_mn_constrs(  # type: ignore[override]
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        lb: ndarray | None,  # (d-1,)
        ub: ndarray | None,  # (d-1,)
    ) -> ndarray:  # (_, 2*d-1) | (_, d+1)
        if lb is None or ub is None:
            raise ValueError(
                "Both lower and upper bounds are required for the S-shape activation function."
            )

        d = c.shape[1] - 1
        # The single-neuron constraints
        cc_s = np.empty((0, 1 + d), dtype=np.float64)
        # The multi-neuron constraints providing lower/upper output bounds
        cc_l, cc_u = c, c.copy()
        v_l, v_u = v, v.copy()

        f, df = self._f, self._df
        xl, xu = lb, ub
        yl: ndarray
        yu: ndarray
        kl: ndarray
        ku: ndarray
        yl, yu, kl, ku = f(xl), f(xu), df(xl), df(xu)  # type: ignore[assignment]
        with np.errstate(divide="ignore", invalid="ignore"):
            klu = (yu - yl) / (xu - xl)

        for i in range(d):
            args = (i, d, xl[i], xu[i], yl[i], yu[i], kl[i], ku[i], klu[i], cc_s)
            dlp_lines_l, dlp_lines_u, dlp_point_l, dlp_point_u, cc_s = self._construct_dlp(
                *args, self._add_sn_constrs
            )

            if self._add_mn_constrs:
                if dlp_lines_l is not None:
                    cc_l, v_l = self._cal_mn_constrs_with_one_y(
                        i, cc_l, v_l, dlp_lines_l, dlp_point_l, is_convex=False
                    )
                if dlp_lines_u is not None:
                    cc_u, v_u = self._cal_mn_constrs_with_one_y(
                        i, cc_u, v_u, dlp_lines_u, dlp_point_u, is_convex=True
                    )

        cc = np.empty((0, 2 * d + 1), dtype=np.float64)

        if self._add_sn_constrs:
            cc = np.vstack((cc, cc_s))

        if self._add_mn_constrs:
            cc = np.vstack((cc, cc_l, cc_u))

        return cc

    @classmethod
    def _cal_mn_constrs_with_one_y(
        cls,
        idx: int,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        dlp_lines: ndarray,  # (2, d+1) | (1, d+1)
        dlp_point: float | ndarray | None,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:  # (n, d+1) | (n+1, d+1), (m, d+1)
        return cast(
            tuple[ndarray, ndarray],
            cal_mn_constrs_with_one_y_dlp(idx, c, v, dlp_lines, dlp_point, is_convex=is_convex),
        )

    @classmethod
    def _construct_dlp(
        cls,
        idx: int,
        dim: int,
        xli: float | ndarray,
        xui: float | ndarray,
        yli: float | ndarray,
        yui: float | ndarray,
        kli: float | ndarray,
        kui: float | ndarray,
        klui: float | ndarray,
        c: ndarray,  # (n, d)
        return_single_neuron_constrs: bool,
    ) -> tuple[ndarray, ndarray, float | ndarray | None, float | ndarray | None, ndarray]:
        """
        Calculate the auxiliary lines, auxiliary point, and the single-neuron constraints.

        There are three cases:

        1. One linear function as the lower bound and one DLP function as the upper
           bound.
        2. One DLP function as the lower bound and one linear function as the upper
           bound.
        3. Two DLP functions as the lower and upper bounds.

        :param idx: The index of dimension to extend in output space.
        :param dim: The dimension of input space.
        :param xli: The lower bound of the input variable.
        :param xui: The upper bound of the input variable.
        :param yli: The lower bound of the output variable.
        :param yui: The upper bound of the output variable.
        :param kli: The slope of the lower bound of the input variable.
        :param kui: The slope of the upper bound of the input variable.
        :param klui: The slope of the linear piece connecting the lower and upper
            bounds.
        :param c: The single-neuron constraints of the output polytope.
        :param return_single_neuron_constrs: Whether to add the single-neuron
            constraints.

        :return: The auxiliary lines, auxiliary point, and the single-neuron
            constraints.
        """
        if np.allclose(xli, xui):
            # Handle the degenerate case
            c1 = np.zeros((1, idx + dim + 2), dtype=np.float64)
            c2 = np.zeros((1, idx + dim + 2), dtype=np.float64)
            c1[:, 0] = [-kli * xli + yli]
            c1[:, idx + 1] = [kli]
            c1[:, -1] = -1.0
            c2[:, 0] = [-kui * xli + yli]
            c2[:, idx + 1] = [kui]
            c2[:, -1] = -1.0
            c = np.hstack((c, np.zeros((c.shape[0], 1))))
            return c1, c2, None, None, c
        if kui > klui:
            resolve_case = cls._construct_dlp_case1  # type: ignore[assignment]
        elif kli > klui:
            resolve_case = cls._construct_dlp_case2  # type: ignore[assignment]
        else:
            resolve_case = cls._construct_dlp_case3  # type: ignore[assignment]

        c = np.hstack((c, np.zeros((c.shape[0], 1))))
        args = (idx, dim, xli, xui, yli, yui, kli, kui, klui, c)
        return resolve_case(*args, return_single_neuron_constrs)

    @classmethod
    def _construct_dlp_case1(
        cls,
        idx: int,
        dim: int,
        xli: float | ndarray,
        xui: float | ndarray,
        yli: float | ndarray,
        yui: float | ndarray,
        kli: float | ndarray,
        kui: float | ndarray,
        klui: float | ndarray,
        c: ndarray,  # (n, d)
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,  # (1, 1+dim+idx+1)
        ndarray,  # (2, 1+dim+idx+1)
        None,
        float | ndarray | None,
        ndarray,  # (n+4, 1+dim+idx+1)
    ]:
        """
        Calculate the auxiliary lines, auxiliary point, and the single-neuron constraints for the case where the slope of the upper linear piece is larger than the slope of the linear piece connecting the lower and upper bounds.

        :param idx: The index of dimension to extend in output space.
        :param dim: The dimension of input space.
        :param xli: The lower bound of the input variable.
        :param xui: The upper bound of the input variable.
        :param yli: The lower bound of the output variable.
        :param yui: The upper bound of the output variable.
        :param kli: The slope of the lower bound of the input variable.
        :param kui: The slope of the upper bound of the input variable.
        :param klui: The slope of the linear piece connecting the lower and upper
            bounds.
        :param: c: The single-neuron constraints of the output polytope.
        :param return_single_neuron_constrs: Whether to return the single-neuron
            constraints.

        :return: The auxiliary lines, auxiliary point, and the single-neuron
            constraints.
        """
        f = cls._f
        blu2, klu2, su = cls._get_parallel_tangent_line(klui, get_big=False)
        kp1, kli = (yli - f(su)) / (xli - su), kli
        bp1, bli = yli - kp1 * xli, yli - kli * xli

        if xui > 0:
            x = np.asarray([xli, su, xui], dtype=np.float64)
            b, k, _ = cls._get_second_tangent_line(x, get_big=True)
            blui, bp2, bui = b  # type: ignore[misc]
            klui, kp2, kui = k  # type: ignore[misc]
        else:
            kp2, klui, kui = (yui - f(su)) / (xui - su), klui, kui
            bp2, blui, bui = yui - kp2 * xui, yli - klui * xli, yui - kui * xui

        aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_l[:, 0] = [bli]
        aux_lines_l[:, idx + 1] = [kli]
        aux_lines_l[:, -1] = -1.0
        aux_point_l = None

        aux_lines_u, aux_point_u = cls._construct_upper_aux_lines_and_points(  # type: ignore[arg-type]
            dim, idx, xli, yli, klui, su, kp1, kp2, bp1, bp2, blu2
        )

        if return_single_neuron_constrs:
            temp = np.zeros((4, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [blui, -bli, -bui, -blu2]
            temp[:, idx + 1] = [klui, -kli, -kui, -klu2]
            temp[:, -1] = [-1.0, 1.0, 1.0, 1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    @classmethod
    def _construct_dlp_case2(
        cls,
        idx: int,
        dim: int,
        xli: float | ndarray,
        xui: float | ndarray,
        yli: float | ndarray,
        yui: float | ndarray,
        kli: float | ndarray,
        kui: float | ndarray,
        klui: float | ndarray,
        c: ndarray,  # (n, d)
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,  # (2, 1+dim+idx+1)
        ndarray,  # (1, 1+dim+idx+1)
        float | ndarray | None,
        None,
        ndarray,  # (n+4, 1+dim+idx+1)
    ]:
        """
        Calculate the auxiliary lines, auxiliary point, and the single-neuron constraints for the case where the slope of the lower linear piece is larger than the slope of the linear piece connecting the lower and upper bounds.

        :param idx: The index of dimension to extend in output space.
        :param dim: The dimension of input space.
        :param xli: The lower bound of the input variable.
        :param xui: The upper bound of the input variable.
        :param yli: The lower bound of the output variable.
        :param yui: The upper bound of the output variable.
        :param kli: The slope of the lower bound of the input variable.
        :param kui: The slope of the upper bound of the input variable.
        :param klui: The slope of the linear piece connecting the lower and upper
            bounds.
        :param c: The single-neuron constraints of the output polytope.
        :param return_single_neuron_constrs: Whether to add the single-neuron
            constraints.

        :return: The auxiliary lines, auxiliary point, and the single-neuron
            constraints.
        """
        f = cls._f
        blu2, klu2, sl = cls._get_parallel_tangent_line(klui, get_big=True)
        kp1, kui = (yui - f(sl)) / (xui - sl), kui
        bp1, bui = yui - kp1 * xui, yui - kui * xui

        if xli < 0:
            x = np.asarray([xui, sl, xli], dtype=np.float64)
            b, k, _ = cls._get_second_tangent_line(x, get_big=False)
            blui, bp2, bli = b  # type: ignore[misc]
            klui, kp2, kli = k  # type: ignore[misc]
        else:
            kp2, klui, kli = (yli - f(sl)) / (xli - sl), klui, kli
            bp2, blui, bli = yli - kp2 * xli, yui - klui * xui, yli - kli * xli

        aux_lines_l, aux_point_l = cls._construct_lower_aux_lines_and_points(  # type: ignore[arg-type]
            dim, idx, xui, yui, klui, sl, kp1, kp2, bp1, bp2, blu2
        )

        aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_u[:, 0] = [bui]
        aux_lines_u[:, idx + 1] = [kui]
        aux_lines_u[:, -1] = -1.0
        aux_point_u = None

        if return_single_neuron_constrs:
            temp = np.zeros((4, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [blui, -bli, -bui, -blu2]
            temp[:, idx + 1] = [klui, -kli, -kui, -klu2]
            temp[:, -1] = [1.0, -1.0, -1.0, -1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    @classmethod
    def _construct_dlp_case3(
        cls,
        idx: int,
        dim: int,
        xli: float | ndarray,
        xui: float | ndarray,
        yli: float | ndarray,
        yui: float | ndarray,
        kli: float | ndarray,
        kui: float | ndarray,
        klui: float | ndarray,
        c: ndarray,  # (n, d)
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,  # (2, 1+dim+idx+1)
        ndarray,  # (2, 1+dim+idx+1)
        float | ndarray | None,
        float | ndarray | None,
        ndarray,  # (n+6, 1+dim+idx+1)
    ]:
        """
        Calculate the auxiliary lines, auxiliary point, and the single-neuron constraints for the case where (1) the slope of the upper linear piece is smaller than the slope of the linear piece connecting the lower and upper bounds, and (2) the slope of the lower linear piece is smaller than the slope of the linear piece connecting the lower and upper bounds.

        :param idx: The index of dimension to extend in output space.
        :param dim: The dimension of input space.
        :param xli: The lower bound of the input variable.
        :param xui: The upper bound of the input variable.
        :param yli: The lower bound of the output variable.
        :param yui: The upper bound of the output variable.
        :param kli: The slope of the lower bound of the input variable.
        :param kui: The slope of the upper bound of the input variable.
        :param klui: The slope of the linear piece connecting the lower and upper
            bounds.
        :param c: The single-neuron constraints of the output polytope.
        :param return_single_neuron_constrs: Whether to add the single-neuron
            constraints.

        :return: The auxiliary lines, auxiliary point, and the single-neuron
            constraints.
        """
        f = cls._f
        blul, klul, su = cls._get_parallel_tangent_line(klui, get_big=False)
        bluu, kluu, sl = cls._get_parallel_tangent_line(klui, get_big=True)

        x_temp = np.asarray([xli, su], dtype=np.float64)
        b_temp, k_temp, _ = cls._get_second_tangent_line(x_temp, get_big=True)
        btu, bp2u = b_temp  # type: ignore[misc]
        ktu, kp2u = k_temp  # type: ignore[misc]
        x_temp = np.asarray([xui, sl], dtype=np.float64)
        b_temp, k_temp, _ = cls._get_second_tangent_line(x_temp, get_big=False)
        btl, bp2l = b_temp  # type: ignore[misc]
        ktl, kp2l = k_temp  # type: ignore[misc]

        kp1u, kp1l = (yli - f(su)) / (xli - su), (yui - f(sl)) / (xui - sl)
        bp1u, bp1l = yli - kp1u * xli, yui - kp1l * xui

        aux_lines_l, aux_point_l = cls._construct_lower_aux_lines_and_points(  # type: ignore[arg-type]
            dim, idx, xui, yui, klui, sl, kp1l, kp2l, bp1l, bp2l, blul
        )
        aux_lines_u, aux_point_u = cls._construct_upper_aux_lines_and_points(  # type: ignore[arg-type]
            dim, idx, xli, yli, klui, su, kp1u, kp2u, bp1u, bp2u, bluu
        )

        if return_single_neuron_constrs:
            bli, bui = yli - kli * xli, yui - kui * xui
            temp = np.zeros((6, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [bui, btu, bluu, bli, btl, blul]
            temp[:, idx + 1] = [kui, ktu, kluu, kli, ktl, klul]
            temp[:, -1] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    @staticmethod
    def _construct_lower_aux_lines_and_points(
        dim: int,
        idx: int,
        xui: float | ndarray,
        yui: float | ndarray,
        klui: float | ndarray,
        sl: float | ndarray,
        kp1l: float | ndarray,
        kp2l: float | ndarray,
        bp1l: float | ndarray,
        bp2l: float | ndarray,
        bluu: float | ndarray,
    ) -> tuple[ndarray, float | ndarray | None]:  # (2, {dim}+{idx}+2)
        if abs((kp1l - kp2l) / (1 - kp1l * kp2l)) < MIN_DLP_ANGLE:
            aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [bluu]
            aux_lines_l[:, idx + 1] = [klui]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = None
        else:
            aux_lines_l = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [bp1l, bp2l]
            aux_lines_l[:, idx + 1] = [kp1l, kp2l]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = sl

        return aux_lines_l, aux_point_l

    @staticmethod
    def _construct_upper_aux_lines_and_points(
        dim: int,
        idx: int,
        xli: float | ndarray,
        yli: float | ndarray,
        klui: float | ndarray,
        su: float | ndarray,
        kp1u: float | ndarray,
        kp2u: float | ndarray,
        bp1u: float | ndarray,
        bp2u: float | ndarray,
        blul: float | ndarray,
    ) -> tuple[ndarray, float | ndarray | None]:  # (2, {dim}+{idx}+2)
        if abs((kp1u - kp2u) / (1 - kp1u * kp2u)) < MIN_DLP_ANGLE:
            aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [blul]
            aux_lines_u[:, idx + 1] = [klui]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = None
        else:
            aux_lines_u = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [bp1u, bp2u]
            aux_lines_u[:, idx + 1] = [kp1u, kp2u]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = su

        return aux_lines_u, aux_point_u

    @staticmethod
    @abstractmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        """
        Get the second tangent line given a point x1, which is not the tangent line taking x1 as tangent point.

        :param x1: The point where the tangent line is not taken.
        :param get_big: Whether to get the tangent line with a larger slope.

        :return: The bias, slope, and the tangent point of the tangent line.
        """

    @staticmethod
    @abstractmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        """
        Get the parallel tangent line given the slope.

        :param k: The slope of the tangent line.
        :param get_big: Whether to get the tangent line with a larger slope.

        :return: The bias, slope, and the tangent point of the tangent line.
        """
