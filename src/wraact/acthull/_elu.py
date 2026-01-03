__docformat__ = "restructuredtext"
__all__ = ["ELUHull"]

import numpy as np
from numpy import ndarray

from wraact._constants import ELU_MAX_AUX_POINT, MIN_DLP_ANGLE
from wraact._functions import delu_np, elu_np
from wraact.acthull._relulike import ReLULikeHull


class ELUHull(ReLULikeHull):
    """This is to calculate the function hull for the exponential linear unit (ELU) activation function."""

    @classmethod
    def cal_sn_constrs(
        cls,
        lb: ndarray,  # (d,)
        ub: ndarray,  # (d,)
    ) -> ndarray:  # (_, 2*d+1)
        """
        Calculate the single-neuron constraints of the function hull for ELU.

        We construct a *double-linear-piece* (DLP) function as the upper bound of
        the activation function and take upper constraints of the DLP function as
        the single-neuron constraints. Also, we construct constraints for the lower
        bound of the activation function.

        .. seealso::

            Refer to the paper ???

        :param lb:
        :param ub:
        :return:
        """
        if np.any(lb >= 0) or np.any(ub <= 0):
            raise ValueError(
                "The lower bounds should be negative and the upper bounds should be "
                "positive because we only handle the non-trivial cases for ELU."
            )

        d = lb.shape[0]
        s = (d, 2 * d + 1)
        cu = np.zeros(s, dtype=np.float64)
        cl1 = np.zeros(s, dtype=np.float64)
        cl2 = np.zeros(s, dtype=np.float64)
        cl3 = np.zeros(s, dtype=np.float64)

        idx_r = np.arange(d)  # The index of the rows
        idx_x = np.arange(1, d + 1)  # The index of the input variables
        idx_y = np.arange(d + 1, 2 * d + 1)  # The index of the output variables

        # For the upper faces.
        # The output constraints have the form of
        # -(yu - yl)*lb + (yu - yl)*x - (ub-lb)*y >= 0
        yu, yl = cls._f(ub), cls._f(lb)
        cu[:, 0] = -lb * (yu - yl)
        cu[idx_r, idx_x] = yu - yl
        cu[idx_r, idx_y] = lb - ub

        # For the lower faces.
        # y >= f'(lb)(x-lb) + f(lb)
        kl = cls._df(lb)
        cl1[:, 0] = -cls._f(lb) + kl * lb
        cl2[idx_r, idx_y] = 1.0
        cl1[idx_r, idx_x] = kl
        # y - x >= 0
        cl2[idx_r, idx_y] = 1.0
        cl2[idx_r, idx_x] = -1.0
        # y >=f'(m)(x-m) + f(m)
        m = (lb + ub) / 2.0
        km = cls._df(m)
        cl3[:, 0] = -cls._f(m) + km * m
        cl3[idx_r, idx_y] = 1.0
        cl3[idx_r, idx_x] = km

        c = np.vstack((cu, cl1, cl2, cl3))
        return c

    @classmethod
    def _construct_dlp(
        cls, idx: int, dim: int, lb: float, ub: float
    ) -> tuple[ndarray, float | None]:
        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)

        yl = cls._f(lb)
        yu = cls._f(ub)
        if lb >= ELU_MAX_AUX_POINT or ub <= ELU_MAX_AUX_POINT:
            k = (yu - yl) / (ub - lb)
            b = yu - k * ub
            aux_lines = np.asarray([[b, *temp1, k, *temp2, -1.0]], dtype=np.float64)
            return aux_lines, None

        # The intersection point of the two linear pieces should not be positive to
        # avoid large coefficients in the upper bound because the linear pieces are
        # too close to the upper bound.
        m = min((lb + ub) / 2.0, ELU_MAX_AUX_POINT)

        kp1 = (yu - cls._df(m)) / (ub - m)
        bp1 = yu - kp1 * ub
        kp2 = (yl - cls._df(m)) / (lb - m)
        bp2 = yl - kp2 * lb

        # Estimate the angle of the two linear pieces to avoid large coefficients.
        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < MIN_DLP_ANGLE:
            k = (yu - yl) / (ub - lb)
            b = yu - k * ub
            aux_lines = np.asarray([[b, *temp1, k, *temp2, -1.0]], dtype=np.float64)
            return aux_lines, None

        aux_lines = np.asarray(
            [
                [bp1, *temp1, kp1, *temp2, -1.0],
                [bp2, *temp1, kp2, *temp2, -1.0],
            ],
            dtype=np.float64,
        )
        aux_point = m

        return aux_lines, aux_point

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return elu_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return delu_np(x)
