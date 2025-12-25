__docformat__ = "restructuredtext"
__all__ = ["ELUHull"]

import numpy as np
from numpy import ndarray

from wraact._functions import delu_np, elu_np
from wraact.acthull._relulike import ReLULikeHull

_ELU_MAX_AUX_POINT = -1.25
_MIN_DLP_ANGLE = 0.1
"""
 The minimum angle between two lines of the DLP function.
 Given two slopes m1 and m2, the angle between them is calculated by
 arctan(abs((m1 - m2) / (1 + m1 * m2))), but we only use abs((m1 - m2) / (1 + m1 * m2))
 to estimate.
"""


class ELUHull(ReLULikeHull):
    """This is to calculate the function hull for the exponential linear unit (ELU) activation function."""

    @classmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (_, 2*d+1)
        """
        Calculate the single-neuron constraints of the function hull for ELU.

        We construct a *double-linear-piece* (DLP) function as the upper bound of
        the activation function and take upper constraints of the DLP function as
        the single-neuron constraints. Also, we construct constraints for the lower
        bound of the activation function.

        .. seealso::

            Refer to the paper ???

        :param l:
        :param u:
        :return:
        """
        if np.any(l >= 0) or np.any(u <= 0):
            raise ValueError(
                "The lower bounds should be negative and the upper bounds should be "
                "positive because we only handle the non-trivial cases for ELU."
            )

        d = l.shape[0]
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
        # -(yu - yl)*l + (yu - yl)*x - (u-l)*y >= 0
        yu, yl = cls._f(u), cls._f(l)
        cu[:, 0] = -l * (yu - yl)
        cu[idx_r, idx_x] = yu - yl
        cu[idx_r, idx_y] = l - u

        # For the lower faces.
        # y >= f'(l)(x-l) + f(l)
        kl = cls._df(l)
        cl1[:, 0] = -cls._f(l) + kl * l
        cl2[idx_r, idx_y] = 1.0
        cl1[idx_r, idx_x] = kl
        # y - x >= 0
        cl2[idx_r, idx_y] = 1.0
        cl2[idx_r, idx_x] = -1.0
        # y >=f'(m)(x-m) + f(m)
        m = (l + u) / 2.0
        km = cls._df(m)
        cl3[:, 0] = -cls._f(m) + km * m
        cl3[idx_r, idx_y] = 1.0
        cl3[idx_r, idx_x] = km

        c = np.vstack((cu, cl1, cl2, cl3))
        return c

    @classmethod
    def _construct_dlp(cls, idx: int, dim: int, l: float, u: float) -> tuple[ndarray, float | None]:
        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)

        yl = cls._f(l)
        yu = cls._f(u)
        if l >= _ELU_MAX_AUX_POINT or u <= _ELU_MAX_AUX_POINT:
            k = (yu - yl) / (u - l)
            b = yu - k * u
            aux_lines = np.asarray([[b] + temp1 + [k] + temp2 + [-1.0]], dtype=np.float64)
            return aux_lines, None

        # The intersection point of the two linear pieces should not be positive to
        # avoid large coefficients in the upper bound because the linear pieces are
        # too close to the upper bound.
        m = min((l + u) / 2.0, _ELU_MAX_AUX_POINT)

        kp1 = (yu - cls._df(m)) / (u - m)
        bp1 = yu - kp1 * u
        kp2 = (yl - cls._df(m)) / (l - m)
        bp2 = yl - kp2 * l

        # Estimate the angle of the two linear pieces to avoid large coefficients.
        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < _MIN_DLP_ANGLE:
            k = (yu - yl) / (u - l)
            b = yu - k * u
            aux_lines = np.asarray([[b] + temp1 + [k] + temp2 + [-1.0]], dtype=np.float64)
            return aux_lines, None

        aux_lines = np.asarray(
            [
                [bp1] + temp1 + [kp1] + temp2 + [-1.0],
                [bp2] + temp1 + [kp2] + temp2 + [-1.0],
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
