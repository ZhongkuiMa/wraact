__docformat__ = "restructuredtext"
__all__ = ["ReLUHullWithOneY"]

import numpy as np
from numpy import ndarray

from wraact._constants import TOLERANCE
from wraact.acthull import ReLUHull
from wraact.oney._relulike import ReLULikeHullWithOneY


class ReLUHullWithOneY(ReLULikeHullWithOneY, ReLUHull):
    """
    The class to calculate the function hull for the rectified linear unit (ReLU) activation function with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`ReLUHull` for more details.
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
        v = np.transpose(v)
        mask_xp, mask_xn = (v > TOLERANCE), (v < -TOLERANCE)
        if not np.any(mask_xp) and not np.any(mask_xn):
            raise ArithmeticError("The vertices should not all positive or all negative.")

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
