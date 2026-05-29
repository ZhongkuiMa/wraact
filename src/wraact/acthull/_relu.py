"""ReLU activation hull computation."""

__docformat__ = "restructuredtext"
__all__ = ["ReLUHull"]

from typing import ClassVar

import numpy as np
from numpy import ndarray

from wraact._constants import TOLERANCE
from wraact._functions import drelu_np, relu_np
from wraact.acthull._relulike import ReLULikeHull


class ReLUHull(ReLULikeHull):
    """
    This is to calculate the function hull for the rectified linear unit (ReLU) activation function.

    .. tip::
        This is an ad hoc implementation for ReLU to obtain the function hull
        considering high efficiency and accuracy based on the two linear pieces (
        :math:`y=x` and :math:`y=0`) of ReLU.
    """

    _lower_constraints: ClassVar[dict[int, ndarray]] = {}

    @classmethod
    def cal_sn_constrs(
        cls,
        lb: ndarray,
        ub: ndarray,
    ) -> ndarray:
        """
        Calculate the single-neuron constraints of the function hull for ReLU.

        We use *triangle relaxation* to calculate the single-neuron constraints of
        ReLU.

        :param lb: The lower bounds of the input variables.
        :param ub: The upper bounds of the input variables.
        :return: The single-neuron constraints of the function hull.
        """
        d = lb.shape[0]
        cc = np.empty((0, 2 * d + 1), dtype=np.float64)

        # Determine which dimensions are trivial (ReLU is linear).
        # all-positive -> y = x; all-negative -> y = 0;
        # crossing zero -> triangle relaxation.
        fully_positive = lb >= 0
        fully_negative = ub <= 0
        crossing = ~(fully_positive | fully_negative)

        if not np.any(crossing):
            # All dimensions are trivial: return identity / zero constraints.
            for i in range(d):
                row = np.zeros(2 * d + 1, dtype=np.float64)
                if fully_positive[i]:
                    row[1 + i] = -1.0  # -x_i + y >= 0
                    row[1 + d + i] = 1.0
                else:
                    row[1 + d + i] = 1.0  # y >= 0
                cc = np.vstack((cc, row))
            return cc

        c = np.zeros((int(np.sum(crossing)), 2 * d + 1), dtype=np.float64)
        cross_idxs = np.where(crossing)[0]

        # For the upper faces across crossing dimensions.
        c[:, 0] = -ub[cross_idxs] * lb[cross_idxs]
        c[:, 1 + cross_idxs] = np.diag(ub[cross_idxs])
        c[:, 1 + d + cross_idxs] = np.diag(-(ub[cross_idxs] - lb[cross_idxs]))

        cc = np.vstack((cc, c))

        # For the lower faces (shared across all dimensions).
        if cls._lower_constraints.get(d) is None:
            c_l1 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            c_l1[np.arange(d), 1 + d + np.arange(d)] = 1.0  # y >= 0

            c_l2 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            c_l2[np.arange(d), 1 + np.arange(d)] = -1.0  # y >= x
            c_l2[np.arange(d), 1 + d + np.arange(d)] = 1.0

            cl = np.vstack((c_l1, c_l2))
            cl.flags.writeable = False
            cls._lower_constraints[d] = cl
        else:
            cl = cls._lower_constraints[d]

        cc = np.vstack((cc, cl))

        return cc

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,
        v: ndarray,
        lb: ndarray | None,
        ub: ndarray | None,
    ) -> ndarray:
        """
        Calculate the multi-neuron constraints of the function hull.

        We use the algorithm called *WraLU* to calculate the multi-neuron
        constraints.

        .. seealso::

            Refer to the paper:
            `ReLU Hull Approximation <https://dl.acm.org/doi/pdf/10.1145/3632917>`__
            :cite:`ma_relu_2024`

        :param c: The constraints of the input polytope.
        :param v: The vertices of the input polytope.
        :param lb: The lower bounds of the input variables.
        :param ub: The upper bounds of the input variables.

        :return: The multi-neuron constraints of the function hull.
        """
        # TODO: Add the code for degenerated input polytope (all vertices lie in one
        # orthant, making all v[i] same-sign -> WraLU iteration degenerates).

        dim = c.shape[1] - 1

        v = np.transpose(v)
        mask_xp, mask_xn = (v > TOLERANCE), (v < -TOLERANCE)
        if not np.any(mask_xp) and not np.any(mask_xn):
            raise RuntimeError("The vertices should not all positive or all negative.")

        y = np.maximum(v, 0)  # The vertices coordinate in output space.

        cv = np.matmul(c, v)

        s = (c.shape[0], 1)
        beta1 = np.zeros(s, dtype=np.float64)
        beta2 = np.zeros(s, dtype=np.float64)

        for i in range(1, dim + 1):
            mask_xp_i, mask_xn_i = mask_xp[i], mask_xn[i]

            if np.any(mask_xp_i):
                temp = cv[:, mask_xp_i] / v[i, mask_xp_i]
                beta1[:, 0] = -np.min(temp, axis=1)

            if np.any(mask_xn_i):
                temp = cv[:, mask_xn_i] / v[i, mask_xn_i]
                beta2[:, 0] = np.max(temp, axis=1)

            c = np.hstack((c, beta1 + beta2))
            c[:, [i]] -= beta2

            v = np.vstack((v, y[i]))
            cv += np.outer(c[:, -1], y[i]) + np.outer(-beta2, v[i])

            beta1.fill(0.0)
            beta2.fill(0.0)

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
        raise RuntimeError("The method should not be called.")

    @classmethod
    def _construct_dlp(cls, *args, **kwargs):
        raise RuntimeError("The method should not be called.")

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return relu_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return drelu_np(x)
