__docformat__ = "restructuredtext"
__all__ = ["LeakyReLUHull"]

from typing import ClassVar

import numpy as np
from numpy import ndarray

from wraact._constants import LEAKY_RELU_ALPHA, MIN_BOUNDS_RANGE_ACTHULL
from wraact._functions import dleakyrelu_np, leakyrelu_np
from wraact.acthull._relulike import ReLULikeHull


class LeakyReLUHull(ReLULikeHull):
    """This is to calculate the function hull for the leaky rectified linear unit (LeakyReLU) activation function."""

    _lower_constraints: ClassVar[dict[int, ndarray]] = {}

    @classmethod
    def cal_sn_constrs(
        cls,
        lb: ndarray,  # (d,)
        ub: ndarray,  # (d,)
    ) -> ndarray:  # (3*d, 2*d+1)
        """
        Calculate the single-neuron constraints of the function hull for LeakyReLU.

        This is similar to the ReLU.
        We use *triangle relaxation* to calculate the single-neuron constraints of
        LeakyReLU.

        .. seealso::

            Refer to the paper ???


        :param lb: The lower bounds of the input variables.
        :param ub: The upper bounds of the input variables.
        :return: The single-neuron constraints of the function hull.
        """
        if np.any(lb >= 0) or np.any(ub <= 0):
            raise ValueError(
                "The lower bounds should be negative and the upper bounds should be "
                "positive because we only handle the non-trivial cases for LeakyReLU."
            )

        d = lb.shape[0]
        c = np.zeros((d, 2 * d + 1), dtype=np.float64)

        idx_r = np.arange(d)  # The index of the rows
        idx_x = np.arange(1, d + 1)  # The index of the input variables
        idx_y = np.arange(d + 1, 2 * d + 1)  # The index of the output variables

        # For the upper faces.
        # The output constraints have the form of
        # alpha * lb - (ub + alpha * lb) * lb + (ub + alpha * lb) * x - (ub - lb) * y >= 0.

        c[:, 0] = LEAKY_RELU_ALPHA * lb - (ub + LEAKY_RELU_ALPHA * lb) * lb
        c[idx_r, idx_x] = ub + LEAKY_RELU_ALPHA * lb
        c[idx_r, idx_y] = -(ub - lb)

        # For the lower faces.
        if cls._lower_constraints.get(d) is None:
            # The output constraints have the form of y >= alpha * x
            cu1 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            cu1[idx_r, idx_x] = -LEAKY_RELU_ALPHA
            cu1[idx_r, idx_y] = 1.0

            # The output constraints have the form of y >= u.
            cu2 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            cu2[idx_r, idx_x] = -1.0
            cu2[idx_r, idx_y] = 1.0

            cu = np.vstack((cu1, cu2))
            cls._lower_constraints[d] = cu
        else:
            cu = cls._lower_constraints[d]

        c = np.vstack((c, cu))

        return c

    @classmethod
    def _construct_dlp(
        cls, idx: int, dim: int, lb: float, ub: float
    ) -> tuple[ndarray, float | None]:
        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)

        if lb >= 0:
            aux_lines = np.asarray([[0.0, *temp1, 1.0, *temp2, -1.0]], dtype=np.float64)
            return aux_lines, None

        if ub <= 0:
            aux_lines = np.asarray(
                [[0.0, *temp1, LEAKY_RELU_ALPHA, *temp2, -1.0]], dtype=np.float64
            )
            return aux_lines, None

        if ub - lb < MIN_BOUNDS_RANGE_ACTHULL:
            k = (ub - lb) / (ub - lb)
            b = ub - k * ub
            aux_lines = np.asarray([[b, *temp1, k, *temp2, -1.0]], dtype=np.float64)
            return aux_lines, None

        kp1 = LEAKY_RELU_ALPHA
        kp2 = 1.0
        aux_lines = np.asarray(
            [
                [0.0, *temp1, kp1, *temp2, -1.0],
                [0.0, *temp1, kp2, *temp2, -1.0],
            ],
            dtype=np.float64,
        )
        aux_point = 0.0
        return aux_lines, aux_point

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return leakyrelu_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dleakyrelu_np(x)
