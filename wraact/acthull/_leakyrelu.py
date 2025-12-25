__docformat__ = "restructuredtext"
__all__ = ["LeakyReLUHull"]

import numpy as np
from numpy import ndarray

from wraact._functions import dleakyrelu_np, leakyrelu_np
from wraact.acthull._relulike import ReLULikeHull

_LEAKY_RELU_ALPHA = 0.01
_MIN_BOUNDS_RANGE = 0.05


class LeakyReLUHull(ReLULikeHull):
    """This is to calculate the function hull for the leaky rectified linear unit (LeakyReLU) activation function."""

    _lower_constraints: dict[int, ndarray] = {}

    @classmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (3*d, 2*d+1)
        """
        Calculate the single-neuron constraints of the function hull for LeakyReLU.

        This is similar to the ReLU.
        We use *triangle relaxation* to calculate the single-neuron constraints of
        LeakyReLU.

        .. seealso::

            Refer to the paper ???


        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.
        :return: The single-neuron constraints of the function hull.
        """
        if np.any(l >= 0) or np.any(u <= 0):
            raise ValueError(
                "The lower bounds should be negative and the upper bounds should be "
                "positive because we only handle the non-trivial cases for LeakyReLU."
            )

        d = l.shape[0]
        c = np.zeros((d, 2 * d + 1), dtype=np.float64)

        idx_r = np.arange(d)  # The index of the rows
        idx_x = np.arange(1, d + 1)  # The index of the input variables
        idx_y = np.arange(d + 1, 2 * d + 1)  # The index of the output variables

        # For the upper faces.
        # The output constraints have the form of
        # alpha * l - (u + alpha * l) * l + (u + alpha * l) * x - (u - l) * y >= 0.

        c[:, 0] = _LEAKY_RELU_ALPHA * l - (u + _LEAKY_RELU_ALPHA * l) * l
        c[idx_r, idx_x] = u + _LEAKY_RELU_ALPHA * l
        c[idx_r, idx_y] = -(u - l)

        # For the lower faces.
        if cls._lower_constraints.get(d) is None:
            # The output constraints have the form of y >= alpha * x
            cu1 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            cu1[idx_r, idx_x] = -_LEAKY_RELU_ALPHA
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
    def _construct_dlp(cls, idx: int, dim: int, l: float, u: float) -> tuple[ndarray, float | None]:
        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)

        if l >= 0:
            aux_lines = np.asarray([[0.0] + temp1 + [1.0] + temp2 + [-1.0]], dtype=np.float64)
            return aux_lines, None

        if u <= 0:
            aux_lines = np.asarray(
                [[0.0] + temp1 + [_LEAKY_RELU_ALPHA] + temp2 + [-1.0]], dtype=np.float64
            )
            return aux_lines, None

        if u - l < _MIN_BOUNDS_RANGE:
            k = (u - l) / (u - l)
            b = u - k * u
            aux_lines = np.asarray([[b] + temp1 + [k] + temp2 + [-1.0]], dtype=np.float64)
            return aux_lines, None

        kp1 = _LEAKY_RELU_ALPHA
        kp2 = 1.0
        aux_lines = np.asarray(
            [
                [0.0] + temp1 + [kp1] + temp2 + [-1.0],
                [0.0] + temp1 + [kp2] + temp2 + [-1.0],
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
