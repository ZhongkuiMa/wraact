"""Utility functions for activation hull constraint computation."""

__docformat__ = "restructuredtext"
__all__ = ["cal_mn_constrs_with_one_y_dlp"]

import numpy as np
from numpy import ndarray

from wraact._constants import MIN_DLP_DENOM


def cal_mn_constrs_with_one_y_dlp(
    idx: int,
    c: ndarray,
    v: ndarray,
    aux_lines: ndarray,
    aux_point: float | ndarray | None,
    is_convex: bool = True,
) -> tuple[ndarray, ndarray]:
    """
    Calculate the multi-neuron constraints for one specified input dimension of the function hull for the DLP (double linear pieces) function.

    :param idx: The index of the input dimension.
    :param c: The constraints of input polytope.
    :param v: The vertices of input polytope.
    :param aux_point: The auxiliary point that is the intersection of the two linear
        pieces.
    :param aux_lines: The auxiliary line where the two linear pieces are located.
        Each row represents a line. If there is only one line, it is the trivial case
        (linear function), which means we do not need a DLP function. If there are two
        lines, the first line is the left line and the second line is the right line.
    :param is_convex: Whether the DLP function is defined by max or min. If it is max,
        then the function is convex and is_convex is True; otherwise, it is False.

    :return: The output constraints and vertices of the function hull after extending
        one specified output dimension.

    :raises RuntimeError: If the auxiliary point is not provided and the auxiliary lines
        should have only one line to represent the trivial case (linear function).
    :raises ArithmeticError: If the vertices should not all greater/smaller than the
        auxiliary point or divided by zero during calculation.
    """
    pad_width = ((0, 0), (0, 1))  # Extend one output dimension that is a new column.

    if aux_point is None:
        # The auxiliary point is not provided. The auxiliary lines should have only one
        # line to represent the trivial case (linear function).
        # Only need to extend the constraints and vertices in output space.
        if aux_lines.shape[0] != 1:
            raise RuntimeError(
                "The auxiliary point is not provided and the auxiliary "
                "lines should have only one line to represent the "
                "trivial case (linear function)."
            )

        line = aux_lines[[-1]]
        v = np.hstack((v, np.matmul(v, line[:, :-1].T)))
        c = np.pad(c, pad_width)
        c = np.vstack((c, line)) if is_convex else np.vstack((c, -line))
        return c, v

    v = np.pad(v, pad_width)
    c = np.pad(c, pad_width)

    vc = v[:, idx + 1]
    mask_vl, mask_vr = (vc < aux_point), (vc > aux_point)

    if not np.any(mask_vl) or not np.any(mask_vr):
        raise RuntimeError("The vertices should not all greater/smaller than the auxiliary point.")

    ll, lr = aux_lines[[[0], [1]]] if is_convex else aux_lines[[[1], [0]]]

    vl, vr = v[mask_vl], v[mask_vr]
    v[mask_vl, -1], v[mask_vr, -1] = np.matmul(vl, ll.T).T, np.matmul(vr, lr.T).T
    vl, vr = v[mask_vl], v[mask_vr]
    constrs_at_vr, constrs_at_vl = np.matmul(c, vr.T), np.matmul(c, vl.T)
    left_line_at_vr, right_line_at_vl = np.matmul(ll, vr.T), np.matmul(lr, vl.T)

    if np.any(np.abs(left_line_at_vr) < MIN_DLP_DENOM) or np.any(
        np.abs(right_line_at_vl) < MIN_DLP_DENOM
    ):
        raise RuntimeError("Zero or near-zero values will be in denominators.")

    constrs_at_vr /= left_line_at_vr
    constrs_at_vl /= right_line_at_vl

    if is_convex:
        beta1 = np.max(constrs_at_vr, axis=1, keepdims=True)
        beta2 = np.max(constrs_at_vl, axis=1, keepdims=True)
    else:
        beta1 = np.min(constrs_at_vr, axis=1, keepdims=True)
        beta2 = np.min(constrs_at_vl, axis=1, keepdims=True)

    c -= beta1 * ll
    c -= beta2 * lr

    return c, v
