__docformat__ = "restructuredtext"
__all__ = ["MaxPoolHull", "MaxPoolHullDLP"]


from typing import ClassVar, Literal

import numpy as np
from numpy import ndarray

from wraact._constants import TOLERANCE
from wraact.acthull._relulike import ReLULikeHull


class MaxPoolHullDLP(ReLULikeHull):
    """
    This is to calculate the function hull for the max pooling activation function.

    In this method, we construct a DLP function as the upper bound of a MaxPool
    function.

    .. tip::

        **Trivial cases of MaxPool function hull**.
        Before calculate the function hull, we can filter some trivial cases of the
        MaxPool function hull. For example, when the lower bound of one input variable
        is larger than the upper bound of all other input variables, the MaxPool is a
        linear function with only outputting the variable having the largest lower
        bound.
        But this method does not filter *all* trivial cases.

        Furthermore, when calculating the function hull, we can filter *all* trivial
        cases based on the vertices of the input polytope. If the largest entry of
        each vertex has the same coordinate, then the MaxPool function hull is a
        trivial case.

        Because we know some input variable will never be the maximum with the given
        input domain, we can reduce some computation by removing these dimension to
        improve the efficiency.

    .. tip::

        For the DLP versioned MaxPool function hull, those input coordinates that never
        be the maximum can be removed when constructing the DLP function.

    .. seealso::
        Refer to the paper ???
        for theoretical proof.

    """

    _lower_constraints: ClassVar[dict[int, ndarray]] = {}

    def cal_constrs(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        lb: ndarray | None,  # (d-1,)
        ub: ndarray | None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, d+1)
        d = c.shape[1] - 1
        c = np.array(c, dtype=np.float64)
        lb = np.array(lb, dtype=np.float64)
        ub = np.array(ub, dtype=np.float64)
        cc = np.empty((0, d + 2), dtype=np.float64)

        if self._add_sn_constrs:
            c1 = self.cal_sn_constrs(lb, ub)
            cc = np.vstack((cc, c1))

        if self._add_mn_constrs:
            c2 = self.cal_mn_constrs(c, v, lb, ub)
            cc = np.vstack((cc, c2))

        return cc, dtype_cdd

    @classmethod
    def cal_sn_constrs(
        cls,
        lb: ndarray,  # (d,)
        ub: ndarray,  # (d,)
    ) -> ndarray:  # (d+1, d+2)
        """
        Calculate the single-neuron constraints for the MaxPool function.

        .. seealso::
            Refer to the paper
            `Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks
            <https://arxiv.org/pdf/1705.01320>`__ :cite:`ehlers_formal_2017`
            for specific constraints and theoretical proof.

        :param lb: The lower bounds of the input variables.
        :param ub: The upper bounds of the input variables.

        :return: The single-neuron constraints.
        """
        d = lb.shape[0]
        c_u = np.zeros((1, d + 2), dtype=lb.dtype)

        # Upper bounds
        # (1) The following solution is not precise enough.
        # lb_sum = np.sum(lb)
        # lb_max = np.max(lb)
        # c_u[-1, 0] = lb_max - lb_sum
        # c_u[-1, 1:-1] = 1.0
        # c_u[-1, -1] = -1.0
        # (2) y < ub_max
        c_u[-1, 0] = np.max(ub)
        c_u[-1, -1] = -1.0

        # Lower bound
        # Here we do not remove the redundant constraints consider the trivial case,
        # because there are a few lower constraints. So we decide to keep them.
        if cls._lower_constraints.get(d) is not None:
            c_l = cls._lower_constraints[d]
        else:
            # y >= x_i for all i
            c_l = np.zeros((d, d + 2), dtype=lb.dtype)
            r_idx = np.arange(d)
            c_idx = np.arange(1, d + 1)
            # Lower bounds
            # - x_i + y >= 0
            c_l[r_idx, c_idx] = -1.0
            c_l[r_idx, -1] = 1.0

        cc = np.vstack((c_u, c_l))

        return cc

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        lb: ndarray | None,  # (d-1,)
        ub: ndarray | None,  # (d-1,)
    ) -> ndarray:  # (n, d+1)
        # ------------------------ Trivial case ------------------------
        cc = cls._handle_case_of_one_vertex(v)
        if cc is not None:
            return cc
        cc = cls._handle_case_of_one_piece(v)
        if cc is not None:
            return cc

        # ------------------------ Non-trivial case ------------------------
        nt_idxs = cls._find_nontrivial_idxs(v)
        # Other degenrate cases are handled in _construct_dlp.
        pieces = cls._construct_dlp(v, nt_idxs)  # (2, 1+d+1)
        # After constructing the DLP function, we still may meet trivial cases due to
        # the construction method.
        # We calculate the maximum value of each piece.
        pv = pieces[:, :-1] @ v.T  # (2, n_v)
        nt_idxs = sorted(set(np.argmax(pv, axis=0)))
        if len(nt_idxs) == 1:
            idx = nt_idxs.pop()
            cc = np.zeros((2, c.shape[1] + 1), dtype=c.dtype)
            # y >= the piece
            cc[0, :] = pieces[idx]
            # y <= the piece
            cc[1, :] = -pieces[idx]
            return cc

        # Enumerate all vertices and calculate (Ax + b) / (y - x_i) for each vertex.
        # Calculate the Ax + b
        Axb = c @ v.T  # (n_c, n_v)
        Axb = np.expand_dims(Axb, 1)  # (n_c, 1, n_v)
        if not np.all(Axb >= -TOLERANCE):
            raise RuntimeError(f"Negative beta.\nAxb={Axb}.")
        # Axb = np.maximum(Axb, 0.0)  # Remove tiny negative values

        # Calculate y - each piece
        v_y = np.max(pv, 0, keepdims=True)  # (1, n_v)
        yx = v_y - pv  # (2, n_v)
        yx = np.expand_dims(yx, 0)  # (1, 2, n_v)
        # Calculate (Ax + b) / (y - \sum x_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(yx != 0, Axb / yx, np.inf)  # (n_c, 2, n_v)
        if not np.all(beta >= -TOLERANCE):
            raise RuntimeError(f"Negative beta.\nbeta={beta}.")
        # beta = np.maximum(beta, 0.0)  # Remove tiny negative values

        # Find the minimum value of beta for all vertices to maintain the soundness of
        # the function hull.
        beta = np.min(beta, 2)  # (n_c, 2)

        if np.isinf(np.max(beta)):
            raise RuntimeError(f"Inf beta.\nbeta={beta}.")

        # Filter the useless constraints.
        # Set the non-largest value to zero
        # Theoretically, there is at most one non-zero beta value, so the following is
        # redundant. But we still do this for numerical stability.
        # That means we only accept one non-zero beta value.
        beta_max = np.max(beta, 1, keepdims=True)  # (n_c, 1)
        beta = np.where(beta < beta_max, 0.0, beta)  # (n_c, 2)

        # \beta * (y - \sum x_i).
        c2 = np.matmul(beta, pieces)  # (n, 1+d+1)

        # The final constraints are Ax + b - \beta * (y - \sum x_i) >= 0.
        # Add -\beta * (y - \sum x_i).
        cc = -c2  # (n, 1+d+1)
        # Add Ax + b
        cc[:, :-1] += c  # (n, d+1)

        return cc

    @staticmethod
    def _handle_case_of_one_vertex(v: np.ndarray) -> np.ndarray | None:
        # If there is only one vertex, the MaxPool function is a constant function.
        # For example, all input variables are zeros.
        if len(v) != 1:
            return None
        cc = np.zeros((2, v.shape[1] + 1), dtype=v.dtype)
        const = np.max(v[0, 1:])
        # y >= const
        cc[0, 0] = -const
        cc[0, -1] = 1.0
        # y <= const
        cc[1, 0] = const
        cc[1, -1] = -1.0
        return cc

    @staticmethod
    def _handle_case_of_one_piece(v: np.ndarray) -> np.ndarray | None:
        # If there is only one non-trivial index, the MaxPool function is a linear
        # function, which including the case of only two vertices.

        # The following code find a dimension is always the maximum.
        # The code consider the equivalent case of some dimension.
        # For example, [[0,0], [0,1]] should have the result of 1, but the first row may
        # tell you the maximum is the first dimension, where the second dimension has
        # the same value.
        row_max = np.max(v[:, 1:], axis=1, keepdims=True)
        # Mask all values that equal to the maximum value.
        mask_max = np.isclose(v[:, 1:], row_max)
        # Check if there is a dimension is always the maximum.
        is_trivial = np.all(mask_max, axis=0)
        if not np.any(is_trivial):
            return None
        idx = np.argmax(is_trivial)

        cc = np.zeros((2, v.shape[1] + 1), dtype=v.dtype)
        # y >= x_idx
        cc[0, idx + 1] = -1.0
        cc[0, -1] = 1.0
        # y <= x_idx
        cc[1, idx + 1] = 1.0
        cc[1, -1] = -1.0
        return cc

    @staticmethod
    def _find_nontrivial_idxs(v: np.ndarray) -> list[int]:
        # Find the maximum value of each vertex.
        row_max = np.max(v[:, 1:], axis=1, keepdims=True)
        # Mask all values that equal to the maximum value.
        mask_max = np.isclose(v[:, 1:], row_max)
        # If one row has multiple maximum values, the row is a trivial case.
        # Find the row that has only one maximum value.
        # Actually, theoretically, there must at least one vertex with only one maximum
        # dimension on a piece.
        mask_nontrivial = np.sum(mask_max, axis=1) == 1
        # Record the column index of the non-trivial maximum value.
        nt_idxs = np.argmax(mask_max[mask_nontrivial], axis=1)
        return nt_idxs

    @classmethod
    def _cal_mn_constrs_with_one_y(
        cls,
        idx: int,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        dlp_lines: ndarray,  # (2, d+1)
        dlp_point: float,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:  # (n, d+1) , (m, d+1)
        raise RuntimeError("This method should not be called.")

    @classmethod
    def _construct_dlp(
        cls,
        v: ndarray,  # (m, d)
        nt_idxs: list[int],
    ) -> ndarray:  # (2, d+1)
        # When constructing the DLP function as the upper bound of the MaxPool
        # function, we only need to consider the nontrivial coordinates.

        d = v.shape[1] - 1

        # Get the lower and upper bounds of the non-trivial indices based on vertices.
        lb = np.min(v[:, 1:], axis=0)
        ub = np.max(v[:, 1:], axis=0)

        r = ub - lb
        # Get the indices of r in descending order.
        ordered_r_idx = np.argsort(r)[::-1]

        # Remove the trivial cases.
        ordered_r_idx = np.asarray([idx for idx in ordered_r_idx if idx in nt_idxs])

        # Group ordered_r_idx into two groups:
        # the first group contains the odd indices of ordered_r_idx,
        # the second group contains the even indices of ordered_r_idx.
        r_idx_odd = ordered_r_idx[::2]
        r_idx_even = ordered_r_idx[1::2]

        dlp_lines = np.zeros((2, 2 + d), dtype=np.float64)
        # y >= \sum x_{r_idx_odd}
        # y >= \sum x_{r_idx_even}
        dlp_lines[:, -1] = 1.0
        dlp_lines[0, r_idx_odd + 1] = -1.0
        dlp_lines[1, r_idx_even + 1] = -1.0

        return dlp_lines

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        raise RuntimeError("This method should not be called.")

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        raise RuntimeError("This method should not be called.")


class MaxPoolHull(MaxPoolHullDLP):
    """
    This is to calculate the function hull for the max pooling activation function.

    In this method, we calculate the function hull without constructing a DLP function.

    .. tip::

        **Trivial cases of MaxPool function hull**.
        Before calculate the function hull, we can filter some trivial cases of the
        MaxPool function hull. For example, when the lower bound of one input variable
        is larger than the upper bound of all other input variables, the MaxPool is a
        linear function with only outputting the variable having the largest lower
        bound.
        But this method does not filter *all* trivial cases.

        Furthermore, when calculating the function hull, we can filter *all* trivial
        cases based on the vertices of the input polytope. If the largest entry of
        each vertex has the same coordinate, then the MaxPool function hull is a
        trivial case.

        Because we know some input variable will never be the maximum with the given
        input domain, we can reduce some computation by removing these dimension to
        improve the efficiency.

    .. seealso::
        Refer to the paper ???
        for theoretical proof.
    """

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        lb: ndarray | None,  # (d-1,)
        ub: ndarray | None,  # (d-1,)
    ) -> ndarray:  # (n, d+1)
        # This only calculate the upper constraints of the function hull, which are
        # non-trivial constraints.

        # ------------------------ Trivial case ------------------------
        cc = cls._handle_case_of_one_vertex(v)
        if cc is not None:
            return cc
        cc = cls._handle_case_of_one_piece(v)
        if cc is not None:
            return cc

        # ------------------------ Non-trivial case ------------------------
        nt_idxs = cls._find_nontrivial_idxs(v)
        # Other degenrate cases are handled in the following calculation.
        # Enumerate all vertices and calculate (Ax + b) / (y - x_i) for each vertex.
        # Calculate the Ax + b
        Axb = c @ v.T  # (n_c, n_v)
        Axb = np.expand_dims(Axb, 1)  # (n_c, 1, n_v)
        if not np.all(Axb >= -TOLERANCE):
            raise RuntimeError(f"Negative beta.\nAxb={Axb}.")
        Axb = np.maximum(Axb, 0.0)  # Remove tiny negative values

        # Calculate y - x_i
        v_y = np.max(v[:, 1:], 1, keepdims=True)  # (n_v, 1)
        yx = v_y - v[:, 1:][:, nt_idxs]  # (n_v, d')
        yx = np.expand_dims(yx.T, 0)  # (1, d', n_v)

        # Calculate (Ax + b) / (y - x_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(yx != 0, Axb / yx, np.inf)  # (n_c, d', n_v)

        if not np.all(beta >= -TOLERANCE):
            raise RuntimeError(f"Negative beta.\nbeta={beta}.")
        # beta = np.minimum(beta, 0.0)  # Remove tiny negative values

        # Find the minimum value of beta for all vertices to maintain the soundness of
        # the function hull.
        beta = np.min(beta, 2)  # (n_c, d')
        if np.isinf(np.max(beta)):
            raise RuntimeError(f"Inf beta.\nbeta={beta}.")

        # Filter the useless constraints.
        # Set the non-largest value to zero
        # Theoretically, there is at most one non-zero beta value, so the following is
        # redundant. But we still do this for numerical stability.
        # That means we only accept one non-zero beta value.
        beta_max = np.max(beta, 1, keepdims=True)  # (n_c, 1)
        beta = np.where(beta < beta_max, 0.0, beta)  # (n_c, d')

        # The final constraints are Ax + b - \beta * (y - x_i) >= 0.
        # Add Ax + b
        cc = c
        # Add - \beta * (y - x_i)
        cc[:, 1:][:, nt_idxs] += beta
        cc = np.hstack((cc, -beta_max))  # (n_c, d+2)

        return cc

    @classmethod
    def _construct_dlp(
        cls,
        v: ndarray,  # (m, d)
        nt_idxs: list[int],
    ) -> ndarray:
        raise RuntimeError("This method should not be called.")

    @classmethod
    def _cal_mn_constrs_with_one_y(
        cls,
        idx: int,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        dlp_lines: ndarray,  # (2, d+1)
        dlp_point: float,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:  # (n, d+1) , (m, d+1)
        raise RuntimeError("This method should not be called.")
