__docformat__ = "restructuredtext"
__all__ = ["ActHullWithOneY"]

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy import ndarray

from wraact._constants import TOLERANCE
from wraact._exceptions import DegeneratedError
from wraact.acthull import ActHull


class ActHullWithOneY(ActHull, ABC):
    """
    An object used to calculate the convex hull of the activation function with only extending one output dimension.

    We only need several output constraints which have big beta values, which provide the important multi-neuron constraints.

    :param dtype_cdd: The data type used in pycddlib library.
    :param n_output_constraints: The number of output constraints.
    """

    __slots__ = [*ActHull.__slots__, "_n_output_constrs"]
    _n_output_constrs: int

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
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
    ) -> ndarray | None:  # (_, 2*d-1) | (_, d+1)
        if c is None:  # pragma: no cover - defensive check, validated by caller in cal_hull
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
            v, dtype_cdd = self._cal_vertices_with_exception(c, lb, ub, self.dtype_cdd)
            new_lb = np.min(v, axis=0)[1:]
            new_ub = np.max(v, axis=0)[1:]
            self._check_degenerated_input_polytope(v, new_lb, new_ub)
            lb = new_lb
            ub = new_ub
        except DegeneratedError:
            v, dtype_cdd = self.cal_vertices(c, "fraction")
            lb = np.min(v, axis=0)[1:]
            ub = np.max(v, axis=0)[1:]
        except Exception as e:
            raise e

        # Update input bounds constraints
        d = lb.shape[0]
        c[-2 * d : -d, 0] = -lb
        c[-d:, 0] = ub

        result = self._cal_constrs_with_exception(c, v, lb, ub, dtype_cdd)
        if result is None:  # pragma: no cover - defensive check, method never returns None
            raise RuntimeError("Expected non-None result from _cal_constrs_with_exception")
        cc, dtype_cdd = result

        # ====================CHECK====================
        # Check if all vertices satisfy the constraints.
        # v_y = self._f(v[:, 1:])
        # vertices = np.hstack((v, v_y))
        # check = np.matmul(cc, vertices.T)
        # if not np.all(check >= -TOLERANCE):
        #     raise RuntimeError("Not all vertices satisfy the constraints.")

        if self._use_double_orders:  # pragma: no cover - always False in OneY (set in __init__)
            # Here we reverse the order of input dimensions to calculate the function
            # hull because our algorithm is a progressive algorithm that calculates the
            # function hull of the output dimensions one by one.
            # Computing with reversed input dimension order can improve precision.
            o_r = self._get_reversed_order(
                c.shape[1] - 1
            )  # Use input constraint dimensions, not output
            c_r = c.copy()  # Reversed constraints
            c_r = c_r[:, o_r]
            result_r = self._cal_constrs_with_exception(c_r, v, lb, ub, dtype_cdd)
            if result_r is None:  # pragma: no cover - defensive check, method never returns None
                raise RuntimeError("Expected non-None result from _cal_constrs_with_exception")
            cc_r, dtype_cdd = result_r
            # Reverse the output dimensions back to match original order
            d_out = cc.shape[1] - 1
            o_r_output = self._get_reversed_order(d_out)
            cc_r = cc_r[:, o_r_output]
            cc = np.vstack((cc, cc_r))

        return cc

    @classmethod
    @abstractmethod
    def cal_mn_constrs(  # type: ignore[override]
        cls,
        c: ndarray,  # (_, d)
        v: ndarray,  # (_, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
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
        c = c[(c[:, -1] < -TOLERANCE) | (c[:, -1] > TOLERANCE)]

        # c = c[np.argsort(-np.abs(c[:, 0] / c[:, -1]))]
        c = c[np.argsort(c[:, -1])]

        # Get the topk maximum or minimum beta values.
        if is_min:
            c = c[:topk]
        else:
            c = c[-topk:]

        return c
