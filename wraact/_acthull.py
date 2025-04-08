__docformat__ = "restructuredtext"
__all__ = [
    "ActHull",
    "ReLULikeHull",
    "SShapeHull",
    "ReLUHull",
    "LeakyReLUHull",
    "ELUHull",
    "SiLUHull",
    "SigmoidHull",
    "TanhHull",
    "MaxPoolHull",
    "MaxPoolHullDLP",
]

import os
import time
from abc import ABC, abstractmethod
from typing import Literal

import cdd
import numpy as np
from numpy import ndarray

from ._exceptions import *
from ._functions import *

_TOL = 1e-4
_LEAKY_RELU_ALPHA = 0.01
_MIN_BOUNDS_RANGE = 0.05
_MIN_DLP_ANGLE = 0.1
"""
 The minimum angle between two lines of the DLP function.
 Given two slopes m1 and m2, the angle between them is calculated by
 arctan(abs((m1 - m2) / (1 + m1 * m2))), but we only use abs((m1 - m2) / (1 + m1 * m2))
 to estimate.
"""

_ELU_MAX_AUX_POINT = -1.25
_DEBUG = False


class ActHull(ABC):
    """
    An object used to calculate the function hull of the activation
    function.

    :param if_cal_single_neuron_constrs: Whether to calculate single-neuron
        constraints.
    :param if_cal_multi_neuron_constrs: Whether to calculate multi-neuron
        constraints.

    .. tip::
        The multi-neuron constraints here means those constraints that cannot obtained
        by trivial methods or the properties of the activation function.

    :param if_use_double_orders: Whether to calculate the function hull of the double
        orders of input variables.

    .. attention::
        When enabled, it cost more time and generate (almost double) constraints.
        There is an improvement for ReLU functions but not very useful for other
        activation functions.

    :param if_return_input_bounds_by_vertices: Whether to return the lower and
        upper bounds of the input variables by the vertices of the input polytope.
        This means that the input constraints and given lower and upper bounds can
        create a more tight input constraints and result in new lower and upper bounds
        of the input variables.

    .. tip::
        This is introduced for the multi-variable maxpool activation function
        because the maxpool function is always followed by an activation function and
        this results in some very different behaviors.

    :param dtype_cdd: The data type used in pycddlib library.

    .. tip::
        Even though the precision is important when calculating the function hull,
        we suggest using "float" instead of "fraction" because the calculation is faster
        and can be accepted in most cases. If there is a numerical error, we will raise
        an exception and use "fraction" to recalculate the function hull.
    """

    _reversed_orders: dict[int, list[int]] = {}
    """This is a cache for the reversed orders of the input variables. The key is the
    dimension of the input space and the value is the reversed order of the input 
    variable indices."""

    __slots__ = [
        "_add_sn_constrs",
        "_add_mn_constrs",
        "_use_double_orders",
        "_dtype_cdd",
    ]

    def __init__(
        self,
        if_cal_single_neuron_constrs: bool = False,
        if_cal_multi_neuron_constrs: bool = True,
        if_use_double_orders: bool = False,
        dtype_cdd: Literal["float", "fraction"] = "float",
        if_return_input_bounds_by_vertices: bool = False,
    ):
        if if_use_double_orders and not if_cal_multi_neuron_constrs:
            raise ValueError(
                "if_use_double_orders should be True if if_cal_multi_neuron_constrs is "
                "True because the double orders are calculated based on the "
                "multi-neuron constraints to improve precision."
            )
        if not if_cal_single_neuron_constrs and not if_cal_multi_neuron_constrs:
            raise ValueError(
                "At least one of if_cal_single_neuron_constrs and "
                "if_cal_multi_neuron_constrs should be True."
            )

        self._add_sn_constrs = if_cal_single_neuron_constrs
        self._add_mn_constrs = if_cal_multi_neuron_constrs
        self._use_double_orders = if_use_double_orders

        self._dtype_cdd = dtype_cdd

    def cal_hull(
        self,
        input_constrs: ndarray | None = None,  # (n, d)
        input_lower_bounds: ndarray | None = None,  # (d-1,)
        input_upper_bounds: ndarray | None = None,  # (d-1,)
    ) -> ndarray | None:  # (n, 2*d-1) | (n, d+1)
        """
        Calculate the function hull of an activation function.
        There are two options:

        1. Calculate the single-neuron constraints with given input lower and upper
           bounds. (Two arguments: input_lower_bounds and input_upper_bounds)
        2. Calculate the multi-neuron constraints with given input constraints and
           input lower and upper bounds. (Three arguments: input_constrs,
           lower_bounds, and upper_bounds). Some functions require the lower and
           upper bounds of the input variables, and we suggest to provide them.

        .. tip::

            The input bounds are used to generate a set of constraints that are
            consistent with the input bounds.

        .. tip::
            The datatye of numpy array is float64 in this function to ensure the
            precision of the calculation.


        :param input_constrs: The input constraints.
        :param input_lower_bounds: The lower bounds of the input variables.
        :param input_upper_bounds: The upper bounds of the input variables.
        :return: The constraints defining the function hull.
        """

        self._check_input_bounds(input_lower_bounds, input_upper_bounds)
        self._check_input_constrs(input_constrs)
        self._check_inputs(input_constrs, input_lower_bounds, input_upper_bounds)

        d = None
        l = u = None
        c_i = c_l = c_u = None
        # Convert the data type to float64 to ensure the precision of the calculation.
        # Make a copy to avoid changing the original data.
        if input_constrs is not None:
            c_i = np.array(input_constrs, dtype=np.float64)
            d = c_i.shape[1] - 1

        if input_lower_bounds is not None:
            l = np.array(input_lower_bounds, dtype=np.float64)
            c_l = self._build_input_bounds_constraints(l, is_lower=True)
            d = l.size

        if input_upper_bounds is not None:
            u = np.array(input_upper_bounds, dtype=np.float64)
            c_u = self._build_input_bounds_constraints(u, is_lower=False)
            d = u.size

        c = np.empty((0, 1 + d), dtype=np.float64)
        if c_i is not None:
            c = np.vstack((c, c_i))
        if c_l is not None:
            c = np.vstack((c, c_l))
        if c_u is not None:
            c = np.vstack((c, c_u))
        c = np.ascontiguousarray(c)

        if self._add_sn_constrs and not self._add_mn_constrs:
            return self._cal_hull_with_sn_constrs(l, u)

        elif self._add_mn_constrs:
            return self._cal_hull_with_mn_constrs(c, l, u)

    @staticmethod
    def _build_input_bounds_constraints(s: ndarray, is_lower: bool = True) -> ndarray:
        """
        Build the constraints based on the lower or upper bounds of the input variables.

        :param s: The lower or upper bounds of the input variables.

        :return: The constraints based on the lower or upper bounds of the input
            variables.
        """
        n = s.size

        c = np.zeros((n, n + 1), dtype=s.dtype)
        c[:, 0] = -s if is_lower else s
        idx_row = np.arange(n)
        idx_col = np.arange(1, n + 1)
        c[idx_row, idx_col] = 1.0 if is_lower else -1.0

        return c

    @staticmethod
    def cal_vertices(
        c: ndarray,  # (n, d)
        dtype_cdd: Literal["float", "fraction"],
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (m, d)
        """
        Calculate the vertices of a polytope from the constraints.

        .. attention::
            The datatype of cdd is important because the precision may cause an error
            when calculating the vertices. Sometimes float number is not enough to
            calculate the vertices, and we need to use the fractional number to
            calculate the vertices.

        .. tip::
            The result of the vertices may have repeated vertices, which is rooted in
            the algorithm of the pycddlib library.
            Considering removing the repeated vertices is not necessary, we just keep
            the repeated vertices, and it is not efficient due to the large number of
            vertices

        :param c: The constraints of the polytope.
        :param dtype_cdd: The data type used in pycddlib library.

        :return: The vertices of the polytope.
        """
        h_repr = cdd.Matrix(c, number_type=dtype_cdd)
        h_repr.rep_type = cdd.RepType.INEQUALITY

        p = cdd.Polyhedron(h_repr)
        v_repr = p.get_generators()
        v = np.asarray(v_repr, dtype=np.float64)

        return v, dtype_cdd

    def _cal_hull_with_sn_constrs(
        self,
        l: ndarray | None,  # (d,)
        u: ndarray | None,  # (d,)
    ) -> ndarray:  # (_, 2*d+1) | (_, d+1)
        if l is None or u is None:
            raise ValueError(
                "The lower and upper bounds of the input variables should be "
                "provided."
            )

        return self.cal_sn_constrs(l, u)

    def _cal_hull_with_mn_constrs(
        self,
        c: ndarray,  # (n, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
    ) -> (
        ndarray  # (_, 2*d-1) | (_, d+1)
        | None
        | tuple[
            ndarray | None,  # (_, 2*d-1) | (_, d+1)
            ndarray,  # (d-1,)
            ndarray,  # (d-1,)
        ]
    ):
        if c is None:
            raise ValueError("The input constraints should be provided.")

        try:
            """
            The bounds need update if we use update scalar bounds per layer of
            DeepPoly. This will cause degenrated input polytope.

            There are two cases:
            (1) One of the input dimension has the same lower and upper bounds, which
            will throw a Degenerated exception.
            (2) The number of vertices is fewer than the dimension, which will call
            a Degenerated exception.

            We will first recalculate the vertices with the fractional number if there
            is an exception. If there is still an exception, we will accept the
            degenerated input polytope.
            """
            v, dtype_cdd = self._cal_vertices_with_exception(c, l, u, self.dtype_cdd)
            new_l = np.min(v, axis=0)[1:]
            new_u = np.max(v, axis=0)[1:]
            self._check_degenerated_input_polytope(v, new_l, new_u)
            l = new_l
            u = new_u
        except Degenerated:
            v, dtype_cdd = self.cal_vertices(c, "fraction")
            l = np.min(v, axis=0)[1:]
            u = np.max(v, axis=0)[1:]
        except Exception as e:
            raise e

        if np.min(np.abs(u - l)) < _MIN_BOUNDS_RANGE and len(v) > 2:
            # We don't want to remove trivial cases for the maxpool function (one vertex
            # and one piece).
            return None

        # Update input bounds constraints
        d = l.shape[0]
        c[-2 * d : -d, 0] = -l  # noqa: E203
        c[-d:, 0] = u

        cc, dtype_cdd = self._cal_constrs_with_exception(c, v, l, u, dtype_cdd)

        # ====================CHECK====================
        # Check if all vertices satisfy the constraints.
        # v_y = self._f(v[:, 1:])
        # vertices = np.hstack((v, v_y))
        # check = np.matmul(cc, vertices.T)
        # if not np.all(check >= -_TOL):
        #     raise RuntimeError("Not all vertices satisfy the constraints.")

        if self._use_double_orders:
            # Here we reverse the order of output dimensions to calculate the function
            # hull because our algorithm is a prgressive algorithm that calculates the
            # function hull of the output dimensions one by one.
            o_r = ActHull._get_reversed_order(cc.shape[1] - 1)
            c_r = c.copy()  # Reversed constraints
            c_r = c_r[:, o_r]
            cc_r, dtype_cdd = self._cal_constrs_with_exception(c_r, v, l, u, dtype_cdd)
            cc_r = cc_r[:, o_r]
            cc = np.vstack((cc, cc_r))

        return cc

    @staticmethod
    def _get_reversed_order(d: int) -> list[int]:
        # The reversed order of the output dimensions is cached to avoid calculating
        # it multiple times.
        if ActHull._reversed_orders.get(d) is None:
            ActHull._reversed_orders[d] = [0] + list(range(d, 0, -1))
        return ActHull._reversed_orders[d]

    def _cal_vertices_with_exception(
        self,
        c: ndarray,  # (n, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]] | None:  # (m, d)
        if _DEBUG:
            # When debugging, we directly calculate the vertices and check the
            # correctness without exception handling to see the error message.
            v, dtype_cdd = self.cal_vertices(c, dtype_cdd)
            self._check_vertices(v)
            return v, dtype_cdd

        v = None
        try:
            # Maybe a bug caused by float number and the fractional number will be used.
            v, dtype_cdd = self.cal_vertices(c, dtype_cdd)
            self._check_vertices(v)
            return v, dtype_cdd
        except Exception:  # noqa
            try:
                # Change to use the fractional number to calculate the vertices.
                dtype_cdd = "fraction"
                v, dtype_cdd = self.cal_vertices(c, dtype_cdd)  # type: ignore
                self._check_vertices(v)

                return v, dtype_cdd
            except Exception as e:
                # This happens when there is an unexpected error.
                self._record_and_raise_exception(e, c, v, l, u)

    def _cal_constrs_with_exception(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        l: ndarray | None = None,  # (d-1,)
        u: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> (
        tuple[
            ndarray,  # (n, 2*d-1) | (n, d+1)
            Literal["float", "fraction"],
        ]
        | None
    ):
        if _DEBUG:
            # When debugging, we directly calculate the constraints and check the
            # correctness without exception handling to see the error message.
            output_constrs, dtype_cdd = self.cal_constrs(c, v, l, u, dtype_cdd)
            return output_constrs, dtype_cdd

        try:
            # Maybe a bug caused by float number and the fractional number will be used.
            output_constrs, dtype_cdd = self.cal_constrs(c, v, l, u, dtype_cdd)
            return output_constrs, dtype_cdd

        except Exception:  # noqa
            try:
                output_constrs, dtype_cdd = self.cal_constrs(c, v, l, u, "fraction")
                return output_constrs, dtype_cdd

            except Exception as e:
                # Normally, there should not be any error.
                # For debugging, we check and record the error.
                self._record_and_raise_exception(e, c, v, l, u)

    @abstractmethod
    def cal_constrs(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[
        ndarray,  # (_, 2*d-1) | (_, d+1)
        Literal["float", "fraction"],
    ]:
        """
        Calculate the function hull of the activation function with a single order of
        input variables.

        :param c: The constraints of the input polytope.
        :param v: The vertices of the input polytope.
        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.
        :param dtype_cdd: The data type used in pycddlib library.

        :return: The constraints defining the function hull.
        """
        pass

    @classmethod
    @abstractmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (_, 2*d+1) | (_, d+1)
        """
        Calculate the single-neuron constraints of the function hull.

        .. tip::
            The single-neuron constraints can be calculated directly from the input
            lower and upper bounds because they only consider one neuron.

        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.
        """
        pass

    @classmethod
    @abstractmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
    ) -> ndarray:  # (_, 2*d-1) | (_, d+1)
        """
        Calculate the multi-neuron constraints of the function hull.

        .. tip::
            The multi-neuron constraints are calculated based on the input constraints
            and vertices. The lower and upper bounds of the input variables are used to
            check the correctness of the input constraints and vertices. Specifically,
            we can get the lower and upper bounds of the calculated vertices and check
            whether they are consistent with the given input bounds.

        :param c: The constraints of the input polytope.
        :param v: The vertices of the input polytope.
        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.

        :return: The constraints defining the function hull.
        """
        pass

    @classmethod
    @abstractmethod
    def _cal_mn_constrs_with_one_y(cls, *args, **kwargs):
        """Calculate the multi-neuron constraint with extending one output dimension."""
        pass

    @classmethod
    @abstractmethod
    def _construct_dlp(cls, *args, **kwargs):
        """Construct a double-linear-piece (DLP) function as the lower or upper bound of
        the activation function."""
        pass

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        """The activation function."""
        pass

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        """The derivative of the activation function."""
        pass

    @staticmethod
    def _check_inputs(c: ndarray | None, l: ndarray | None, u: ndarray | None):
        if c is not None and l is not None and u is not None:
            if not c.shape[1] - 1 == l.size == u.size:
                raise ValueError(
                    "The dimensions of the input constraints, lower bounds, and upper "
                    f"bounds should be the same but {c.shape[1] - 1}, {l.size}, and "
                    f"{u.size} are provided."
                )
        elif c is None and l is None and u is None:
            raise ValueError(
                "At least the input constraints, or lower bounds and upper bounds "
                "should be provided."
            )

    @staticmethod
    def _check_input_constrs(c: ndarray | None):
        if c is not None:
            d = c.shape[1] - 1
            if c.shape[0] < d + 1:
                raise ValueError(
                    "The number of input constraints should be at least the dimension "
                    "of the input space plus one. Otherwise, the polytope is unbounded."
                    f"The shape of the input constraints is {c.shape}."
                )

    @staticmethod
    def _check_input_bounds(l: ndarray | None, u: ndarray | None):
        if l is not None and u is not None:
            if not l.ndim == u.ndim == 1:
                raise ValueError(
                    "The lower and upper bounds of the input variables should be "
                    f"1-dimensional arrays but {l.ndim} and "
                    f"{u.ndim} are provided."
                )

            if not l.size == u.size:
                raise ValueError(
                    "The lower and upper bounds of the input variables should have the "
                    f"same size but {l.size} and "
                    f"{u.size} are provided."
                )
            if not np.all(l <= u):
                raise ValueError(
                    "The lower bounds should be less than the upper bounds but "
                    f"{l} and {u} are provided."
                )

    @staticmethod
    def _check_vertices(v: ndarray):  # (m, d)

        if len(v) == 0:
            raise RuntimeError(
                "Zero vertices. The input polytope is infeasible. "
                "This should not happen and there is a bug in the code."
            )

        if np.any(v[:, 0] != 1.0):
            raise ArithmeticError(
                f"Unbounded polytope. The first column of the vertices should "
                f"be 1, which means the vertex is not a ray that is used to "
                f"define a unbounded polytope."
            )

    @staticmethod
    def _check_degenerated_input_polytope(v: ndarray, l: ndarray, u: ndarray):
        d = v.shape[1] - 1
        if len(v) < d + 1:
            raise Degenerated(
                f"The {d}-d input polytope should not be with only {len(v)} vertices."
            )
        if np.any(np.isclose(l, u)):
            raise Degenerated(
                f"The input polytope is degenerated because one of the input dimension "
                f"has the same lower and upper bounds."
            )

    def _record_and_raise_exception(
        self, e: Exception, c: ndarray, v: ndarray, l: ndarray | None, u: ndarray | None
    ):
        current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        os.makedirs(".temp", exist_ok=True)
        error_log = f".temp/acthull_{current_time}.log"
        with open(error_log, "w") as f:
            f.write(f"{self.__class__.__name__}\n")
            f.write(f"Exception: {e}\n")
            f.write(f"Created time: {current_time}\n")
            if c is not None:
                f.write(f"Input constraints shape: {c.shape}\n")
                f.write(f"Input constraints: {c.tolist()}\n")
            if l is not None:
                f.write(f"Input constraints lower bounds: {l.tolist()}\n")
            if u is not None:
                f.write(f"Input constraints upper bounds: {u.tolist()}\n")
            f.write(f"Input vertices shape: {v.shape}\n")
            if v is not None and len(v) > 0:
                v_l = np.min(v, axis=0)[1:]
                v_u = np.max(v, axis=0)[1:]
                f.write(f"Input vertices: {v.tolist()}\n")
                f.write(f"Input vertices lower bounds: {v_l.tolist()}\n")
                f.write(f"Input vertices upper bounds: {v_u.tolist()}\n")

        raise RuntimeError(f"Error: {e}. Please check the log: {error_log}")

    @property
    def dtype_cdd(self) -> Literal["float", "fraction"]:
        """The data type used in pycddlib library."""
        return self._dtype_cdd


def cal_mn_constrs_with_one_y_dlp(
    idx: int,
    c: ndarray,  # (n, d)
    v: ndarray,  # (m, d)
    aux_lines: ndarray,  # (_, d+1)
    aux_point: float | None,
    is_convex: bool = True,
) -> tuple[ndarray, ndarray]:  # (n, d+1) | (n+1, d+1) , (m, d+1)
    """
    Calculate the multi-neuron constraints for one specified input dimension of the
    function hull for the DLP (double linear pieces) function.

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
        raise RuntimeError(
            "The vertices should not all greater/smaller than the auxiliary point."
        )

    ll, lr = aux_lines[[[0], [1]]] if is_convex else aux_lines[[[1], [0]]]

    vl, vr = v[mask_vl], v[mask_vr]
    v[mask_vl, -1], v[mask_vr, -1] = np.matmul(vl, ll.T).T, np.matmul(vr, lr.T).T
    vl, vr = v[mask_vl], v[mask_vr]
    d1, d2 = np.matmul(c, vr.T), np.matmul(c, vl.T)
    h1, h2 = np.matmul(ll, vr.T), np.matmul(lr, vl.T)

    if np.any(h1 == 0) or np.any(h2 == 0):
        raise RuntimeError("Zero values will be in denominators.")

    # if is_convex:
    #     assert np.all(h1 <= 0), f"{h1}"
    #     assert np.all(h2 <= 0), f"{h2}"
    # else:
    #     assert np.all(h1 >= 0), f"{h1}"
    #     assert np.all(h2 >= 0), f"{h2}"

    d1 /= h1
    d2 /= h2

    if is_convex:
        beta1 = np.max(d1, axis=1, keepdims=True)
        beta2 = np.max(d2, axis=1, keepdims=True)
    else:
        beta1 = np.min(d1, axis=1, keepdims=True)
        beta2 = np.min(d2, axis=1, keepdims=True)

    # beta1 = np.maximum(beta1, 0)
    # beta2 = np.maximum(beta2, 0)

    c -= beta1 * ll
    c -= beta2 * lr

    # assert np.all(c @ vl.T >= -_TOL)

    return c, v


class ReLULikeHull(ActHull, ABC):
    """
    This is the base class for the ReLU-like activation functions to calculate the
    function hull.
    """

    def cal_constrs(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, 2*d-1)

        d = c.shape[1] - 1
        c = np.array(c, dtype=np.float64)
        l = np.array(l, dtype=np.float64)
        u = np.array(u, dtype=np.float64)
        cc = np.empty((0, 1 + 2 * d), dtype=np.float64)

        if self._add_sn_constrs:
            c1 = self.cal_sn_constrs(l, u)
            cc = np.vstack((cc, c1))

        if self._add_mn_constrs:
            c2 = self.cal_mn_constrs(c, v, l, u)
            cc = np.vstack((cc, c2))

        return cc, dtype_cdd

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
    ) -> ndarray:  # (_, 2*d-1)
        d = c.shape[1] - 1

        for i in range(d):
            lines, point = cls._construct_dlp(i, d, l[i], u[i])
            c, v = cls._cal_mn_constrs_with_one_y(i, c, v, lines, point, True)

        return c

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
        return cal_mn_constrs_with_one_y_dlp(
            idx, c, v, dlp_lines, dlp_point, is_convex=is_convex
        )

    @classmethod
    @abstractmethod
    def _construct_dlp(cls, *args, **kwargs):
        pass


class ReLUHull(ReLULikeHull):
    """
    This is to calculate the function hull for the rectified linear unit (ReLU)
    activation function.

    .. tip::
        This is an ad hoc implementation for ReLU to obtain the function hull
        considering high efficiency and accuracy based on the two linear pieces (
        :math:`y=x` and :math:`y=0`) of ReLU.
    """

    _lower_constraints: dict[int, ndarray] = {}

    @classmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (3*d, 1+2*d)
        """
        Calculate the single-neuron constraints of the function hull for ReLU.
        We use *triangle relaxation* to calculate the single-neuron constraints of
        ReLU.

        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.
        :return: The single-neuron constraints of the function hull.
        """
        if np.any(l >= 0) or np.any(u <= 0):
            raise ValueError(
                "The lower bounds should be negative and the upper bounds should be "
                "positive because we only handle the non-trivial cases for ReLU."
            )

        d = l.shape[0]
        c = np.zeros((d, 2 * d + 1), dtype=np.float64)

        idx_r = np.arange(d)  # The index of the rows
        idx_x = np.arange(1, d + 1)  # The index of the input variables
        idx_y = np.arange(d + 1, 2 * d + 1)  # The index of the output variables

        # For the upper faces.
        # The output constraints have the form of -u*l + u*x - (u-l)*y >= 0.
        c[:, 0] = -u * l
        c[idx_r, idx_x] = u
        c[idx_r, idx_y] = -(u - l)

        # For the lower faces.
        if cls._lower_constraints.get(d) is None:
            # The output constraints have the form of y >= 0.
            c_l1 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            c_l1[idx_r, idx_y] = 1.0

            # The output constraints have the form of y >= x.
            c_l2 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            c_l2[idx_r, idx_x] = -1.0
            c_l2[idx_r, idx_y] = 1.0

            cl = np.vstack((c_l1, c_l2))
            cls._lower_constraints[d] = cl
        else:
            cl = cls._lower_constraints[d]

        c = np.vstack((c, cl))

        return c

    @classmethod
    def cal_mn_constrs(
        cls,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
    ) -> ndarray:  # (_, 2*d-1)
        """
        Calculate the multi-neuron constraints of the function hull.
        We use the algorithm called *WraLU* to calculate the multi-neuron
        constraints

        .. seealso::

            Refer to the paper:
            `ReLU Hull Approximation <https://dl.acm.org/doi/pdf/10.1145/3632917>`__
            :cite:`ma_relu_2024`

        :param c: The constraints of the input polytope.
        :param v: The vertices of the input polytope.
        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.

        :return: The multi-neuron constraints of the function hull.
        """
        # FIXME: Add the code for degenerated input polytope.

        dim = c.shape[1] - 1

        v = np.transpose(v)
        mask_xp, mask_xn = (v > _TOL), (v < -_TOL)
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

            # Eliminate tiny positive values
            # beta1 = np.minimum(beta1, 0)
            # beta2 = np.minimum(beta2, 0)

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
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        dlp_lines: ndarray,  # (2, d+1)
        dlp_point: float,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:  # (n, d+1) , (m, d+1)
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


class LeakyReLUHull(ReLULikeHull):
    """
    This is to calculate the function hull for the leaky rectified linear unit
    (LeakyReLU) activation function.
    """

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
    def _construct_dlp(
        cls, idx: int, dim: int, l: float, u: float
    ) -> tuple[ndarray, float | None]:
        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)

        if l >= 0:
            aux_lines = np.asarray(
                [[0.0] + temp1 + [1.0] + temp2 + [-1.0]], dtype=np.float64
            )
            return aux_lines, None

        if u <= 0:
            aux_lines = np.asarray(
                [[0.0] + temp1 + [_LEAKY_RELU_ALPHA] + temp2 + [-1.0]], dtype=np.float64
            )
            return aux_lines, None

        if u - l < _MIN_BOUNDS_RANGE:
            k = (u - l) / (u - l)
            b = u - k * u
            aux_lines = np.asarray(
                [[b] + temp1 + [k] + temp2 + [-1.0]], dtype=np.float64
            )
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


class ELUHull(ReLULikeHull):
    """
    This is to calculate the function hull for the exponential linear unit (ELU)
    activation function.
    """

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
    def _construct_dlp(
        cls, idx: int, dim: int, l: float, u: float
    ) -> tuple[ndarray, float | None]:

        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)

        yl = cls._f(l)
        yu = cls._f(u)
        if l >= _ELU_MAX_AUX_POINT or u <= _ELU_MAX_AUX_POINT:
            k = (yu - yl) / (u - l)
            b = yu - k * u
            aux_lines = np.asarray(
                [[b] + temp1 + [k] + temp2 + [-1.0]], dtype=np.float64
            )
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
            aux_lines = np.asarray(
                [[b] + temp1 + [k] + temp2 + [-1.0]], dtype=np.float64
            )
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


class SiLUHull(ReLULikeHull):
    """
    This is to calculate the function hull for the sigmoid linear unit (SiLU)

    .. warning::
        This class is not implemented yet.

    """

    def __int__(self):
        raise NotImplementedError

    @classmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (_, 1+2*d)
        raise NotImplementedError

    @classmethod
    def _construct_dlp(
        cls, idx: int, dim: int, l: float, u: float
    ) -> tuple[ndarray, float]:  # (2, 1+{dim}+{idx}+1)
        temp1, temp2 = [0.0] * idx, [0.0] * (dim - 1)
        kp1 = cls._f(l) / l
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
        return silu_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dsilu_np(x)


class SShapeHull(ActHull, ABC):
    """
    This is the base class for the S-shaped activation functions to calculate the
    function hull.

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
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, 2*d-1)

        d = c.shape[1] - 1
        c = np.array(c, dtype=np.float64)
        l = np.array(l, dtype=np.float64)
        u = np.array(u, dtype=np.float64)
        cc = np.empty((0, 2 * d + 1), dtype=np.float64)

        if self._add_sn_constrs and not self._add_mn_constrs:
            c_s = self.cal_sn_constrs(l, u)
            cc = np.vstack((cc, c_s))

        if self._add_mn_constrs:
            c_m = self.cal_mn_constrs(c, v, l, u)
            cc = np.vstack((cc, c_m))

        return cc, dtype_cdd

    def cal_sn_constrs(
        self,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (_, 1+2*d)

        d = l.shape[0]
        cc = np.empty((0, 1 + d), dtype=np.float64)

        f, df = self._f, self._df
        xl, xu = l, u
        yl, yu, kl, ku = f(xl), f(xu), df(xl), df(xu)
        klu = (yu - yl) / (xu - xl)

        for i in range(d):
            args = (i, d, xl[i], xu[i], yl[i], yu[i], kl[i], ku[i], klu[i], cc)
            _, _, _, _, cc = self._construct_dlp(*args, self._add_sn_constrs)

        return cc

    def cal_mn_constrs(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
    ) -> ndarray:  # (_, 2*d-1) | (_, d+1)

        if l is None and u is None:
            raise ValueError(
                "The lower and upper bounds should be provided for the S-shape "
                "activation function."
            )

        d = c.shape[1] - 1
        # The single-neuron constraints
        cc_s = np.empty((0, 1 + d), dtype=np.float64)
        # The multi-neuron constraints providing lower/upper output bounds
        cc_l, cc_u = c, c.copy()
        v_l, v_u = v, v.copy()

        f, df = self._f, self._df
        xl, xu = l, u
        yl, yu, kl, ku = f(xl), f(xu), df(xl), df(xu)
        klu = (yu - yl) / (xu - xl)

        for i in range(d):
            args = (i, d, xl[i], xu[i], yl[i], yu[i], kl[i], ku[i], klu[i], cc_s)
            dlp_lines_l, dlp_lines_u, dlp_point_l, dlp_point_u, cc_s = (
                self._construct_dlp(*args, self._add_sn_constrs)
            )

            if self._add_mn_constrs:
                cc_l, v_l = self._cal_mn_constrs_with_one_y(
                    i, cc_l, v_l, dlp_lines_l, dlp_point_l, is_convex=False
                )
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
        dlp_point: float | None,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:  # (n, d+1) | (n+1, d+1), (m, d+1)
        return cal_mn_constrs_with_one_y_dlp(
            idx, c, v, dlp_lines, dlp_point, is_convex=is_convex
        )

    @classmethod
    def _construct_dlp(
        cls,
        idx: int,
        dim: int,
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,  # (n, d)
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray | None, ndarray | None, float | None, float | None, ndarray | None
    ]:
        """
        Calculate the auxiliary lines, auxiliary point, and the single-neuron
        constraints.
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
        elif kui > klui:
            resolve_case = cls._construct_dlp_case1
        elif kli > klui:
            resolve_case = cls._construct_dlp_case2
        else:
            resolve_case = cls._construct_dlp_case3

        c = np.hstack((c, np.zeros((c.shape[0], 1))))
        args = (idx, dim, xli, xui, yli, yui, kli, kui, klui, c)
        return resolve_case(*args, return_single_neuron_constrs)

    @classmethod
    def _construct_dlp_case1(
        cls,
        idx: int,
        dim: int,
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,  # (n, d)
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,  # (1, 1+dim+idx+1)
        ndarray,  # (2, 1+dim+idx+1)
        None,
        float,
        ndarray | None,  # (n+4, 1+dim+idx+1)
    ]:
        """
        Calculate the auxiliary lines, auxiliary point, and the single-neuron
        constraints for the case where the slope of the upper linear piece is
        larger than the slope of the linear piece connecting the lower and upper
        bounds.

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
            blui, bp2, bui = b
            klui, kp2, kui = k
        else:
            kp2, klui, kui = (yui - f(su)) / (xui - su), klui, kui
            bp2, blui, bui = yui - kp2 * xui, yli - klui * xli, yui - kui * xui

        aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_l[:, 0] = [bli]
        aux_lines_l[:, idx + 1] = [kli]
        aux_lines_l[:, -1] = -1.0
        aux_point_l = None

        aux_lines_u, aux_point_u = cls._construct_upper_aux_lines_and_points(
            dim, idx, xli, yli, klui, su, kp1, kp2, bp1, bp2
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
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,  # (n, d)
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,  # (2, 1+dim+idx+1)
        ndarray,  # (1, 1+dim+idx+1)
        float,
        None,
        ndarray | None,  # (n+4, 1+dim+idx+1)
    ]:
        """
        Calculate the auxiliary lines, auxiliary point, and the single-neuron
        constraints for the case where the slope of the lower linear piece is
        larger than the slope of the linear piece connecting the lower and upper
        bounds.

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
            blui, bp2, bli = b
            klui, kp2, kli = k
        else:
            kp2, klui, kli = (yli - f(sl)) / (xli - sl), klui, kli
            bp2, blui, bli = yli - kp2 * xli, yui - klui * xui, yli - kli * xli

        aux_lines_l, aux_point_l = cls._construct_lower_aux_lines_and_points(
            dim, idx, xui, yui, klui, sl, kp1, kp2, bp1, bp2
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
        xli: float,
        xui: float,
        yli: float,
        yui: float,
        kli: float,
        kui: float,
        klui: float,
        c: ndarray,  # (n, d)
        return_single_neuron_constrs: bool,
    ) -> tuple[
        ndarray,  # (2, 1+dim+idx+1)
        ndarray,  # (2, 1+dim+idx+1)
        float,
        float,
        ndarray | None,  # (n+6, 1+dim+idx+1)
    ]:
        """
        Calculate the auxiliary lines, auxiliary point, and the single-neuron
        constraints for the case where (1) the slope of the upper linear piece is
        smaller than the slope of the linear piece connecting the lower and upper
        bounds, and (2) the slope of the lower linear piece is smaller than the
        slope of the linear piece connecting the lower and upper bounds.

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
        btu, bp2u = b_temp
        ktu, kp2u = k_temp
        x_temp = np.asarray([xui, sl], dtype=np.float64)
        b_temp, k_temp, _ = cls._get_second_tangent_line(x_temp, get_big=False)
        btl, bp2l = b_temp
        ktl, kp2l = k_temp

        kp1u, kp1l = (yli - f(su)) / (xli - su), (yui - f(sl)) / (xui - sl)
        bp1u, bp1l = yli - kp1u * xli, yui - kp1l * xui

        aux_lines_l, aux_point_l = cls._construct_lower_aux_lines_and_points(
            dim, idx, xui, yui, klui, sl, kp1l, kp2l, bp1l, bp2l
        )
        aux_lines_u, aux_point_u = cls._construct_upper_aux_lines_and_points(
            dim, idx, xli, yli, klui, su, kp1u, kp2u, bp1u, bp2u
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
        xui: float,
        yui: float,
        klui: float,
        sl: float,
        kp1l: float,
        kp2l: float,
        bp1l: float,
        bp2l: float,
    ) -> tuple[ndarray, float | None]:  # (2, {dim}+{idx}+2)
        if abs((kp1l - kp2l) / (1 - kp1l * kp2l)) < _MIN_DLP_ANGLE:
            aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [yui - klui * xui]
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
        xli: float,
        yli: float,
        klui: float,
        su: float,
        kp1u: float,
        kp2u: float,
        bp1u: float,
        bp2u: float,
    ) -> tuple[ndarray, float | None]:  # (2, {dim}+{idx}+2)
        if abs((kp1u - kp2u) / (1 - kp1u * kp2u)) < _MIN_DLP_ANGLE:
            aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [yli - klui * xli]
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
        Get the second tangent line given a point x1, which is not the tangent line
        taking x1 as tangent point.

        :param x1: The point where the tangent line is not taken.
        :param get_big: Whether to get the tangent line with a larger slope.

        :return: The bias, slope, and the tangent point of the tangent line.
        """

        pass

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
        pass


class SigmoidHull(SShapeHull):
    """
    This is to calculate the function hull for the sigmoid activation function.

    Please refer to the :class:`SShapeHull` for more details.
    """

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        from ._tangent_lines import get_second_tangent_line_sigmoid_np

        return get_second_tangent_line_sigmoid_np(x1, get_big)

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        from ._tangent_lines import get_parallel_tangent_line_sigmoid_np

        return get_parallel_tangent_line_sigmoid_np(k, get_big)

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return sigmoid_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dsigmoid_np(x)


class TanhHull(SShapeHull):
    """
    This is to calculate the function hull for the hyperbolic tangent activation
    function.

    Please refer to the :class:`SShapeHull` for more details.
    """

    @staticmethod
    def _get_second_tangent_line(
        x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        from ._tangent_lines import get_second_tangent_line_tanh_np

        return get_second_tangent_line_tanh_np(x1, get_big)

    @staticmethod
    def _get_parallel_tangent_line(
        k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        from _tangent_lines import get_parallel_tangent_line_tanh_np

        return get_parallel_tangent_line_tanh_np(k, get_big)

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        return tanh_np(x)

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        return dtanh_np(x)


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

    _lower_constraints: dict[int, ndarray] = {}

    def cal_constrs(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (_, d+1)

        d = c.shape[1] - 1
        c = np.array(c, dtype=np.float64)
        l = np.array(l, dtype=np.float64)
        u = np.array(u, dtype=np.float64)
        cc = np.empty((0, d + 2), dtype=np.float64)

        if self._add_sn_constrs:
            c1 = self.cal_sn_constrs(l, u)
            cc = np.vstack((cc, c1))

        if self._add_mn_constrs:
            c2 = self.cal_mn_constrs(c, v, l, u)
            cc = np.vstack((cc, c2))

        return cc, dtype_cdd

    @classmethod
    def cal_sn_constrs(
        cls,
        l: ndarray,  # (d,)
        u: ndarray,  # (d,)
    ) -> ndarray:  # (d+1, d+2)
        """
        Calculate the single-neuron constraints for the MaxPool function.

        .. seealso::
            Refer to the paper
            `Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks
            <https://arxiv.org/pdf/1705.01320>`__ :cite:`ehlers_formal_2017`
            for specific constraints and theoretical proof.

        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.

        :return: The single-neuron constraints.
        """
        d = l.shape[0]
        c_u = np.zeros((1, d + 2), dtype=l.dtype)

        # Upper bounds
        # (1) The following solution is not precise enough.
        # l_sum = np.sum(l)
        # l_max = np.max(l)
        # c_u[-1, 0] = l_max - l_sum
        # c_u[-1, 1:-1] = 1.0
        # c_u[-1, -1] = -1.0
        # (2) y < u_max
        c_u[-1, 0] = np.max(u)
        c_u[-1, -1] = -1.0

        # Lower bound
        # Here we do not remove the redundant constraints consider the trivial case,
        # because there are a few lower constraints. So we decide to keep them.
        if cls._lower_constraints.get(d) is not None:
            c_l = cls._lower_constraints[d]
        else:
            # y >= x_i for all i
            c_l = np.zeros((d, d + 2), dtype=l.dtype)
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
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
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
        if not np.all(Axb >= -_TOL):
            raise RuntimeError(f"Negative beta.\nAxb={Axb}.")
        # Axb = np.maximum(Axb, 0.0)  # Remove tiny negative values

        # Calculate y - each piece
        v_y = np.max(pv, 0, keepdims=True)  # (1, n_v)
        yx = v_y - pv  # (2, n_v)
        yx = np.expand_dims(yx, 0)  # (1, 2, n_v)
        # Calculate (Ax + b) / (y - \sum x_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(yx != 0, Axb / yx, np.inf)  # (n_c, 2, n_v)
        if not np.all(beta >= -_TOL):
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
        l = np.min(v[:, 1:], axis=0)
        u = np.max(v[:, 1:], axis=0)

        r = u - l
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
        return RuntimeError("This method should not be called.")

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
        l: ndarray | None,  # (d-1,)
        u: ndarray | None,  # (d-1,)
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
        if not np.all(Axb >= -_TOL):
            raise RuntimeError(f"Negative beta.\nAxb={Axb}.")
        Axb = np.maximum(Axb, 0.0)  # Remove tiny negative values

        # Calculate y - x_i
        v_y = np.max(v[:, 1:], 1, keepdims=True)  # (n_v, 1)
        yx = v_y - v[:, 1:][:, nt_idxs]  # (n_v, d')
        yx = np.expand_dims(yx.T, 0)  # (1, d', n_v)

        # Calculate (Ax + b) / (y - x_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(yx != 0, Axb / yx, np.inf)  # (n_c, d', n_v)

        if not np.all(beta >= -_TOL):
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
