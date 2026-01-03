__docformat__ = "restructuredtext"
__all__ = ["ActHull"]

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal, NoReturn

import cdd
import numpy as np
from numpy import ndarray

from wraact._constants import DEBUG, MIN_BOUNDS_RANGE_ACTHULL
from wraact._exceptions import DegeneratedError


class ActHull(ABC):
    """
    An object used to calculate the function hull of the activation function.

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

    _reversed_orders: ClassVar[dict[int, list[int]]] = {}
    """This is a cache for the reversed orders of the input variables. The key is the
    dimension of the input space and the value is the reversed order of the input
    variable indices."""

    __slots__ = [
        "_add_mn_constrs",
        "_add_sn_constrs",
        "_dtype_cdd",
        "_use_double_orders",
    ]

    _add_sn_constrs: bool
    _add_mn_constrs: bool
    _use_double_orders: bool
    _dtype_cdd: Literal["float", "fraction"]

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
        lb = ub = None
        c_i = c_l = c_u = None
        # Convert the data type to float64 to ensure the precision of the calculation.
        # Make a copy to avoid changing the original data.
        if input_constrs is not None:
            c_i = np.array(input_constrs, dtype=np.float64)
            d = c_i.shape[1] - 1

        if input_lower_bounds is not None:
            lb = np.array(input_lower_bounds, dtype=np.float64)
            c_l = self._build_input_bounds_constraints(lb, is_lower=True)
            d = lb.size

        if input_upper_bounds is not None:
            ub = np.array(input_upper_bounds, dtype=np.float64)
            c_u = self._build_input_bounds_constraints(ub, is_lower=False)
            d = ub.size

        assert d is not None, (
            "At least one of input_constrs, input_lower_bounds, or input_upper_bounds must be provided"
        )
        c = np.empty((0, 1 + d), dtype=np.float64)
        if c_i is not None:
            c = np.vstack((c, c_i))
        if c_l is not None:
            c = np.vstack((c, c_l))
        if c_u is not None:
            c = np.vstack((c, c_u))
        c = np.ascontiguousarray(c)

        if self._add_mn_constrs:
            return self._cal_hull_with_mn_constrs(c, lb, ub)
        return self._cal_hull_with_sn_constrs(lb, ub)

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
        lb: ndarray | None,  # (d,)
        ub: ndarray | None,  # (d,)
    ) -> ndarray:  # (_, 2*d+1) | (_, d+1)
        if lb is None or ub is None:
            raise ValueError(
                "The lower and upper bounds of the input variables should be provided."
            )

        return self.cal_sn_constrs(lb, ub)

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

        if np.min(np.abs(ub - lb)) < MIN_BOUNDS_RANGE_ACTHULL and len(v) > 2:
            # We don't want to remove trivial cases for the maxpool function (one vertex
            # and one piece).
            min_range = np.min(np.abs(ub - lb))
            raise ValueError(
                f"Polytope too small: minimum range {min_range:.6f} < "
                f"threshold {MIN_BOUNDS_RANGE_ACTHULL}. Cannot compute reliable constraints."
            )

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
        # if not np.all(check >= -_TOL):
        #     raise RuntimeError("Not all vertices satisfy the constraints.")

        if self._use_double_orders:
            # Here we reverse the order of input dimensions to calculate the function
            # hull because our algorithm is a progressive algorithm that calculates the
            # function hull of the output dimensions one by one.
            # Computing with reversed input dimension order can improve precision.
            o_r = ActHull._get_reversed_order(
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
            o_r_output = ActHull._get_reversed_order(d_out)
            cc_r = cc_r[:, o_r_output]
            cc = np.vstack((cc, cc_r))

        return cc

    @staticmethod
    def _get_reversed_order(d: int) -> list[int]:
        # The reversed order of the output dimensions is cached to avoid calculating
        # it multiple times.
        if ActHull._reversed_orders.get(d) is None:
            ActHull._reversed_orders[d] = [0, *list(range(d, 0, -1))]
        return ActHull._reversed_orders[d]

    def _cal_vertices_with_exception(
        self,
        c: ndarray,  # (n, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:  # (m, d)
        v: ndarray | None = None
        if DEBUG:
            # When debugging, we directly calculate the vertices and check the
            # correctness without exception handling to see the error message.
            v, dtype_cdd = self.cal_vertices(c, dtype_cdd)
            self._check_vertices(v)
            return v, dtype_cdd
        try:
            # Maybe a bug caused by float number and the fractional number will be used.
            v, dtype_cdd = self.cal_vertices(c, dtype_cdd)
            self._check_vertices(v)

        except (cdd.Error, RuntimeError, ArithmeticError, ValueError):
            try:
                # Change to use the fractional number to calculate the vertices.
                dtype_cdd = "fraction"
                v, dtype_cdd = self.cal_vertices(c, dtype_cdd)
                self._check_vertices(v)

            except (cdd.Error, RuntimeError, ArithmeticError, ValueError) as e:
                # This happens when there is an unexpected error.
                self._record_and_raise_exception(e, c, v, lb, ub)

        assert v is not None
        return v, dtype_cdd

    def _cal_constrs_with_exception(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        lb: ndarray | None = None,  # (d-1,)
        ub: ndarray | None = None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> (
        tuple[
            ndarray,  # (n, 2*d-1) | (n, d+1)
            Literal["float", "fraction"],
        ]
        | None
    ):
        if DEBUG:
            # When debugging, we directly calculate the constraints and check the
            # correctness without exception handling to see the error message.
            output_constrs, dtype_cdd = self.cal_constrs(c, v, lb, ub, dtype_cdd)
            return output_constrs, dtype_cdd

        try:
            # Maybe a bug caused by float number and the fractional number will be used.
            output_constrs, dtype_cdd = self.cal_constrs(c, v, lb, ub, dtype_cdd)

        except (cdd.Error, RuntimeError, ArithmeticError, ValueError):
            try:
                output_constrs, dtype_cdd = self.cal_constrs(c, v, lb, ub, "fraction")

            except (cdd.Error, RuntimeError, ArithmeticError, ValueError) as e:
                # Normally, there should not be any error.
                # For debugging, we check and record the error.
                self._record_and_raise_exception(e, c, v, lb, ub)

        return output_constrs, dtype_cdd

    @abstractmethod
    def cal_constrs(
        self,
        c: ndarray,  # (n, d)
        v: ndarray,  # (m, d)
        lb: ndarray | None,  # (d-1,)
        ub: ndarray | None,  # (d-1,)
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[
        ndarray,  # (_, 2*d-1) | (_, d+1)
        Literal["float", "fraction"],
    ]:
        """
        Calculate the function hull of the activation function with a single order of input variables.

        :param c: The constraints of the input polytope.
        :param v: The vertices of the input polytope.
        :param lb: The lower bounds of the input variables.
        :param ub: The upper bounds of the input variables.
        :param dtype_cdd: The data type used in pycddlib library.

        :return: The constraints defining the function hull.
        """

    @classmethod
    @abstractmethod
    def cal_sn_constrs(  # type: ignore[override]
        cls,
        lb: ndarray,  # (d,)
        ub: ndarray,  # (d,)
    ) -> ndarray:  # (_, 2*d+1) | (_, d+1)
        """
        Calculate the single-neuron constraints of the function hull.

        .. tip::
            The single-neuron constraints can be calculated directly from the input
            lower and upper bounds because they only consider one neuron.

        :param lb: The lower bounds of the input variables.
        :param ub: The upper bounds of the input variables.
        """

    @classmethod
    @abstractmethod
    def cal_mn_constrs(  # type: ignore[override]
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

    @classmethod
    @abstractmethod
    def _cal_mn_constrs_with_one_y(cls, *args, **kwargs):
        """Calculate the multi-neuron constraint with extending one output dimension."""

    @classmethod
    @abstractmethod
    def _construct_dlp(cls, *args, **kwargs):
        """Construct a double-linear-piece (DLP) function as the lower or upper bound of the activation function."""

    @staticmethod
    @abstractmethod
    def _f(x: ndarray | float) -> ndarray | float:
        """Compute the activation function."""

    @staticmethod
    @abstractmethod
    def _df(x: ndarray | float) -> ndarray | float:
        """Compute the derivative of the activation function."""

    @staticmethod
    def _check_inputs(c: ndarray | None, lb: ndarray | None, ub: ndarray | None):
        if c is not None and lb is not None and ub is not None:
            if not c.shape[1] - 1 == lb.size == ub.size:
                raise ValueError(
                    "The dimensions of the input constraints, lower bounds, and upper "
                    f"bounds should be the same but {c.shape[1] - 1}, {lb.size}, and "
                    f"{ub.size} are provided."
                )
        elif c is None and lb is None and ub is None:
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
                "Unbounded polytope. The first column of the vertices should "
                "be 1, which means the vertex is not a ray that is used to "
                "define a unbounded polytope."
            )

    @staticmethod
    def _check_degenerated_input_polytope(v: ndarray, l: ndarray, u: ndarray):
        d = v.shape[1] - 1
        if len(v) < d + 1:
            raise DegeneratedError(
                f"The {d}-d input polytope should not be with only {len(v)} vertices."
            )
        if np.any(np.isclose(l, u)):
            raise DegeneratedError(
                "The input polytope is degenerated because one of the input dimension "
                "has the same lower and upper bounds."
            )

    def _record_and_raise_exception(
        self, e: Exception, c: ndarray, v: ndarray | None, l: ndarray | None, u: ndarray | None
    ) -> NoReturn:
        current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        Path(".temp").mkdir(parents=True, exist_ok=True)
        error_log = f".temp/acthull_{current_time}.log"
        with Path(error_log).open("w") as f:
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
            if v is not None:
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
