"""Base class for activation function convex hull computation."""

__docformat__ = "restructuredtext"
__all__ = ["ActHull"]

import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal, NoReturn

import cdd
import numpy as np
from numpy import ndarray

from wraact._constants import DEBUG, MIN_BOUNDS_RANGE_ACTHULL
from wraact._exceptions import DegeneratedError, NotConvergedError

# cdd.Error inherits from object, not BaseException (Python 3.11+ cannot
# catch non-BaseException types in except clauses). It is kept here as a
# safety gate: we catch it at a wider RuntimeError level instead.
_CDD_ERRORS = (
    RuntimeError,
    ArithmeticError,
    AttributeError,
    ValueError,
    NotConvergedError,
)


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

    __slots__ = (
        "_dtype_cdd",
        "_if_cal_mn_constrs",
        "_if_cal_sn_constrs",
        "_use_double_orders",
    )

    _if_cal_sn_constrs: bool
    _if_cal_mn_constrs: bool
    _use_double_orders: bool
    _dtype_cdd: Literal["float", "fraction"]

    def __init__(
        self,
        if_cal_single_neuron_constrs: bool = False,
        if_cal_multi_neuron_constrs: bool = True,
        if_use_double_orders: bool = False,
        dtype_cdd: Literal["float", "fraction"] = "float",
    ):
        """Initialize the activation hull calculator.

        :param if_cal_single_neuron_constrs: Whether to calculate single-neuron
            constraints.
        :param if_cal_multi_neuron_constrs: Whether to calculate multi-neuron
            constraints.
        :param if_use_double_orders: Whether to use reversed input dimension
            order for improved precision.
        :param dtype_cdd: Data type for pycddlib library.
        :raises ValueError: If parameter combination is invalid.
        """
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

        self._if_cal_sn_constrs = if_cal_single_neuron_constrs
        self._if_cal_mn_constrs = if_cal_multi_neuron_constrs
        self._use_double_orders = if_use_double_orders

        self._dtype_cdd = dtype_cdd

    def cal_hull(
        self,
        input_constrs: ndarray | None = None,
        input_lower_bounds: ndarray | None = None,
        input_upper_bounds: ndarray | None = None,
    ) -> ndarray | None:
        """
        Calculate the function hull of an activation function.

        There are two usage modes:

        1. **Single-neuron mode**: Provide only ``input_lower_bounds`` and
           ``input_upper_bounds``. Calculates constraints directly from bounds.
           Requires ``if_cal_single_neuron_constrs=True`` in constructor.

        2. **Multi-neuron mode**: Provide ``input_constrs`` with optionally
           ``input_lower_bounds`` and ``input_upper_bounds``. Calculates more
           sophisticated constraints considering polytope geometry.
           Requires ``if_cal_multi_neuron_constrs=True`` in constructor (default).

        .. tip::
            The datatype of numpy array is float64 in this function to ensure the
            precision of the calculation.

        :param input_constrs: Input polytope constraints in H-representation.
            Shape: ``(n, d+1)`` where ``n`` = number of constraints, ``d`` = input dimension.
            Format: ``[b | A]`` where each row represents ``b + A @ x >= 0``.
            The first column is the bias ``b``, remaining columns are coefficients ``A``.
            Example for a 2D box [0,1] x [0,1]::

                input_constrs = np.array([
                    [0,   1,  0],   # 0 + 1*x1 + 0*x2 >= 0  =>  x1 >= 0
                    [1,  -1,  0],   # 1 - 1*x1 + 0*x2 >= 0  =>  x1 <= 1
                    [0,   0,  1],   # 0 + 0*x1 + 1*x2 >= 0  =>  x2 >= 0
                    [1,   0, -1],   # 1 + 0*x1 - 1*x2 >= 0  =>  x2 <= 1
                ], dtype=np.float64)

        :param input_lower_bounds: Lower bounds for each input variable.
            Shape: ``(d,)`` where ``d`` = input dimension.
            Example: ``np.array([-1.0, -1.0])`` for 2D input with lower bounds -1.

        :param input_upper_bounds: Upper bounds for each input variable.
            Shape: ``(d,)`` where ``d`` = input dimension.
            Example: ``np.array([1.0, 1.0])`` for 2D input with upper bounds 1.

        :return: Constraint matrix in H-representation defining the function hull.
            Shape: ``(num_constraints, 2*d+1)`` for standard hulls.
            Format: ``[b | A_x | A_y]`` where ``b + A_x @ x + A_y @ y >= 0``.
            The first column is the bias, next ``d`` columns are input coefficients,
            last ``d`` columns are output coefficients.
            Returns ``None`` if only single-neuron constraints are requested without
            multi-neuron constraints.

        :raises ValueError: If parameters are invalid or bounds don't match dimensions.
        :raises DegeneratedError: If the input polytope is degenerate (too few vertices).

        Example::

            from wraact import ReLUHull
            import numpy as np

            hull = ReLUHull()
            lb = np.array([-1.0, -1.0], dtype=np.float64)
            ub = np.array([1.0, 1.0], dtype=np.float64)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # constraints.shape = (num_constraints, 5)  # [b, x1, x2, y1, y2]
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
        if self._if_cal_mn_constrs:
            return self._cal_hull_with_mn_constrs(c, lb, ub)
        return self._cal_hull_with_sn_constrs(lb, ub)

    @staticmethod
    def _build_input_bounds_constraints(bounds: ndarray, is_lower: bool = True) -> ndarray:
        """
        Build the constraints based on the lower or upper bounds of the input variables.

        :param bounds: The lower or upper bounds of the input variables.

        :return: The constraints based on the lower or upper bounds of the input
            variables.
        """
        n = bounds.size

        c = np.zeros((n, n + 1), dtype=bounds.dtype)
        c[:, 0] = -bounds if is_lower else bounds
        idx_row = np.arange(n)
        idx_col = np.arange(1, n + 1)
        c[idx_row, idx_col] = 1.0 if is_lower else -1.0

        return c

    @staticmethod
    def cal_vertices(
        c: ndarray,
        dtype_cdd: Literal["float", "fraction"],
    ) -> tuple[ndarray, Literal["float", "fraction"]]:
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
        if dtype_cdd == "float":
            h_repr = cdd.matrix_from_array(c.tolist(), rep_type=cdd.RepType.INEQUALITY)
            p = cdd.polyhedron_from_matrix(h_repr)
            v_repr = cdd.copy_generators(p)
            v = np.array(v_repr.array, dtype=np.float64)
            return v, dtype_cdd

        from fractions import Fraction

        import cdd.gmp as _cdd  # type: ignore[import-untyped]

        c_frac = [[Fraction(x) for x in row] for row in c.tolist()]
        h_repr = _cdd.matrix_from_array(c_frac, rep_type=cdd.RepType.INEQUALITY)  # type: ignore  # noqa: PGH003
        p = _cdd.polyhedron_from_matrix(h_repr)  # type: ignore  # noqa: PGH003
        v_repr = _cdd.copy_generators(p)  # type: ignore  # noqa: PGH003
        v = np.array(v_repr.array, dtype=np.float64)
        return v, dtype_cdd

    def _cal_hull_with_sn_constrs(
        self,
        lb: ndarray | None,
        ub: ndarray | None,
    ) -> ndarray:
        """Compute hull using only single-neuron constraints from bounds.

        :param lb: Lower bounds per input dimension.
        :param ub: Upper bounds per input dimension.
        :return: Single-neuron hull constraints.
        :raises ValueError: If bounds are not provided.
        """
        if lb is None or ub is None:
            raise ValueError(
                "The lower and upper bounds of the input variables should be provided."
            )

        return self.cal_sn_constrs(lb, ub)

    def _compute_vertices_and_update_bounds(
        self,
        c: ndarray,
        lb: ndarray | None,
        ub: ndarray | None,
    ) -> tuple[ndarray, ndarray | None, ndarray | None, Literal["float", "fraction"]]:
        """Compute polytope vertices and update bounds from them.

        Shared logic extracted from ``_cal_hull_with_mn_constrs`` to avoid
        duplication between ``ActHull`` and ``ActHullWithOneY``.

        :param c: Input constraints. Shape: ``n, d``.
        :param lb: Lower bounds per dimension.
        :param ub: Upper bounds per dimension.
        :return: Tuple of (vertices, updated_lb, updated_ub, dtype_cdd_used).
        """
        try:
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
        return v, lb, ub, dtype_cdd

    def _cal_hull_with_mn_constrs(
        self,
        c: ndarray,
        lb: ndarray | None = None,
        ub: ndarray | None = None,
    ) -> ndarray | None:
        """Compute hull using multi-neuron constraints from input polytope.

        Computes vertices from the input constraints, updates bounds, and
        generates sound over-approximation constraints. Falls back to
        fractional arithmetic on degenerate polytopes.

        :param c: Input constraints in H-representation. Shape: ``n, d``.
        :param lb: Lower bounds per dimension. Shape: ``d-1,``.
        :param ub: Upper bounds per dimension. Shape: ``d-1,``.
        :return: Multi-neuron hull constraints, or ``None`` on failure.
        :raises ValueError: If input constraints are not provided or
            polytope is too small.
        """
        if c is None:  # pragma: no cover - defensive check, validated by caller in cal_hull
            raise ValueError("The input constraints should be provided.")

        # Track whether explicit bounds rows were appended so we only update those rows.
        had_bounds = lb is not None and ub is not None

        v, lb, ub, dtype_cdd = self._compute_vertices_and_update_bounds(c, lb, ub)

        if lb is None or ub is None:
            raise ValueError("Bounds became None after vertex computation.")

        min_range = np.min(np.abs(ub - lb))
        if min_range < MIN_BOUNDS_RANGE_ACTHULL and len(v) > 2:
            raise ValueError(
                f"Polytope too small: minimum range {min_range:.6f} < "
                f"threshold {MIN_BOUNDS_RANGE_ACTHULL}. Cannot compute reliable constraints."
            )

        # Update input bounds constraints only when bound rows were appended to c.
        if had_bounds and lb is not None and ub is not None:
            d = lb.shape[0]
            c[-2 * d : -d, 0] = -lb
            c[-d:, 0] = ub

        result = self._cal_constrs_with_exception(c, v, lb, ub, dtype_cdd)
        if result is None:
            raise RuntimeError("Expected non-None result from _cal_constrs_with_exception")
        cc, dtype_cdd = result

        if self._use_double_orders:
            o_r = ActHull._get_reversed_order(c.shape[1] - 1)
            c_r = c.copy()
            c_r = c_r[:, o_r]
            result_r = self._cal_constrs_with_exception(c_r, v, lb, ub, dtype_cdd)
            if result_r is None:
                raise RuntimeError("Expected non-None result from _cal_constrs_with_exception")
            cc_r, dtype_cdd = result_r
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

    @staticmethod
    def _cdd_retry(
        fn: Callable[..., tuple[ndarray, Literal["float", "fraction"]]],
        fallback_args: tuple,
        fallback_kwargs: dict,
        error_ctx: dict,
    ) -> tuple[ndarray, Literal["float", "fraction"]]:
        """Try CDD operation with float; fall back to fraction on error.

        :param fn: CDD operation (cal_vertices or cal_constrs).
        :param fallback_args: Positional args for fn (excluding dtype_cdd).
        :param fallback_kwargs: Keyword args for fn.
        :param error_ctx: Context dict for _record_and_raise_exception.
        :return: (result, dtype_cdd_used).
        :raises RuntimeError: If both float and fraction fail.
        """
        if DEBUG:
            result, dtype_cdd = fn(*fallback_args, **fallback_kwargs)
            return result, dtype_cdd

        try:
            result, dtype_cdd = fn(*fallback_args, **fallback_kwargs)
        except _CDD_ERRORS:
            try:
                result, dtype_cdd = fn(
                    *fallback_args, **{**fallback_kwargs, "dtype_cdd": "fraction"}
                )
            except _CDD_ERRORS as e:
                ActHull._record_and_raise_exception(e, **error_ctx)
        return result, dtype_cdd

    def _cal_vertices_with_exception(
        self,
        c: ndarray,
        lb: ndarray | None = None,
        ub: ndarray | None = None,
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[ndarray, Literal["float", "fraction"]]:
        """Compute polytope vertices with automatic fallback to fractional arithmetic.

        :param c: Input constraints. Shape: ``n, d``.
        :param lb: Lower bounds per dimension.
        :param ub: Upper bounds per dimension.
        :param dtype_cdd: Initial data type for pycddlib.
        :return: Tuple of (vertices, dtype_cdd used).
        :raises RuntimeError: If vertex computation fails with both float
            and fractional arithmetic.
        """

        def _run_and_check(dtype: str) -> tuple[ndarray, str]:
            v, _ = self.cal_vertices(c, dtype)  # type: ignore[arg-type]
            self._check_vertices(v)
            return v, dtype

        if DEBUG:
            return _run_and_check(dtype_cdd)  # type: ignore[return-value]

        try:
            return _run_and_check(dtype_cdd)  # type: ignore[return-value]
        except _CDD_ERRORS:
            try:
                return _run_and_check("fraction")  # type: ignore[return-value]
            except _CDD_ERRORS as e:
                self._record_and_raise_exception(e, c, None, lb, ub)
        # unreachable
        raise RuntimeError("Vertex computation failed.")  # pragma: no cover

    def _cal_constrs_with_exception(
        self,
        c: ndarray,
        v: ndarray,
        lb: ndarray | None = None,
        ub: ndarray | None = None,
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> (
        tuple[
            ndarray,
            Literal["float", "fraction"],
        ]
        | None
    ):
        """Compute hull constraints with automatic fallback to fractional arithmetic.

        :param c: Input constraints. Shape: ``n, d``.
        :param v: Vertices of input polytope. Shape: ``m, d``.
        :param lb: Lower bounds per dimension.
        :param ub: Upper bounds per dimension.
        :param dtype_cdd: Initial data type for pycddlib.
        :return: Tuple of (hull constraints, dtype_cdd used), or ``None``.
        :raises RuntimeError: If constraint computation fails with both float
            and fractional arithmetic.
        """
        output_constrs, dtype_cdd = ActHull._cdd_retry(
            self.cal_constrs,
            (c, v, lb, ub),
            {"dtype_cdd": dtype_cdd},
            {"c": c, "v": v, "l": lb, "u": ub},
        )
        return output_constrs, dtype_cdd

    @abstractmethod
    def cal_constrs(
        self,
        c: ndarray,
        v: ndarray,
        lb: ndarray | None,
        ub: ndarray | None,
        dtype_cdd: Literal["float", "fraction"] = "float",
    ) -> tuple[
        ndarray,
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
        lb: ndarray,
        ub: ndarray,
    ) -> ndarray:
        """
        Calculate the single-neuron constraints of the function hull.

        .. tip::
            The single-neuron constraints can be calculated directly from the input
            lower and upper bounds because they only consider one neuron.

        :param lb: The lower bounds of the input variables.
        :param ub: The upper bounds of the input variables.

        :return: The single-neuron constraints of the function hull.
        """

    @classmethod
    @abstractmethod
    def cal_mn_constrs(  # type: ignore[override]
        cls,
        c: ndarray,
        v: ndarray,
        l: ndarray | None,
        u: ndarray | None,
    ) -> ndarray:
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

    @staticmethod
    def _f(x: ndarray | float) -> ndarray | float:
        """Compute the activation function."""
        raise NotImplementedError(
            "This ActHull subclass does not implement _f. "
            "Only differentiable activation hulls need this method."
        )

    @staticmethod
    def _df(x: ndarray | float) -> ndarray | float:
        """Compute the derivative of the activation function."""
        raise NotImplementedError(
            "This ActHull subclass does not implement _df. "
            "Only differentiable activation hulls need this method."
        )

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

            if np.any(np.isnan(l)) or np.any(np.isnan(u)):
                raise ValueError("The lower and upper bounds contain NaN values.")
            if np.any(np.isinf(l)) or np.any(np.isinf(u)):
                raise ValueError("The lower and upper bounds contain Inf values.")

            if not np.all(l <= u):
                raise ValueError(
                    "The lower bounds should be less than the upper bounds but "
                    f"{l} and {u} are provided."
                )

    @staticmethod
    def _check_vertices(v: ndarray):
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

    @classmethod
    def _record_and_raise_exception(
        cls, e: Exception, c: ndarray, v: ndarray | None, l: ndarray | None, u: ndarray | None
    ) -> NoReturn:
        """Log error details to a file and re-raise as RuntimeError.

        :param e: The original exception.
        :param c: Input constraints at the time of failure.
        :param v: Vertices at the time of failure, or ``None``.
        :param l: Lower bounds, or ``None``.
        :param u: Upper bounds, or ``None``.
        :raises RuntimeError: Always raised with path to the error log file.
        """
        current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        log_dir = Path(tempfile.gettempdir()) / "wraact_errors"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            error_log = str(log_dir / f"acthull_{current_time}.log")
        except OSError:
            raise RuntimeError(f"Error: {e}") from e
        with Path(error_log).open("w") as f:
            f.write(f"{cls.__name__}\n")
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
