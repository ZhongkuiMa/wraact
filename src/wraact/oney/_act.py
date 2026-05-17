"""Base class for single-output activation hull computation.

This module provides ActHullWithOneY, an optimized variant that extends
only one output dimension for faster multi-neuron constraint computation.
"""

__docformat__ = "restructuredtext"
__all__ = ["ActHullWithOneY"]

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy import ndarray

from wraact._constants import TOLERANCE
from wraact._enums import TopKSelector
from wraact._exceptions import DegeneratedError
from wraact.acthull import ActHull

_TOPK_RNG = np.random.default_rng(0)


class ActHullWithOneY(ActHull, ABC):
    """
    An object used to calculate the convex hull of the activation function with only extending one output dimension.

    We only need several output constraints which have big beta values, which provide the important multi-neuron constraints.

    :param dtype_cdd: The data type used in pycddlib library.
    :param n_output_constraints: The number of output constraints.
    """

    __slots__ = [*ActHull.__slots__, "_n_output_constrs", "_topk_selector"]
    _n_output_constrs: int
    _topk_selector: TopKSelector

    def __init__(
        self,
        dtype_cdd: Literal["fraction", "float"] = "float",
        n_output_constraints: int = 1,
        if_return_input_bounds_by_vertices: bool = False,
        topk_selector: TopKSelector = TopKSelector.BETA_MIN,
    ):
        """Initialize the single-output activation hull calculator.

        :param dtype_cdd: Data type for pycddlib. Default: "float".
        :param n_output_constraints: Number of output constraints to generate.
        :param if_return_input_bounds_by_vertices: Whether to return tightened
            input bounds derived from polytope vertices.
        :param topk_selector: Strategy for choosing ``n_output_constraints``
            constraints from the full hull output. See :class:`TopKSelector`.
        """
        super().__init__(
            if_cal_single_neuron_constrs=True,
            if_cal_multi_neuron_constrs=True,
            if_use_double_orders=False,
            if_return_input_bounds_by_vertices=if_return_input_bounds_by_vertices,
            dtype_cdd=dtype_cdd,
        )

        self._n_output_constrs = n_output_constraints
        self._topk_selector = topk_selector

    def _cal_hull_with_mn_constrs(
        self,
        c: ndarray,
        lb: ndarray | None = None,
        ub: ndarray | None = None,
    ) -> ndarray | None:
        """Compute hull with multi-neuron constraints for single output.

        Computes vertices from input constraints, updates bounds, and
        generates multi-neuron constraints extending one output dimension.

        :param c: Input constraints in H-representation. Shape: ``n, d``.
        :param lb: Lower bounds per dimension. Shape: ``d-1,``.
        :param ub: Upper bounds per dimension. Shape: ``d-1,``.
        :return: Hull constraints or None if degenerate.
        :raises ValueError: If input constraints are not provided.
        """
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

        # Update input bounds constraints
        d = lb.shape[0]
        c[-2 * d : -d, 0] = -lb
        c[-d:, 0] = ub

        result = self._cal_constrs_with_exception(c, v, lb, ub, dtype_cdd)
        if result is None:  # pragma: no cover - defensive check, method never returns None
            raise RuntimeError("Expected non-None result from _cal_constrs_with_exception")
        cc, dtype_cdd = result

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
        c: ndarray,
        v: ndarray,
        lb: ndarray | None = None,
        ub: ndarray | None = None,
        n_output_constrs: int = 1,
    ) -> ndarray:
        """Compute multi-neuron constraints for single output dimension.

        Abstract method to be implemented by subclasses.

        :param c: Input constraints. Shape: ``_, d``.
        :param v: Vertices. Shape: ``_, d``.
        :param lb: Lower bounds. Shape: ``d-1,``.
        :param ub: Upper bounds. Shape: ``d-1,``.
        :param n_output_constrs: Number of output constraints to generate.
        :return: Multi-neuron constraints. Shape: ``_, d+1``.
        """

    @staticmethod
    def _get_topk_constrs(
        c: ndarray,
        topk: int,
        is_min: bool = True,
        selector: TopKSelector = TopKSelector.BETA_MIN,
    ) -> ndarray:
        """Select top-k constraints from the hull output.

        Filters constraints with non-zero output coefficients (``|beta| >
        TOLERANCE``), then picks ``topk`` rows according to ``selector``.
        See :class:`TopKSelector` for strategy semantics.

        :param c: Constraints to filter. Shape: ``(n_constrs, d)``.
        :param topk: Number of constraints to return.
        :param is_min: When ``selector == BETA_MIN``: ``True`` returns the
            ``topk`` smallest beta rows, ``False`` returns the largest.
            Ignored by other selectors.
        :param selector: Selection strategy. Default ``BETA_MIN`` preserves
            the legacy behavior.
        :return: Selected constraints. Shape: ``(<= topk, d)``.
        """
        # Filter near-zero beta (constraints that don't bind the output).
        c = c[(c[:, -1] < -TOLERANCE) | (c[:, -1] > TOLERANCE)]
        if c.shape[0] == 0:
            return c

        if selector == TopKSelector.BETA_MIN:
            order = np.argsort(c[:, -1])
            return c[order[:topk]] if is_min else c[order[-topk:]]
        if selector == TopKSelector.BETA_ABS_MAX:
            order = np.argsort(-np.abs(c[:, -1]))
            return c[order[:topk]]
        if selector == TopKSelector.COEF_L1_MAX:
            l1 = np.abs(c[:, 1:-1]).sum(axis=1)
            order = np.argsort(-l1)
            return c[order[:topk]]
        if selector == TopKSelector.COEF_L1_MIN:
            l1 = np.abs(c[:, 1:-1]).sum(axis=1)
            order = np.argsort(l1)
            return c[order[:topk]]
        if selector == TopKSelector.FIRST:
            return c[: min(topk, c.shape[0])]
        if selector == TopKSelector.RANDOM:
            picked = _TOPK_RNG.permutation(c.shape[0])[:topk]
            return c[picked]
        raise ValueError(f"unknown topk selector {selector}")
