"""Enums for wraact configuration.

Strategy enums for selecting subsets of hull constraints when the full
set exceeds the caller's budget (``n_output_constraints``).
"""

__docformat__ = "restructuredtext"
__all__ = ["TopKSelector"]

from enum import StrEnum, unique


@unique
class TopKSelector(StrEnum):
    """Strategy for selecting top-k hull constraints in ``ActHullWithOneY``.

    The wraact one-y hull may emit dozens of multi-neuron constraints per
    activation; ``ActHullWithOneY._n_output_constrs`` caps the number kept.
    This enum picks the selection criterion. Each constraint has the form
    ``const + sum(input_coefs * x) + beta * y >= 0``; for ReLU
    ``beta <= 0`` after the clamp at :func:`wraact.oney._relu.cal_mn_constrs`.

    :cvar BETA_MIN: Sort ascending by beta; pick first k. Default. Selects
        constraints with the strongest output-direction tilt (most negative
        beta for ReLU).
    :cvar BETA_ABS_MAX: Sort descending by ``|beta|``; pick first k.
        Identical to ``BETA_MIN`` when beta is one-signed (ReLU upper);
        meaningful for S-shape activations where beta crosses zero.
    :cvar COEF_L1_MAX: Sort descending by ``sum(|input_coefs|)``; pick
        first k. Strongest multi-input leverage; intuition: constraints
        that mix many input neurons reach further through downstream
        linear layers.
    :cvar COEF_L1_MIN: Sort ascending by ``sum(|input_coefs|)``; pick
        first k. Closest-to-single-neuron form; degenerates toward
        adalin secant when input coefs concentrate on one neuron.
    :cvar FIRST: No sort; take first k after the non-zero-beta filter.
        Control baseline.
    :cvar RANDOM: Uniform random k after the filter. Uses a fresh
        :func:`numpy.random.default_rng` — non-deterministic across runs
        unless the caller seeds the global generator via
        :func:`numpy.random.seed`.
    """

    BETA_MIN = "beta_min"
    BETA_ABS_MAX = "beta_abs_max"
    COEF_L1_MAX = "coef_l1_max"
    COEF_L1_MIN = "coef_l1_min"
    FIRST = "first"
    RANDOM = "random"
