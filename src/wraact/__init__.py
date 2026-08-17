"""WRAACT: Precise Activation Function Over-Approximation for Neural Network Verification.

This package computes tight convex hulls for neural network activation
functions (ReLU, Sigmoid, Tanh, ELU, LeakyReLU, MaxPool) with sound
over-approximation guarantees for abstract interpretation during verification.

Main classes:
    - ActHull: Base class for activation hull computation
    - ReLUHull, SigmoidHull, TanhHull, ELUHull, LeakyReLUHull: Standard hulls
    - MaxPoolHull, MaxPoolHullDLP: Max pooling hulls
    - *WithOneY variants: Optimized single-output versions

.. seealso:: ``wraact/CONVENTIONS.md`` for code conventions.
"""

__docformat__ = "restructuredtext"
__version__ = "2026.8.0"

from wraact._enums import TopKSelector
from wraact._exceptions import DegeneratedError, NotConvergedError
from wraact.acthull import (
    ActHull,
    ELUHull,
    LeakyReLUHull,
    MaxPoolHull,
    MaxPoolHullDLP,
    ReLUHull,
    ReLULikeHull,
    SigmoidHull,
    SShapeHull,
    TanhHull,
    cal_mn_constrs_with_one_y_dlp,
)
from wraact.oney import (
    ActHullWithOneY,
    ELUHullWithOneY,
    LeakyReLUHullWithOneY,
    MaxPoolHullDLPWithOneY,
    MaxPoolHullWithOneY,
    ReLUHullWithOneY,
    ReLULikeHullWithOneY,
    SigmoidHullWithOneY,
    SShapeHullWithOneY,
    TanhHullWithOneY,
)

__all__ = [
    "ActHull",
    "ActHullWithOneY",
    "DegeneratedError",
    "ELUHull",
    "ELUHullWithOneY",
    "LeakyReLUHull",
    "LeakyReLUHullWithOneY",
    "MaxPoolHull",
    "MaxPoolHullDLP",
    "MaxPoolHullDLPWithOneY",
    "MaxPoolHullWithOneY",
    "NotConvergedError",
    "ReLUHull",
    "ReLUHullWithOneY",
    "ReLULikeHull",
    "ReLULikeHullWithOneY",
    "SShapeHull",
    "SShapeHullWithOneY",
    "SigmoidHull",
    "SigmoidHullWithOneY",
    "TanhHull",
    "TanhHullWithOneY",
    "TopKSelector",
    "__version__",
    "cal_mn_constrs_with_one_y_dlp",
]
