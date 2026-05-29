"""WRAACT: Precise Activation Function Over-Approximation for Neural Network Verification.

This package computes tight convex hulls for neural network activation
functions (ReLU, Sigmoid, Tanh, ELU, LeakyReLU, MaxPool) with sound
over-approximation guarantees for abstract interpretation during verification.

Main classes:
    - ActHull: Base class for activation hull computation
    - ReLUHull, SigmoidHull, TanhHull, ELUHull, LeakyReLUHull: Standard hulls
    - MaxPoolHull, MaxPoolHullDLP: Max pooling hulls
    - *WithOneY variants: Optimized single-output versions
"""

__docformat__ = "restructuredtext"
__version__ = "2026.5.4"

from wraact._enums import TopKSelector
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
    "ELUHull",
    "ELUHullWithOneY",
    "LeakyReLUHull",
    "LeakyReLUHullWithOneY",
    "MaxPoolHull",
    "MaxPoolHullDLP",
    "MaxPoolHullDLPWithOneY",
    "MaxPoolHullWithOneY",
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
