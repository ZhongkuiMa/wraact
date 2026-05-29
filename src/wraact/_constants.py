"""Constants used throughout the wraact package.

This module defines numerical tolerances, bounds ranges, and default
parameter values used in activation hull computations.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "DEBUG",
    "ELU_MAX_AUX_POINT",
    "LEAKY_RELU_ALPHA",
    "MIN_BOUNDS_RANGE_ACTHULL",
    "MIN_BOUNDS_RANGE_ONEY",
    "MIN_DLP_ANGLE",
    "MIN_DLP_DENOM",
    "TOLERANCE",
]

#: Enable debug mode for additional logging and checks.
DEBUG: bool = False

#: Tolerance for numerical comparisons and constraint satisfaction checks.
#: Values smaller than this are considered zero.
TOLERANCE: float = 1e-4

#: Minimum range between lower and upper bounds for acthull module.
#: Bounds closer than this may cause numerical instability.
MIN_BOUNDS_RANGE_ACTHULL: float = 0.05

#: Minimum range between lower and upper bounds for oney module.
#: Slightly tighter than acthull version for single-output optimization.
MIN_BOUNDS_RANGE_ONEY: float = 0.04

#: Minimum angle in radians between two DLP function pieces.
#: Prevents degenerate DLP constructions.
MIN_DLP_ANGLE: float = 0.1

#: Minimum absolute denominator threshold for DLP beta computation.
#: Values below this trigger a RuntimeError to prevent numerical blowup.
MIN_DLP_DENOM: float = 1e-12

#: Default negative slope coefficient for LeakyReLU activation.
LEAKY_RELU_ALPHA: float = 0.01

#: Maximum auxiliary point x-coordinate for ELU DLP construction.
#: Used to define the transition region in ELU piecewise approximation.
ELU_MAX_AUX_POINT: float = -1.25
