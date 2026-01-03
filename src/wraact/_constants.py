"""Constants used throughout the wraact package."""

__docformat__ = "restructuredtext"
__all__ = [
    "DEBUG",
    "ELU_MAX_AUX_POINT",
    "LEAKY_RELU_ALPHA",
    "MIN_BOUNDS_RANGE_ACTHULL",
    "MIN_BOUNDS_RANGE_ONEY",
    "MIN_DLP_ANGLE",
    "TOLERANCE",
]

# Debugging flag
DEBUG: bool = False

# Tolerance for numerical comparisons
TOLERANCE: float = 1e-4

# Minimum range for bounds - acthull version
MIN_BOUNDS_RANGE_ACTHULL: float = 0.05

# Minimum range for bounds - oney version (slightly tighter)
MIN_BOUNDS_RANGE_ONEY: float = 0.04

# Minimum angle between two DLP function pieces (in radians)
MIN_DLP_ANGLE: float = 0.1

# LeakyReLU slope coefficient
LEAKY_RELU_ALPHA: float = 0.01

# Maximum auxiliary point for ELU DLP construction
ELU_MAX_AUX_POINT: float = -1.25
