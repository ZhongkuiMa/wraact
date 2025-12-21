__docformat__ = "restructuredtext"
__all__ = ["ELUHullWithOneY"]

from wraact.wraact.acthull import ELUHull
from wraact.wraact.oney._relulike import ReLULikeHullWithOneY


class ELUHullWithOneY(ReLULikeHullWithOneY, ELUHull):
    """
    The class to calculate the function hull for the exponential linear unit (ELU) activation function with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`ELUHull` for more details.
    """
