__docformat__ = ["restructuredtext"]
__all__ = ["ELUHullWithOneY"]

from ._relulike import ReLULikeHullWithOneY
from ..acthull import ELUHull


class ELUHullWithOneY(ReLULikeHullWithOneY, ELUHull):
    """
    The class to calculate the function hull for the exponential linear unit (ELU)
    activation function with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`ELUHull` for more
    details
    """

    pass
