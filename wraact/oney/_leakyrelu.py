__docformat__ = ["restructuredtext"]
__all__ = ["LeakyReLUHullWithOneY"]

from ._relulike import ReLULikeHullWithOneY
from ..acthull import LeakyReLUHull


class LeakyReLUHullWithOneY(ReLULikeHullWithOneY, LeakyReLUHull):
    """
    The class to calculate the function hull for the leaky rectified linear unit (Leaky
    ReLU) activation function with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`LeakyReLUHull` for
    more details.
    """

    pass
