__docformat__ = "restructuredtext"
__all__ = ["LeakyReLUHullWithOneY"]

from wraact.acthull import LeakyReLUHull
from wraact.oney._relulike import ReLULikeHullWithOneY


class LeakyReLUHullWithOneY(ReLULikeHullWithOneY, LeakyReLUHull):
    """
    The class to calculate the function hull for the leaky rectified linear unit (Leaky ReLU) activation function with only one output dimension.

    Please refer to the :class:`ReLULikeHullWithOneY` and :class:`LeakyReLUHull` for more details.
    """
