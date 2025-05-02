__docformat__ = ["restructuredtext"]
__all__ = ["SigmoidHullWithOneY"]

from ._sshape import SShapeHullWithOneY
from ..acthull import SigmoidHull


class SigmoidHullWithOneY(SShapeHullWithOneY, SigmoidHull):
    """
    The class used to calculate the convex hull of the sigmoid activation function with
    only extending one output dimension.

    Please refer to the :class:`SShapeHullWithOneY` and :class:`SigmoidHull` for more
    details.
    """

    pass
