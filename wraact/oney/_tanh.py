__docformat__ = ["restructuredtext"]
__all__ = ["TanhHullWithOneY"]

from ._sshape import SShapeHullWithOneY
from ..acthull import TanhHull


class TanhHullWithOneY(SShapeHullWithOneY, TanhHull):
    """
    The class used to calculate the convex hull of the hyperbolic tangent (tanh)
    activation function with only extending one output dimension.

    Please refer to the :class:`SShapeHullWithOneY` and :class:`TanhHull` for more
    details.
    """

    pass
