"""Custom exceptions for wraact hull computations."""

__docformat__ = "restructuredtext"
__all__ = ["DegeneratedError", "NotConvergedError"]


class DegeneratedError(Exception):
    """
    An exception for degenerated input polytope when calculating function hull.

    It means the number of vertices is fewer than the dimension.
    """

    def __init__(self, message="The polytope is degenerated."):
        """Initialize with an optional error message.

        :param message: The error message. Default: "The polytope is degenerated."
        """
        super().__init__(message)

    def __str__(self):
        """Return string representation."""
        return f"{self.__class__.__name__}: {super().__str__()}"


class NotConvergedError(Exception):
    """
    An exception for not converged calculation in an algorithm.

    :param message: The shown exception message.
    """

    def __init__(self, message="The calculation is not converged."):
        """Initialize with an optional error message.

        :param message: The error message. Default: "The calculation is not converged."
        """
        super().__init__(message)

    def __str__(self):
        """Return string representation."""
        return f"{self.__class__.__name__}: {super().__str__()}"
