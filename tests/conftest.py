"""Shared test configuration and fixtures for wraact tests.

Provides:
- Polytope fixtures (2D box, 3D octahedron, random polytopes up to 4D)
- Numerical tolerance constants
- Monte Carlo sampling parameters
- Pytest markers configuration
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Register custom markers for test selection."""
    config.addinivalue_line("markers", "slow: Slow-running performance tests")
    config.addinivalue_line("markers", "soundness: Soundness verification tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "requires_elina: Tests requiring ELINA library")


# ============================================================================
# TOLERANCE AND NUMERICAL PARAMETERS
# ============================================================================


@pytest.fixture
def tolerance():
    """Numerical tolerance for floating-point comparisons.

    Used for np.allclose() checks and constraint satisfaction verification.
    """
    return 1e-8


@pytest.fixture
def large_tolerance():
    """Larger tolerance for relaxed comparisons."""
    return 1e-6


@pytest.fixture
def monte_carlo_samples():
    """Provide number of samples for Monte Carlo soundness checks."""
    return 10000


@pytest.fixture
def soundness_satisfaction_threshold():
    """Minimum acceptable constraint satisfaction rate (percentage).

    A sound hull must satisfy ALL constraints for valid input points.
    Allow <0.1% numerical precision violations.
    """
    return 99.9  # percent


# ============================================================================
# SIMPLE POLYTOPE FIXTURES
# ============================================================================


@pytest.fixture
def simple_2d_box_constraints():
    """Provide simple 2D box polytope: [0, 1] x [0, 1].

    H-representation: 4 halfspace constraints
    - x >= 0      : [0, 1, 0]
    - x <= 1      : [-1, 1, 0]
    - y >= 0      : [0, 0, 1]
    - y <= 1      : [-1, 0, 1]

    Returns:
        np.ndarray: Shape (4, 3), H-representation constraints
    """
    return np.array(
        [
            [0, 1, 0],  # x >= 0
            [-1, 1, 0],  # x <= 1
            [0, 0, 1],  # y >= 0
            [-1, 0, 1],  # y <= 1
        ],
        dtype=np.float64,
    )


@pytest.fixture
def simple_2d_box_bounds():
    """Provide bounds for 2D box: [0, 1] x [0, 1].

    Returns:
        tuple: (lower_bounds, upper_bounds) for 2D input
    """
    lb = np.array([0.0, 0.0], dtype=np.float64)
    ub = np.array([1.0, 1.0], dtype=np.float64)
    return lb, ub


@pytest.fixture
def simple_3d_octahedron_constraints():
    """3D octahedron polytope (approximation).

    Octahedron with vertices at (±1, 0, 0), (0, ±1, 0), (0, 0, ±1).
    H-representation with 8 halfspace constraints.

    Returns:
        np.ndarray: Shape (8, 4), H-representation constraints
    """
    # Each constraint is of form: b + a1*x1 + a2*x2 + a3*x3 >= 0
    return np.array(
        [
            [1, 1, 0, 0],  # x >= -1
            [1, -1, 0, 0],  # x <= 1
            [1, 0, 1, 0],  # y >= -1
            [1, 0, -1, 0],  # y <= 1
            [1, 0, 0, 1],  # z >= -1
            [1, 0, 0, -1],  # z <= 1
            [1, 1, 1, 1],  # x + y + z <= 1
            [1, -1, -1, -1],  # -x - y - z <= 1
        ],
        dtype=np.float64,
    )


@pytest.fixture
def simple_3d_octahedron_bounds():
    """Bounds for 3D octahedron: [-1, 1]³ (loose bounds).

    Returns:
        tuple: (lower_bounds, upper_bounds) for 3D input
    """
    lb = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    ub = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return lb, ub


# ============================================================================
# DEGENERATE AND EDGE CASE POLYTOPE FIXTURES
# ============================================================================


@pytest.fixture
def tiny_polytope_2d():
    """2D tiny polytope: range = 0.04 < MIN_BOUNDS_RANGE (0.05).

    Should trigger ValueError when passed to cal_hull().
    This polytope has dimensions too small to compute meaningful constraints.

    Returns:
        tuple: (lower_bounds, upper_bounds) where min range = 0.04
    """
    lb = np.array([0.0, 0.0], dtype=np.float64)
    ub = np.array([0.04, 0.04], dtype=np.float64)
    return lb, ub


@pytest.fixture
def extreme_scale_polytope_2d():
    """2D polytope with 500,000x scale difference.

    Dimension 0: range = 0.002 (very small)
    Dimension 1: range = 999 (very large)

    Should trigger ValueError due to min range < 0.05.
    This tests numerical stability with disparate scales.

    Returns:
        tuple: (lower_bounds, upper_bounds) with extreme scale mismatch
    """
    lb = np.array([-1e-3, 1.0], dtype=np.float64)
    ub = np.array([1e-3, 1e3], dtype=np.float64)
    return lb, ub


@pytest.fixture
def collapsed_dimension_polytope():
    """2D polytope with first dimension collapsed (lb[0] == ub[0]).

    This is a degenerate polytope where one dimension has zero width.
    Should trigger DegeneratedError during vertex computation.

    Returns:
        tuple: (lower_bounds, upper_bounds) where dim 0 is collapsed
    """
    lb = np.array([1.0, -1.0], dtype=np.float64)
    ub = np.array([1.0, 1.0], dtype=np.float64)  # First dim collapsed
    return lb, ub


@pytest.fixture
def line_segment_polytope():
    """Line segment polytope in 2D space (degenerate to 1D).

    Vertices would define a line from (0,0) to (1,0).
    This polytope is degenerate: only 2 vertices in 2D space (need 3 for non-degenerate).
    Should trigger DegeneratedError due to insufficient vertices.

    Note: This is defined as H-representation constraints that force y = 0
    and 0 <= x <= 1.

    Returns:
        tuple: (constraints, lower_bounds, upper_bounds) for line segment polytope
    """
    # Constraints that define: y = 0 (y >= 0 AND y <= 0) and 0 <= x <= 1
    constraints = np.array(
        [
            [0, 0, 1],  # y >= 0
            [0, 0, -1],  # y <= 0  (forces y = 0)
            [0, 1, 0],  # x >= 0
            [-1, 1, 0],  # x <= 1
        ],
        dtype=np.float64,
    )

    lb = np.array([0.0, 0.0], dtype=np.float64)
    ub = np.array([1.0, 0.0], dtype=np.float64)  # ub[1] = 0 (collapsed)

    return constraints, lb, ub


@pytest.fixture
def infeasible_polytope_relu():
    """Infeasible polytope for ReLU: constraints may contradict bounds.

    Constraints define box [0,1]²:
    - 0 <= x <= 1
    - 0 <= y <= 1

    Bounds require: x in [-0.5, 0.5], y in [0.5, 1.0]

    Geometric intersection: x in [0, 0.5], y in [0.5, 1.0] is non-empty.

    However, ReLU requires mixed signs: lb < 0 < ub for all dimensions.
    After vertex computation, the algorithm detects this constraint violation
    and raises RuntimeError.

    Returns:
        tuple: (constraints, lower_bounds, upper_bounds)
    """
    constraints = np.array(
        [
            [0, 1, 0],  # x >= 0
            [-1, 1, 0],  # x <= 1
            [0, 0, 1],  # y >= 0
            [-1, 0, 1],  # y <= 1
        ],
        dtype=np.float64,
    )

    lb = np.array([-0.5, 0.5], dtype=np.float64)
    ub = np.array([0.5, 1.0], dtype=np.float64)

    return constraints, lb, ub


# ============================================================================
# RANDOM POLYTOPE FIXTURES
# ============================================================================


def generate_random_polytope_constraints(
    dim: int, num_constraints: int | None = None, seed: int = 42
):
    """Generate random polytope constraints.

    Creates a random polytope by:
    1. Generating random constraint coefficients
    2. Ensuring feasibility by setting bounds appropriately

    Args:
        dim: Input dimension (1 means 1D, 2 means 2D, etc.)
        num_constraints: Number of halfspace constraints. If None, uses 3^dim.
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (num_constraints, dim+1), H-representation constraints
    """
    rng = np.random.default_rng(seed)

    if num_constraints is None:
        num_constraints = 3**dim

    # Generate random constraint coefficients [-1, 1]
    coeff = rng.uniform(-1, 1, (num_constraints, dim))

    # Generate random offsets to ensure feasibility
    # Use positive offsets to ensure polytope includes origin region
    offset = rng.uniform(0.5, 2.0, num_constraints)

    # Combine into H-representation [b | A]
    constraints = np.hstack([offset.reshape(-1, 1), coeff])

    return constraints.astype(np.float64)


def generate_random_polytope_bounds(dim: int, seed: int = 42):
    """Generate bounds for a random polytope.

    Creates bounds as [-R, R]^dim where R scales with dimension.

    Args:
        dim: Input dimension
        seed: Random seed for reproducibility

    Returns:
        tuple: (lower_bounds, upper_bounds)
    """
    # Bounds scale with dimension (larger polytope for higher dimensions)
    radius = 1.0 + 0.5 * dim

    lb = -radius * np.ones(dim, dtype=np.float64)
    ub = radius * np.ones(dim, dtype=np.float64)

    return lb, ub


def generate_feasible_random_polytope(dim: int, num_constraints: int | None = None, seed: int = 42):
    """Generate random polytope with guaranteed feasibility.

    Uses dimension-scaled margin to control constraint distance from origin:
    - min_offset = 1.0 + 0.5*dim
    - max_offset = 5.0 + 0.5*dim

    This ensures constraints stay far enough from origin to remain feasible
    even with random coefficients. The margin grows with dimension to maintain
    proportional spacing.

    Args:
        dim: Input dimension
        num_constraints: Number of constraints (default: 3^dim)
        seed: Random seed for reproducibility

    Returns:
        tuple: (constraints, lb, ub) where:
            - constraints: Shape (num_constraints, dim+1), H-representation [b | A]
            - lb: Lower bounds, shape (dim,)
            - ub: Upper bounds, shape (dim,)
    """
    if num_constraints is None:
        num_constraints = 3**dim

    rng = np.random.default_rng(seed)

    # Random constraint coefficients
    coeff = rng.uniform(-1, 1, (num_constraints, dim))

    # Dimension-scaled offsets to ensure feasibility
    # User requirement: 1.0 + 0.5*d for margin control
    min_offset = 1.0 + 0.5 * dim
    max_offset = 5.0 + 0.5 * dim
    offset = rng.uniform(min_offset, max_offset, num_constraints)

    # H-representation: [b | A]
    constraints = np.hstack([offset.reshape(-1, 1), coeff])

    # Symmetric bounds scaled with dimension
    radius = 2.0 + 0.5 * dim
    lb = np.full(dim, -radius, dtype=np.float64)
    ub = np.full(dim, radius, dtype=np.float64)

    return constraints.astype(np.float64), lb, ub


@pytest.fixture(params=[2, 3, 4])
def random_polytope_constraints(request):
    """Parametrized random polytope constraints (2D, 3D, 4D).

    Generates random polytopes for each dimension.
    Uses dimension as random seed for reproducibility.

    Yields:
        np.ndarray: H-representation constraints
    """
    dim = request.param
    return generate_random_polytope_constraints(dim, seed=42 + dim)


@pytest.fixture(params=[2, 3, 4])
def random_polytope_bounds(request):
    """Parametrized random polytope bounds (2D, 3D, 4D).

    Yields:
        tuple: (lower_bounds, upper_bounds)
    """
    dim = request.param
    return generate_random_polytope_bounds(dim, seed=42 + dim)


@pytest.fixture(params=[2, 3, 4])
def random_polytope(request):
    """Parametrized random polytope (constraints + bounds).

    Yields:
        tuple: (constraints, lower_bounds, upper_bounds)
    """
    dim = request.param
    constraints = generate_random_polytope_constraints(dim, seed=42 + dim)
    bounds = generate_random_polytope_bounds(dim, seed=42 + dim)
    return constraints, bounds[0], bounds[1]


# ============================================================================
# DIMENSION FIXTURES
# ============================================================================


@pytest.fixture(params=[2, 3, 4])
def dimension(request):
    """Parametrized dimension fixture (2D, 3D, 4D only).

    Dimension is limited to 4D due to time constraints of high-dimensional
    polytope operations. Higher dimensions would make tests too slow.

    Yields:
        int: Dimension (2, 3, or 4)
    """
    return request.param


# ============================================================================
# ACTIVATION FUNCTION FIXTURES
# ============================================================================


@pytest.fixture
def activation_functions():
    """Provide dictionary of activation functions for testing.

    Returns:
        dict: Maps function names to numpy implementations
              (e.g., {'relu': relu_np, 'sigmoid': sigmoid_np, ...})
    """
    try:
        from wraact.wraact._functions import (
            elu_np,
            leakyrelu_np,
            relu_np,
            sigmoid_np,
            tanh_np,
        )

        return {
            "relu": relu_np,
            "sigmoid": sigmoid_np,
            "tanh": tanh_np,
            "elu": elu_np,
            "leakyrelu": leakyrelu_np,
        }
    except ImportError:
        pytest.skip("wraact.wraact._functions not available")


# ============================================================================
# HULL CLASS FIXTURES
# ============================================================================


@pytest.fixture
def relu_hull_class():
    """ReLUHull class for instantiation."""
    try:
        from wraact.acthull import ReLUHull

        return ReLUHull
    except ImportError:
        pytest.skip("ReLUHull not available")


@pytest.fixture
def leakyrelu_hull_class():
    """LeakyReLUHull class for instantiation."""
    try:
        from wraact.acthull import LeakyReLUHull

        return LeakyReLUHull
    except ImportError:
        pytest.skip("LeakyReLUHull not available")


@pytest.fixture
def elu_hull_class():
    """ELUHull class for instantiation."""
    try:
        from wraact.acthull import ELUHull

        return ELUHull
    except ImportError:
        pytest.skip("ELUHull not available")


@pytest.fixture
def sigmoid_hull_class():
    """SigmoidHull class for instantiation."""
    try:
        from wraact.acthull import SigmoidHull

        return SigmoidHull
    except ImportError:
        pytest.skip("SigmoidHull not available")


@pytest.fixture
def tanh_hull_class():
    """TanhHull class for instantiation."""
    try:
        from wraact.acthull import TanhHull

        return TanhHull
    except ImportError:
        pytest.skip("TanhHull not available")


@pytest.fixture
def maxpool_hull_class():
    """MaxPoolHull class for instantiation."""
    try:
        from wraact.acthull import MaxPoolHull

        return MaxPoolHull
    except ImportError:
        pytest.skip("MaxPoolHull not available")


# ============================================================================
# SAMPLING HELPERS
# ============================================================================


def sample_points_in_box(lb: np.ndarray, ub: np.ndarray, num_samples: int, seed: int | None = None):
    """Generate random points uniformly in a box [lb, ub]^d.

    Args:
        lb: Lower bounds (shape (d,))
        ub: Upper bounds (shape (d,))
        num_samples: Number of points to generate
        seed: Random seed

    Returns:
        np.ndarray: Shape (num_samples, d), random points
    """
    rng = np.random.default_rng(seed)

    d = len(lb)
    points = rng.uniform(0, 1, (num_samples, d))
    # Scale to [lb, ub]
    points = points * (ub - lb) + lb
    return points


@pytest.fixture
def point_sampler():
    """Fixture providing sampling helper function."""
    return sample_points_in_box


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================


def check_constraints_satisfied(
    constraints: np.ndarray, points: np.ndarray, tolerance: float = 1e-8
):
    """Check if points satisfy all constraints (H-representation).

    For each point p and constraints in H-form [b | A]:
        Constraint is: b + A @ p >= 0

    Args:
        constraints: Shape (num_constraints, d+1), [b | A]
        points: Shape (num_points, d), points to check
        tolerance: Numerical tolerance for >= 0 check

    Returns:
        np.ndarray: Shape (num_points,), boolean mask of satisfaction
    """
    b = constraints[:, :1]  # (num_constraints, 1)
    coeff = constraints[:, 1:]  # (num_constraints, d)

    # Compute: b + coeff @ p^T
    # Result shape: (num_constraints, num_points)
    ax = b + coeff @ points.T

    # All constraints satisfied for a point if ALL rows > -tolerance
    # Result shape: (num_points,)
    satisfied = np.all(ax > -tolerance, axis=0)

    return satisfied


@pytest.fixture
def constraint_checker():
    """Fixture providing constraint satisfaction checker."""
    return check_constraints_satisfied
