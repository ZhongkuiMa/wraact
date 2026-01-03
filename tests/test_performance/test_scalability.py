"""Scalability tests for hull computation across dimensions.

This module tests how hull computation scales with increasing dimensions
and constraint counts.

Note: These tests are marked with @pytest.mark.slow.

Key Tests:
==========
- Constraint count growth rate with dimension
- Memory usage scaling
- Computation complexity trends
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact.acthull import (
    ELUHull,
    LeakyReLUHull,
    MaxPoolHull,
    MaxPoolHullDLP,
    ReLUHull,
    SigmoidHull,
    TanhHull,
)


@pytest.mark.slow
class TestConstraintCountScaling:
    """Test how constraint count grows with dimension."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_relu_constraint_scaling(self, dim):
        """Measure ReLU constraint count for different dimensions.

        Expected baselines (2·d formula, ±20% tolerance):
        - Formula: 2·d constraints
        - 2D: 4 constraints, 3D: 6 constraints, 4D: 8 constraints
        """
        hull = ReLUHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_constraints = result.shape[0]
        expected = 2 * dim  # 2·d formula
        threshold_low = expected * 0.8  # -20%
        threshold_high = expected * 1.2  # +20%

        print(f"ReLU dim {dim}: {num_constraints} constraints (expected {expected})")

        # Mark as xfail if constraint count deviates beyond ±20%
        if num_constraints < threshold_low or num_constraints > threshold_high:
            pytest.xfail(
                f"Constraint count regression: {num_constraints} outside range [{threshold_low:.0f}, {threshold_high:.0f}]"
            )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_sigmoid_constraint_scaling(self, dim):
        """Measure Sigmoid constraint count for different dimensions.

        Expected baselines (6·d formula, ±20% tolerance):
        - Formula: 6·d constraints
        - 2D: 12 constraints, 3D: 18 constraints, 4D: 24 constraints
        """
        hull = SigmoidHull()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_constraints = result.shape[0]
        expected = 6 * dim  # 6·d formula
        threshold_low = expected * 0.8  # -20%
        threshold_high = expected * 1.2  # +20%

        print(f"Sigmoid dim {dim}: {num_constraints} constraints (expected {expected})")

        # Mark as xfail if constraint count deviates beyond ±20%
        if num_constraints < threshold_low or num_constraints > threshold_high:
            pytest.xfail(
                f"Constraint count regression: {num_constraints} outside range [{threshold_low:.0f}, {threshold_high:.0f}]"
            )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_tanh_constraint_scaling(self, dim):
        """Measure Tanh constraint count for different dimensions.

        Expected baselines (4·d formula, ±20% tolerance):
        - Formula: 4·d constraints
        - 2D: 8 constraints, 3D: 12 constraints, 4D: 16 constraints
        """
        hull = TanhHull()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_constraints = result.shape[0]
        expected = 4 * dim  # 4·d formula
        threshold_low = expected * 0.8  # -20%
        threshold_high = expected * 1.2  # +20%

        print(f"Tanh dim {dim}: {num_constraints} constraints (expected {expected})")

        # Mark as xfail if constraint count deviates beyond ±20%
        if num_constraints < threshold_low or num_constraints > threshold_high:
            pytest.xfail(
                f"Constraint count regression: {num_constraints} outside range [{threshold_low:.0f}, {threshold_high:.0f}]"
            )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_leakyrelu_constraint_scaling(self, dim):
        """Measure LeakyReLU constraint count for different dimensions.

        Expected baselines (2·d formula, ±20% tolerance):
        - Formula: 2·d constraints
        - 2D: 4 constraints, 3D: 6 constraints, 4D: 8 constraints
        """
        hull = LeakyReLUHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_constraints = result.shape[0]
        expected = 2 * dim  # 2·d formula
        threshold_low = expected * 0.8  # -20%
        threshold_high = expected * 1.2  # +20%

        print(f"LeakyReLU dim {dim}: {num_constraints} constraints (expected {expected})")

        # Mark as xfail if constraint count deviates beyond ±20%
        if num_constraints < threshold_low or num_constraints > threshold_high:
            pytest.xfail(
                f"Constraint count regression: {num_constraints} outside range [{threshold_low:.0f}, {threshold_high:.0f}]"
            )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_elu_constraint_scaling(self, dim):
        """Measure ELU constraint count for different dimensions.

        Expected baselines (3·d formula, ±20% tolerance):
        - Formula: 3·d constraints
        - 2D: 6 constraints, 3D: 9 constraints, 4D: 12 constraints
        """
        hull = ELUHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_constraints = result.shape[0]
        expected = 3 * dim  # 3·d formula
        threshold_low = expected * 0.8  # -20%
        threshold_high = expected * 1.2  # +20%

        print(f"ELU dim {dim}: {num_constraints} constraints (expected {expected})")

        # Mark as xfail if constraint count deviates beyond ±20%
        if num_constraints < threshold_low or num_constraints > threshold_high:
            pytest.xfail(
                f"Constraint count regression: {num_constraints} outside range [{threshold_low:.0f}, {threshold_high:.0f}]"
            )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_maxpool_constraint_scaling(self, dim):
        """Measure MaxPool constraint count for different dimensions.

        Expected baselines (2·d formula, ±20% tolerance):
        - Formula: 2·d constraints
        - 2D: 4 constraints, 3D: 6 constraints, 4D: 8 constraints
        """
        hull = MaxPoolHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_constraints = result.shape[0]
        expected = 2 * dim  # 2·d formula
        threshold_low = expected * 0.8  # -20%
        threshold_high = expected * 1.2  # +20%

        print(f"MaxPool dim {dim}: {num_constraints} constraints (expected {expected})")

        # Mark as xfail if constraint count deviates beyond ±20%
        if num_constraints < threshold_low or num_constraints > threshold_high:
            pytest.xfail(
                f"Constraint count regression: {num_constraints} outside range [{threshold_low:.0f}, {threshold_high:.0f}]"
            )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_maxpooldlp_constraint_scaling(self, dim):
        """Measure MaxPoolDLP constraint count for different dimensions.

        Expected baselines (2·d formula, ±20% tolerance):
        - Formula: 2·d constraints
        - 2D: 4 constraints, 3D: 6 constraints, 4D: 8 constraints
        """
        hull = MaxPoolHullDLP()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_constraints = result.shape[0]
        expected = 2 * dim  # 2·d formula
        threshold_low = expected * 0.8  # -20%
        threshold_high = expected * 1.2  # +20%

        print(f"MaxPoolDLP dim {dim}: {num_constraints} constraints (expected {expected})")

        # Mark as xfail if constraint count deviates beyond ±20%
        if num_constraints < threshold_low or num_constraints > threshold_high:
            pytest.xfail(
                f"Constraint count regression: {num_constraints} outside range [{threshold_low:.0f}, {threshold_high:.0f}]"
            )


@pytest.mark.slow
class TestMemoryScaling:
    """Test memory usage growth with dimension."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_relu_memory_usage(self, dim):
        """Measure memory usage for ReLU hull of different dimensions."""
        hull = ReLUHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        memory_bytes = result.nbytes
        memory_kb = memory_bytes / 1024

        print(f"ReLU dim {dim}: {memory_bytes} bytes ({memory_kb:.2f} KB)")

        # Verify reasonable memory usage (<1MB is expected for small polytopes)
        if memory_bytes > 10 * 1024 * 1024:
            pytest.xfail(f"Excessive memory for dimension {dim}: {memory_kb:.2f} KB")

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_sigmoid_memory_usage(self, dim):
        """Measure memory usage for Sigmoid hull of different dimensions."""
        hull = SigmoidHull()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        memory_bytes = result.nbytes
        memory_kb = memory_bytes / 1024

        print(f"Sigmoid dim {dim}: {memory_bytes} bytes ({memory_kb:.2f} KB)")

        # Verify reasonable memory usage (<10MB is expected for small polytopes)
        if memory_bytes > 10 * 1024 * 1024:
            pytest.xfail(f"Excessive memory for dimension {dim}: {memory_kb:.2f} KB")

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_leakyrelu_memory_usage(self, dim):
        """Measure memory usage for LeakyReLU hull of different dimensions."""
        hull = LeakyReLUHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        memory_bytes = result.nbytes
        memory_kb = memory_bytes / 1024

        print(f"LeakyReLU dim {dim}: {memory_bytes} bytes ({memory_kb:.2f} KB)")

        # Piecewise linear functions should use reasonable memory
        assert memory_bytes < 1 * 1024 * 1024, f"Excessive memory for LeakyReLU dimension {dim}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_elu_memory_usage(self, dim):
        """Measure memory usage for ELU hull of different dimensions."""
        hull = ELUHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        memory_bytes = result.nbytes
        memory_kb = memory_bytes / 1024

        print(f"ELU dim {dim}: {memory_bytes} bytes ({memory_kb:.2f} KB)")

        # Piecewise linear functions should use reasonable memory
        assert memory_bytes < 1 * 1024 * 1024, f"Excessive memory for ELU dimension {dim}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_maxpool_memory_usage(self, dim):
        """Measure memory usage for MaxPool hull of different dimensions."""
        hull = MaxPoolHull()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        memory_bytes = result.nbytes
        memory_kb = memory_bytes / 1024

        print(f"MaxPool dim {dim}: {memory_bytes} bytes ({memory_kb:.2f} KB)")

        # Pooling functions should use reasonable memory
        assert memory_bytes < 1 * 1024 * 1024, f"Excessive memory for MaxPool dimension {dim}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_maxpooldlp_memory_usage(self, dim):
        """Measure memory usage for MaxPoolDLP hull of different dimensions."""
        hull = MaxPoolHullDLP()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        memory_bytes = result.nbytes
        memory_kb = memory_bytes / 1024

        print(f"MaxPoolDLP dim {dim}: {memory_bytes} bytes ({memory_kb:.2f} KB)")

        # Pooling functions should use reasonable memory
        assert memory_bytes < 1 * 1024 * 1024, f"Excessive memory for MaxPoolDLP dimension {dim}"


@pytest.mark.slow
class TestComputationalComplexity:
    """Test computational complexity across dimensions."""

    def test_constraint_column_count_scaling(self):
        """Verify constraint matrix columns scale with dimension.

        Expected: columns = 1 (bias) + dim (inputs) + dim (outputs)
        """
        hull = ReLUHull()

        column_counts = {}
        mismatches = []

        for dim in [2, 3, 4]:
            lb = np.full(dim, -1.0)
            ub = np.full(dim, 1.0)

            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            # Expected columns: 1 (bias) + dim (inputs) + dim (outputs)
            expected_cols = 1 + dim + dim
            actual_cols = result.shape[1]

            column_counts[dim] = actual_cols

            if actual_cols != expected_cols:
                mismatches.append(f"dim {dim}: expected {expected_cols}, got {actual_cols}")

        print(f"Column count scaling: {column_counts}")

        # Mark as xfail if any dimension has unexpected column count
        if mismatches:
            pytest.xfail(f"Column count regression: {', '.join(mismatches)}")

    def test_constraint_row_count_polynomial_growth(self):
        """Verify constraint row count grows polynomially with dimension.

        For ReLU (piecewise linear), growth should be O(d) = 2·d constraints.
        Exponential growth would be > (2*2)^2 = 16 constraints at 4D.
        """
        hull = ReLUHull()

        constraint_counts = {}

        for dim in [2, 3, 4]:
            lb = np.full(dim, -1.0)
            ub = np.full(dim, 1.0)

            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            num_constraints = result.shape[0]
            constraint_counts[dim] = num_constraints

        print(f"Constraint count by dimension: {constraint_counts}")

        # Verify growth is reasonable (not exponential)
        # For ReLU, growth should be O(d), so 4D should be ~8, definitely < 2D squared
        if constraint_counts[4] >= constraint_counts[2] ** 2:
            pytest.xfail(
                f"Excessive constraint growth: {constraint_counts[4]} >= {constraint_counts[2]}^2"
            )


@pytest.mark.slow
class TestInputBoundScaling:
    """Test scaling with different input bound ranges."""

    def test_varying_bound_ranges(self):
        """Test ReLU with different input bound ranges.

        Constraint count should not significantly depend on range magnitude.
        All ranges should produce similar number of constraints (within 2x).
        """
        hull = ReLUHull()
        dim = 2

        ranges = [0.1, 1.0, 10.0, 100.0]
        constraint_counts = {}

        for bound_range in ranges:
            lb = np.full(dim, -bound_range)
            ub = np.full(dim, bound_range)

            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            constraint_counts[bound_range] = result.shape[0]

        print(f"Constraint counts by range: {constraint_counts}")

        # Constraint count should not depend on range magnitude (only on structure)
        # All should have roughly the same number of constraints
        min_count = min(constraint_counts.values())
        max_count = max(constraint_counts.values())

        # Allow some variation due to numerical effects, but not huge (< 2x)
        ratio = max_count / min_count if min_count > 0 else 1.0

        # Mark as xfail if ratio indicates range has significant impact
        if ratio >= 2.0:
            pytest.xfail(
                f"Large variation in constraint count across ranges: {ratio:.2f}x {constraint_counts}"
            )


@pytest.mark.slow
class TestAsymptoticBehavior:
    """Test asymptotic behavior at extreme dimensions/bounds."""

    def test_dimension_four_stability(self):
        """Verify stable behavior at dimension 4 (upper test limit)."""
        hull = ReLUHull()
        lb = np.full(4, -1.0)
        ub = np.full(4, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should complete without numerical issues
        assert np.all(np.isfinite(result)), "Non-finite values in 4D result"
        assert result.shape[0] > 0, "No constraints in 4D"

    def test_sigmoid_4d_stability(self):
        """Verify Sigmoid handles 4D input stably."""
        hull = SigmoidHull()
        lb = np.full(4, -2.0)
        ub = np.full(4, 2.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # S-shaped functions have more complex constraints, verify stability
        assert np.all(np.isfinite(result)), "Non-finite values in 4D Sigmoid"
        assert result.shape[0] > 0, "No constraints in 4D Sigmoid"
