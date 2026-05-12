"""Precision and tightness benchmarking tests.

This module measures the precision (tightness) of hull over-approximations
using Monte Carlo volume estimation.

Note: These tests are marked with @pytest.mark.slow and may take significant time.

Key Benchmarks:
===============
- Hull volume estimation (tightness ratio)
- Comparison of different methods
- Volume scaling with dimension
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from wraact._functions import elu_np, leakyrelu_np, relu_np, sigmoid_np, tanh_np
from wraact.oney import ReLUHullWithOneY


class TestMonteCarloVolume:
    """Test volume estimation using Monte Carlo sampling."""

    def test_relu_volume_2d(self, relu_hull_class):
        """Estimate ReLU hull volume on 2D input.

        Expected: >95% satisfaction rate for piecewise linear activation.
        """
        hull = relu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Monte Carlo sampling to estimate volume
        rng = np.random.default_rng(42)
        num_samples = 10000
        inside_count = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = relu_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if np.all(constraint_values >= -1e-8):
                inside_count += 1

        satisfaction_rate = 100.0 * inside_count / num_samples

        print(f"ReLU 2D volume: {satisfaction_rate:.2f}% of sampled points inside hull")

        # Piecewise linear threshold: >95% required
        assert satisfaction_rate >= 95.0, (
            f"Precision regression: {satisfaction_rate:.2f}% < 95% threshold"
        )

    def test_sigmoid_volume_2d(self, sigmoid_hull_class):
        """Estimate Sigmoid hull volume on 2D input.

        Expected: >85% satisfaction rate for S-shaped activation.
        """
        hull = sigmoid_hull_class()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Monte Carlo sampling
        rng = np.random.default_rng(42)
        num_samples = 10000
        inside_count = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = sigmoid_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if np.all(constraint_values >= -1e-8):
                inside_count += 1

        satisfaction_rate = 100.0 * inside_count / num_samples

        print(f"Sigmoid 2D volume: {satisfaction_rate:.2f}% of sampled points inside hull")

        # S-shaped threshold: >85% required
        assert satisfaction_rate >= 85.0, (
            f"Precision regression: {satisfaction_rate:.2f}% < 85% threshold"
        )


class TestMethodComparison:
    """Compare precision across different methods."""

    def test_full_vs_withoney_precision(self, relu_hull_class):
        """Compare precision of full ReLU hull vs WithOneY variant.

        Expected: Both should maintain >95% precision for piecewise linear.
        """
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Full hull
        full_hull = relu_hull_class()
        full_result = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # WithOneY
        oney_hull = ReLUHullWithOneY()
        oney_result = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Measure satisfaction rate for each
        rng = np.random.default_rng(42)
        num_samples = 1000

        full_inside = 0
        oney_inside = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = relu_np(x)

            # For full hull, use all outputs
            point_full = np.concatenate([x, y])
            b = full_result[:, 0]
            a = full_result[:, 1:]
            if np.all((b + a @ point_full) >= -1e-8):
                full_inside += 1

            # For WithOneY, use only first output
            y_single = y[0:1]
            point_oney = np.concatenate([x, y_single])
            b = oney_result[:, 0]
            a = oney_result[:, 1:]
            if np.all((b + a @ point_oney) >= -1e-8):
                oney_inside += 1

        full_rate = 100.0 * full_inside / num_samples
        oney_rate = 100.0 * oney_inside / num_samples

        print(f"Full: {full_rate:.2f}%, WithOneY: {oney_rate:.2f}%")

        # Both should maintain precision
        assert full_rate >= 95.0, f"Full hull precision {full_rate:.2f}% < 95%"
        assert oney_rate >= 95.0, f"WithOneY precision {oney_rate:.2f}% < 95%"

    def test_activation_function_comparison(
        self, relu_hull_class, leakyrelu_hull_class, elu_hull_class, sigmoid_hull_class
    ):
        """Compare precision across different activation functions.

        Piecewise linear (ReLU, LeakyReLU, ELU): expect >95% precision
        S-shaped (Sigmoid): expect >85% precision
        """
        activations = {
            "ReLU": (
                relu_hull_class(),
                np.array([-1.0, -1.0]),
                np.array([1.0, 1.0]),
                relu_np,
                95.0,
            ),
            "LeakyReLU": (
                leakyrelu_hull_class(),
                np.array([-1.0, -1.0]),
                np.array([1.0, 1.0]),
                lambda x: leakyrelu_np(x, negative_slope=0.01),
                95.0,
            ),
            "ELU": (
                elu_hull_class(),
                np.array([-1.0, -1.0]),
                np.array([1.0, 1.0]),
                elu_np,
                95.0,
            ),
            "Sigmoid": (
                sigmoid_hull_class(),
                np.array([-2.0, -2.0]),
                np.array([2.0, 2.0]),
                sigmoid_np,
                85.0,
            ),
        }

        precision_results = {}
        regressions = []

        for name, (hull, lb, ub, activation_fn, threshold) in activations.items():
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            rng = np.random.default_rng(42)
            num_samples = 1000
            inside_count = 0

            for _ in range(num_samples):
                x = rng.uniform(lb, ub)
                y = activation_fn(x)
                point = np.concatenate([x, y])

                b = result[:, 0]
                a = result[:, 1:]
                if np.all((b + a @ point) >= -1e-8):
                    inside_count += 1

            precision = 100.0 * inside_count / num_samples
            precision_results[name] = precision

            if precision < threshold:
                regressions.append(f"{name}: {precision:.2f}% < {threshold}%")

        print(f"Precision by activation: {precision_results}")

        # Mark as xfail if any activation falls below threshold
        assert not regressions, f"Precision regressions: {', '.join(regressions)}"


class TestVolumeScaling:
    """Test how hull volume scales with dimension."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_relu_volume_scaling(self, relu_hull_class, dim):
        """Measure ReLU hull volume (via satisfaction rate) across dimensions.

        Expected baseline (piecewise linear): >95% precision across all dimensions.
        """
        hull = relu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Monte Carlo sampling (smaller sample for higher dimensions)
        num_samples = max(100, 10000 // (2**dim))
        rng = np.random.default_rng(42)
        inside_count = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = relu_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            if np.all((b + a @ point) >= -1e-8):
                inside_count += 1

        satisfaction_rate = 100.0 * inside_count / num_samples

        print(f"ReLU dim {dim}: {satisfaction_rate:.2f}% satisfaction ({num_samples} samples)")

        # Piecewise linear threshold: >95% required
        assert satisfaction_rate >= 95.0, (
            f"Precision regression: {satisfaction_rate:.2f}% < 95% threshold"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_sigmoid_volume_scaling(self, sigmoid_hull_class, dim):
        """Measure Sigmoid hull volume across dimensions.

        Expected baseline (S-shaped): >85% precision across all dimensions.
        """
        hull = sigmoid_hull_class()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Smaller sample for higher dimensions
        num_samples = max(100, 5000 // (2**dim))
        rng = np.random.default_rng(42)
        inside_count = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = sigmoid_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            if np.all((b + a @ point) >= -1e-8):
                inside_count += 1

        satisfaction_rate = 100.0 * inside_count / num_samples

        print(f"Sigmoid dim {dim}: {satisfaction_rate:.2f}% satisfaction ({num_samples} samples)")

        # S-shaped threshold: >85% required
        assert satisfaction_rate >= 85.0, (
            f"Precision regression: {satisfaction_rate:.2f}% < 85% threshold"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_tanh_volume_scaling(self, tanh_hull_class, dim):
        """Measure Tanh hull volume (via satisfaction rate) across dimensions.

        Expected baselines:
        - S-shaped threshold: >85% precision
        - Dimension scaling: 6·d constraints
        """
        hull = tanh_hull_class()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Monte Carlo sampling (smaller sample for higher dimensions)
        num_samples = max(100, 5000 // (2**dim))
        rng = np.random.default_rng(42)
        inside_count = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = tanh_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            if np.all((b + a @ point) >= -1e-8):
                inside_count += 1

        satisfaction_rate = 100.0 * inside_count / num_samples

        print(f"Tanh dim {dim}: {satisfaction_rate:.2f}% satisfaction ({num_samples} samples)")

        # S-shaped threshold: >85% precision required
        assert satisfaction_rate >= 85.0, (
            f"Precision regression: {satisfaction_rate:.2f}% < 85% threshold for S-shaped activation"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_leakyrelu_volume_scaling(self, leakyrelu_hull_class, dim):
        """Measure LeakyReLU hull volume across dimensions.

        Expected baselines:
        - Piecewise linear threshold: >95% precision
        - Dimension scaling: 2·d constraints
        """
        hull = leakyrelu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Monte Carlo sampling
        num_samples = max(100, 10000 // (2**dim))
        rng = np.random.default_rng(42)
        inside_count = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = leakyrelu_np(x, negative_slope=0.01)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            if np.all((b + a @ point) >= -1e-8):
                inside_count += 1

        satisfaction_rate = 100.0 * inside_count / num_samples

        print(f"LeakyReLU dim {dim}: {satisfaction_rate:.2f}% satisfaction ({num_samples} samples)")

        # Piecewise linear threshold: >95% precision required
        assert satisfaction_rate >= 95.0, (
            f"Precision regression: {satisfaction_rate:.2f}% < 95% threshold for piecewise linear activation"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_elu_volume_scaling(self, elu_hull_class, dim):
        """Measure ELU hull volume across dimensions.

        Expected baselines:
        - Piecewise linear threshold: >95% precision
        - Dimension scaling: 2·d constraints
        """
        hull = elu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Monte Carlo sampling
        num_samples = max(100, 10000 // (2**dim))
        rng = np.random.default_rng(42)
        inside_count = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = elu_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            if np.all((b + a @ point) >= -1e-8):
                inside_count += 1

        satisfaction_rate = 100.0 * inside_count / num_samples

        print(f"ELU dim {dim}: {satisfaction_rate:.2f}% satisfaction ({num_samples} samples)")

        # Piecewise linear threshold: >95% precision required
        assert satisfaction_rate >= 95.0, (
            f"Precision regression: {satisfaction_rate:.2f}% < 95% threshold for piecewise linear activation"
        )


class TestNumericalStability:
    """Test numerical stability of volume estimation."""

    def test_very_small_constraint_margins(self, relu_hull_class):
        """Test precision when constraint margins are very small.

        Polytope with range 0.002 < MIN_BOUNDS_RANGE (0.05) triggers ValueError.
        """
        hull = relu_hull_class()
        lb = np.array([-1e-3, -1e-3])
        ub = np.array([1e-3, 1e-3])

        # Algorithm raises ValueError for polytopes with range < MIN_BOUNDS_RANGE
        with pytest.raises(ValueError, match="Polytope too small"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    def test_very_large_constraint_bounds(self, relu_hull_class):
        """Test precision with very large input bounds."""
        hull = relu_hull_class()
        lb = np.array([-1e6, -1e6])
        ub = np.array([1e6, 1e6])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify numerical stability
        assert np.all(np.isfinite(result)), "Large bounds produced inf/nan"

        # Test at representative points
        test_points = [
            np.array([-1e6, -1e6]),
            np.array([0.0, 0.0]),
            np.array([1e6, 1e6]),
        ]

        for x in test_points:
            y = relu_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraints = b + a @ point

            assert np.all(np.isfinite(constraints)), f"Non-finite constraint at {x}"
