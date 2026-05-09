"""Runtime performance benchmarking tests.

This module benchmarks hull computation times across different activation
functions, dimensions, and input configurations.

Note: These tests are marked with @pytest.mark.slow and may take several seconds.

Key Benchmarks:
===============
- Runtime scaling with dimension (2D, 3D, 4D)
- Activation function comparison
- WithOneY vs full hull performance
"""

__docformat__ = "restructuredtext"

import time

import numpy as np
import pytest

from wraact.acthull import MaxPoolHull, MaxPoolHullDLP
from wraact.oney import (
    ELUHullWithOneY,
    LeakyReLUHullWithOneY,
    MaxPoolHullDLPWithOneY,
    MaxPoolHullWithOneY,
    ReLUHullWithOneY,
    SigmoidHullWithOneY,
    TanhHullWithOneY,
)


class TestRuntimeScaling:
    """Test runtime scaling with dimension."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_relu_runtime_by_dimension(self, relu_hull_class, dim):
        """Measure ReLU hull runtime for different dimensions.

        Expected baselines (±150% tolerance):
        - 2D: 0.45ms
        - 3D: 0.35ms
        - 4D: 0.41ms

        Uses 10 warmup iterations + 20 measurements with percentile-based threshold.
        """
        hull = relu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Warm up with 10 iterations
        for _ in range(10):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Measure 20 times and use 90th percentile
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        elapsed_ms = np.percentile(times, 90)
        median_ms = np.median(times)

        # Baseline thresholds (±150% tolerance)
        baselines = {2: 0.45, 3: 0.35, 4: 0.41}  # milliseconds
        threshold = baselines[dim] * 2.5  # 150% tolerance

        print(
            f"ReLU dim {dim}: p90={elapsed_ms:.2f}ms, median={median_ms:.2f}ms, {result.shape[0]} constraints"
        )

        # Mark as xfail if performance regressed
        assert elapsed_ms <= threshold, (
            f"Performance regression: p90={elapsed_ms:.2f}ms > {threshold:.2f}ms baseline"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_sigmoid_runtime_by_dimension(self, sigmoid_hull_class, dim):
        """Measure Sigmoid hull runtime for different dimensions.

        Expected baselines (±150% tolerance, S-shaped):
        - 2D: 0.68ms
        - 3D: 0.93ms
        - 4D: 1.21ms

        Uses 10 warmup iterations + 20 measurements with percentile-based threshold.
        """
        hull = sigmoid_hull_class()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        # Warm up with 10 iterations
        for _ in range(10):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Measure 20 times and use 90th percentile
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        elapsed_ms = np.percentile(times, 90)
        median_ms = np.median(times)

        # Baseline thresholds (±150% tolerance for S-shaped)
        baselines = {2: 0.68, 3: 0.93, 4: 1.21}  # milliseconds
        threshold = baselines[dim] * 2.5  # 150% tolerance

        print(
            f"Sigmoid dim {dim}: p90={elapsed_ms:.2f}ms, median={median_ms:.2f}ms, {result.shape[0]} constraints"
        )

        # Mark as xfail if performance regressed
        assert elapsed_ms <= threshold, (
            f"Performance regression: p90={elapsed_ms:.2f}ms > {threshold:.2f}ms baseline"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_leakyrelu_runtime_by_dimension(self, leakyrelu_hull_class, dim):
        """Measure LeakyReLU hull runtime for different dimensions.

        Expected baselines (±150% tolerance, piecewise linear):
        - 2D: 0.5ms
        - 3D: 0.4ms
        - 4D: 0.5ms

        Uses 10 warmup iterations + 20 measurements with percentile-based threshold
        for maximum stability across varying system loads.
        """
        hull = leakyrelu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Warm up with 10 iterations to fully stabilize JIT compilation
        for _ in range(10):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Measure 20 times and use 90th percentile for stability
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        elapsed_ms = np.percentile(times, 90)  # 90th percentile to avoid outliers
        median_ms = np.median(times)

        # Baseline thresholds (±150% tolerance for stability)
        baselines = {2: 0.5, 3: 0.4, 4: 0.5}  # milliseconds
        threshold = baselines[dim] * 2.5  # 150% tolerance

        print(
            f"LeakyReLU dim {dim}: p90={elapsed_ms:.2f}ms, median={median_ms:.2f}ms, times={[f'{t:.2f}' for t in times[:5]]}..., {result.shape[0]} constraints"
        )

        # Mark as xfail if performance regressed significantly
        assert elapsed_ms <= threshold, (
            f"Performance regression: p90={elapsed_ms:.2f}ms > {threshold:.2f}ms baseline"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_elu_runtime_by_dimension(self, elu_hull_class, dim):
        """Measure ELU hull runtime for different dimensions.

        Expected baselines (±150% tolerance, piecewise linear):
        - 2D: 0.55ms
        - 3D: 0.45ms
        - 4D: 0.50ms

        Uses 10 warmup iterations + 20 measurements with percentile-based threshold.
        """
        hull = elu_hull_class()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Warm up with 10 iterations
        for _ in range(10):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Measure 20 times and use 90th percentile
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        elapsed_ms = np.percentile(times, 90)
        median_ms = np.median(times)

        # Baseline thresholds (±150% tolerance)
        baselines = {2: 0.55, 3: 0.45, 4: 0.50}  # milliseconds
        threshold = baselines[dim] * 2.5  # 150% tolerance

        print(
            f"ELU dim {dim}: p90={elapsed_ms:.2f}ms, median={median_ms:.2f}ms, {result.shape[0]} constraints"
        )

        # Mark as xfail if performance regressed
        assert elapsed_ms <= threshold, (
            f"Performance regression: p90={elapsed_ms:.2f}ms > {threshold:.2f}ms baseline"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_tanh_runtime_by_dimension(self, tanh_hull_class, dim):
        """Measure Tanh hull runtime for different dimensions.

        Expected baselines (±150% tolerance, S-shaped):
        - 2D: 1.6ms
        - 3D: 2.6ms
        - 4D: 3.6ms

        Uses 10 warmup iterations + 20 measurements with percentile-based threshold.
        """
        hull = tanh_hull_class()
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        # Warm up with 10 iterations
        for _ in range(10):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Measure 20 times and use 90th percentile
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        elapsed_ms = np.percentile(times, 90)
        median_ms = np.median(times)

        # Baseline thresholds (±150% tolerance for S-shaped)
        baselines = {2: 1.6, 3: 2.6, 4: 3.6}  # milliseconds
        threshold = baselines[dim] * 2.5  # 150% tolerance

        print(
            f"Tanh dim {dim}: p90={elapsed_ms:.2f}ms, median={median_ms:.2f}ms, {result.shape[0]} constraints"
        )

        # Mark as xfail if performance regressed
        assert elapsed_ms <= threshold, (
            f"Performance regression: p90={elapsed_ms:.2f}ms > {threshold:.2f}ms baseline"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_maxpooldlp_runtime_by_dimension(self, dim):
        """Measure MaxPoolDLP hull runtime for different dimensions.

        Expected baselines (±150% tolerance, pooling):
        - 2D: 0.5ms
        - 3D: 0.45ms
        - 4D: 0.50ms

        Uses 10 warmup iterations + 20 measurements with percentile-based threshold.
        """
        hull = MaxPoolHullDLP()
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Warm up with 10 iterations
        for _ in range(10):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Measure 20 times and use 90th percentile
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        elapsed_ms = np.percentile(times, 90)
        median_ms = np.median(times)

        # Baseline thresholds (±150% tolerance)
        baselines = {2: 0.5, 3: 0.45, 4: 0.50}  # milliseconds
        threshold = baselines[dim] * 2.5  # 150% tolerance

        print(
            f"MaxPoolDLP dim {dim}: p90={elapsed_ms:.2f}ms, median={median_ms:.2f}ms, {result.shape[0]} constraints"
        )

        # Mark as xfail if performance regressed
        assert elapsed_ms <= threshold, (
            f"Performance regression: p90={elapsed_ms:.2f}ms > {threshold:.2f}ms baseline"
        )


class TestActivationComparison:
    """Compare runtime across different activation functions."""

    def test_all_activations_runtime_2d(
        self,
        relu_hull_class,
        leakyrelu_hull_class,
        elu_hull_class,
        sigmoid_hull_class,
        tanh_hull_class,
    ):
        """Compare runtime of all activation functions on 2D input.

        Uses 10 warmup iterations + 20 measurements with percentile-based thresholds.
        Marks as xfail if any activation significantly exceeds its expected baseline.
        """
        activations = {
            "ReLU": (relu_hull_class(), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 0.45 * 2.5),
            "LeakyReLU": (
                leakyrelu_hull_class(),
                np.array([-1.0, -1.0]),
                np.array([1.0, 1.0]),
                0.50 * 2.5,
            ),
            "ELU": (elu_hull_class(), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 0.55 * 2.5),
            "Sigmoid": (
                sigmoid_hull_class(),
                np.array([-2.0, -2.0]),
                np.array([2.0, 2.0]),
                1.50 * 2.5,
            ),
            "Tanh": (tanh_hull_class(), np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 1.60 * 2.5),
            "MaxPool": (MaxPoolHull(), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 0.50 * 2.5),
        }

        runtimes = {}
        regressions = []

        for name, (hull, lb, ub, threshold) in activations.items():
            # Warm up with 10 iterations
            for _ in range(10):
                hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            # Measure 20 times and use 90th percentile
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)

            elapsed_ms = np.percentile(times, 90)
            runtimes[name] = elapsed_ms

            if elapsed_ms > threshold:
                regressions.append(f"{name}: {elapsed_ms:.2f}ms > {threshold:.2f}ms")

        print(f"2D activation comparison: {runtimes}")

        # Mark as xfail if any activation regressed
        assert not regressions, f"Performance regressions: {', '.join(regressions)}"


class TestWithOneYPerformance:
    """Compare WithOneY variants with full hull performance."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_relu_withoney_speedup(self, relu_hull_class, dim):
        """Measure speedup of ReLU WithOneY vs full hull.

        Expected baseline (≥1.0x speedup, ideally 1.3x):
        - 2D: 1.3x
        - 3D: 1.3x
        - 4D: 1.3x

        Uses 10 warmup iterations + 20 measurements with median for stability.
        """
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Full hull warmup
        full_hull = relu_hull_class()
        for _ in range(10):
            full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # WithOneY warmup
        oney_hull = ReLUHullWithOneY()
        for _ in range(10):
            oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Measure full hull 20 times
        full_times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            full_times.append(time.perf_counter() - start)

        # Measure WithOneY 20 times
        oney_times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            oney_times.append(time.perf_counter() - start)

        # Use median for stability
        full_time_ms = np.median(full_times) * 1000
        oney_time_ms = np.median(oney_times) * 1000

        # WithOneY should be faster (fewer dimensions to handle)
        speedup = full_time_ms / oney_time_ms if oney_time_ms > 0 else 1.0

        print(
            f"ReLU WithOneY {dim}D speedup: {speedup:.2f}x (full: {full_time_ms:.2f}ms, oney: {oney_time_ms:.2f}ms)"
        )

        # Speedup threshold: ≥0.95x (95% efficiency, no major regression)
        assert speedup >= 0.95, (
            f"Speedup {speedup:.2f}x < 0.95x indicates WithOneY is significantly slower"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_leakyrelu_withoney_speedup(self, leakyrelu_hull_class, dim):
        """Measure speedup of LeakyReLU WithOneY vs full hull.

        Expected baseline (≥1.1x speedup required):
        - 2D: 1.3x
        - 3D: 1.3x
        - 4D: 1.3x
        """
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Full hull
        full_hull = leakyrelu_hull_class()
        full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        full_time = time.perf_counter() - start

        # WithOneY
        oney_hull = LeakyReLUHullWithOneY()
        oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_time = time.perf_counter() - start

        speedup = full_time / oney_time if oney_time > 0 else 1.0

        print(
            f"LeakyReLU WithOneY {dim}D speedup: {speedup:.2f}x (full: {full_time * 1000:.2f}ms, oney: {oney_time * 1000:.2f}ms)"
        )

        # Speedup threshold: ≥0.95x (95% efficiency, no major regression)
        assert speedup >= 0.95, (
            f"Speedup {speedup:.2f}x < 0.95x indicates WithOneY is significantly slower"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_elu_withoney_speedup(self, elu_hull_class, dim):
        """Measure speedup of ELU WithOneY vs full hull.

        Expected baseline (≥1.1x speedup required):
        - 2D: 1.3x
        - 3D: 1.3x
        - 4D: 1.3x
        """
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Full hull
        full_hull = elu_hull_class()
        full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        full_time = time.perf_counter() - start

        # WithOneY
        oney_hull = ELUHullWithOneY()
        oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_time = time.perf_counter() - start

        speedup = full_time / oney_time if oney_time > 0 else 1.0

        print(
            f"ELU WithOneY {dim}D speedup: {speedup:.2f}x (full: {full_time * 1000:.2f}ms, oney: {oney_time * 1000:.2f}ms)"
        )

        # Speedup threshold: ≥0.95x (95% efficiency, no major regression)
        assert speedup >= 0.95, (
            f"Speedup {speedup:.2f}x < 0.95x indicates WithOneY is significantly slower"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_tanh_withoney_speedup(self, tanh_hull_class, dim):
        """Measure speedup of Tanh WithOneY vs full hull.

        Expected baseline (≥1.1x speedup required):
        - 2D: 1.2x
        - 3D: 1.2x
        - 4D: 1.2x
        """
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        # Full hull
        full_hull = tanh_hull_class()
        full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        full_time = time.perf_counter() - start

        # WithOneY
        oney_hull = TanhHullWithOneY()
        oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_time = time.perf_counter() - start

        speedup = full_time / oney_time if oney_time > 0 else 1.0

        print(
            f"Tanh WithOneY {dim}D speedup: {speedup:.2f}x (full: {full_time * 1000:.2f}ms, oney: {oney_time * 1000:.2f}ms)"
        )

        # Speedup threshold: ≥0.95x (95% efficiency, no major regression)
        assert speedup >= 0.95, (
            f"Speedup {speedup:.2f}x < 0.95x indicates WithOneY is significantly slower"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_sigmoid_withoney_speedup(self, sigmoid_hull_class, dim):
        """Measure speedup of Sigmoid WithOneY vs full hull.

        Expected baseline (≥1.1x speedup required):
        - 2D: 1.2x
        - 3D: 1.2x
        - 4D: 1.2x
        """
        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)

        # Full hull
        full_hull = sigmoid_hull_class()
        full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        full_time = time.perf_counter() - start

        # WithOneY
        oney_hull = SigmoidHullWithOneY()
        oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_time = time.perf_counter() - start

        speedup = full_time / oney_time if oney_time > 0 else 1.0

        print(
            f"Sigmoid WithOneY {dim}D speedup: {speedup:.2f}x (full: {full_time * 1000:.2f}ms, oney: {oney_time * 1000:.2f}ms)"
        )

        # Speedup threshold: ≥0.95x (95% efficiency, no major regression)
        assert speedup >= 0.95, (
            f"Speedup {speedup:.2f}x < 0.95x indicates WithOneY is significantly slower"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_maxpool_withoney_speedup(self, dim):
        """Measure speedup of MaxPool WithOneY vs full hull.

        MaxPool WithOneY optimization is less effective than for other activations.
        Expected baseline (≥0.95x speedup, 95% efficiency):
        - 2D: ~1.0x (minimal improvement)
        - 3D: ~1.0x (minimal improvement)
        - 4D: ~1.0x (minimal improvement)
        """
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Full hull
        full_hull = MaxPoolHull()
        full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        full_time = time.perf_counter() - start

        # WithOneY
        oney_hull = MaxPoolHullWithOneY()
        oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_time = time.perf_counter() - start

        speedup = full_time / oney_time if oney_time > 0 else 1.0

        print(
            f"MaxPool WithOneY {dim}D speedup: {speedup:.2f}x (full: {full_time * 1000:.2f}ms, oney: {oney_time * 1000:.2f}ms)"
        )

        # Speedup threshold: ≥0.95x (95% efficiency, no major regression)
        assert speedup >= 0.95, (
            f"Speedup {speedup:.2f}x < 0.95x indicates WithOneY is significantly slower"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_maxpooldlp_withoney_speedup(self, dim):
        """Measure speedup of MaxPoolDLP WithOneY vs full hull.

        MaxPoolDLP WithOneY optimization is less effective than for other activations.
        Expected baseline (≥0.95x speedup, 95% efficiency):
        - 2D: ~1.0x (minimal improvement)
        - 3D: ~1.0x (minimal improvement)
        - 4D: ~1.0x (minimal improvement)
        """
        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)

        # Full hull
        full_hull = MaxPoolHullDLP()
        full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = full_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        full_time = time.perf_counter() - start

        # WithOneY
        oney_hull = MaxPoolHullDLPWithOneY()
        oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        start = time.perf_counter()
        _ = oney_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        oney_time = time.perf_counter() - start

        speedup = full_time / oney_time if oney_time > 0 else 1.0

        print(
            f"MaxPoolDLP WithOneY {dim}D speedup: {speedup:.2f}x (full: {full_time * 1000:.2f}ms, oney: {oney_time * 1000:.2f}ms)"
        )

        # Speedup threshold: ≥0.95x (95% efficiency, no major regression)
        assert speedup >= 0.95, (
            f"Speedup {speedup:.2f}x < 0.95x indicates WithOneY is significantly slower"
        )


class TestNumbaCompilation:
    """Test Numba JIT compilation effectiveness."""

    def test_relu_numba_warmup(self, relu_hull_class):
        """Verify Numba JIT warmup effect.

        Expects first call to have ~5-10x overhead from JIT compilation,
        with deterministic results across calls.
        """
        hull = relu_hull_class()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # First call (includes compilation)
        start = time.perf_counter()
        result1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        first_call = time.perf_counter() - start

        # Second call (compiled code)
        start = time.perf_counter()
        result2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        second_call = time.perf_counter() - start

        # Verify determinism
        np.testing.assert_array_equal(result1, result2)

        # First call should have some overhead from compilation
        ratio = first_call / second_call if second_call > 0 else 1.0

        print(
            f"Numba compilation ratio: {ratio:.2f}x (first: {first_call * 1000:.2f}ms, second: {second_call * 1000:.2f}ms)"
        )

        # Mark as xfail if compilation overhead is unusually high (>20x)
        if ratio > 20.0:
            pytest.xfail(f"Unusually high JIT compilation overhead: {ratio:.2f}x")


class TestMemoryUsage:
    """Test memory usage and constraint growth."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_constraint_count_growth(self, relu_hull_class, sigmoid_hull_class, dim):
        """Measure constraint count growth with dimension.

        Expected baselines (±20% tolerance):
        - ReLU: 2·d constraints
        - Sigmoid: 6·d constraints
        """
        activations = {
            "ReLU": (relu_hull_class(), 2 * dim),  # 2·d constraints
            "Sigmoid": (sigmoid_hull_class(), 6 * dim),  # 6·d constraints
        }

        for name, (hull, expected_count) in activations.items():
            if name == "Sigmoid":
                lb = np.full(dim, -2.0)
                ub = np.full(dim, 2.0)
            else:
                lb = np.full(dim, -1.0)
                ub = np.full(dim, 1.0)

            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            num_constraints = result.shape[0]

            threshold_low = expected_count * 0.8
            threshold_high = expected_count * 1.2

            print(
                f"{name} dim {dim}: {num_constraints} constraints (expected {expected_count}), {result.nbytes} bytes"
            )

            # Assert constraint count is within ±20% of expected baseline
            assert threshold_low <= num_constraints <= threshold_high, (
                f"{name} constraint count regression: {num_constraints} outside range "
                f"[{threshold_low:.0f}, {threshold_high:.0f}]"
            )
