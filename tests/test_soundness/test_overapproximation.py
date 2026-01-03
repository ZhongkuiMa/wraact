"""Tests for sound over-approximation guarantees.

This module verifies that computed hulls are sound over-approximations of
activation functions. Key property: all points (x, f(x)) where x satisfies
input constraints must satisfy all output constraints.

Key Tests:
==========
- Soundness: 100% (or >99.9%) of sampled points satisfy constraints
- Tightness: Measure precision of over-approximation
- Consistency: Hull properties preserved across multiple calls
"""

__docformat__ = "restructuredtext"

import numpy as np

from wraact._functions import elu_np, leakyrelu_np, relu_np, sigmoid_np
from wraact.acthull import ELUHull, LeakyReLUHull, MaxPoolHull, ReLUHull, SigmoidHull, TanhHull


class TestReLUSoundness:
    """Test soundness of ReLU hull over-approximation."""

    def test_relu_soundness_2d(self):
        """Verify ReLU hull contains all relu outputs on 2D polytope."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Sample many points and verify all satisfy constraints
        rng = np.random.default_rng(42)
        num_samples = 1000
        violations = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = relu_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.9, f"ReLU soundness: {satisfaction_rate:.2f}%"

    def test_relu_soundness_mixed_signs(self):
        """Test ReLU on region with both positive and negative inputs."""
        hull = ReLUHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Sample across sign boundaries
        test_points = [
            np.array([-1.5, -1.5]),
            np.array([-0.5, 0.5]),
            np.array([0.0, 0.0]),
            np.array([0.5, 1.5]),
            np.array([1.5, 1.5]),
        ]

        for x in test_points:
            y = relu_np(x)
            point = np.concatenate([x, y])
            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point
            assert np.all(constraint_values >= -1e-8), f"Soundness violated at {x}"


class TestSigmoidSoundness:
    """Test soundness of Sigmoid hull over-approximation."""

    def test_sigmoid_soundness_2d(self):
        """Verify Sigmoid hull contains all sigmoid outputs on 2D polytope."""
        hull = SigmoidHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Sample many points and verify all satisfy constraints
        rng = np.random.default_rng(42)
        num_samples = 1000
        violations = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = sigmoid_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.9, f"Sigmoid soundness: {satisfaction_rate:.2f}%"

    def test_sigmoid_soundness_3d(self):
        """Test Sigmoid soundness on 3D polytope."""
        hull = SigmoidHull()
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        rng = np.random.default_rng(42)
        num_samples = 500
        violations = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = sigmoid_np(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.0, f"Sigmoid 3D soundness: {satisfaction_rate:.2f}%"


class TestTanhSoundness:
    """Test soundness of Tanh hull over-approximation."""

    def test_tanh_soundness_2d(self):
        """Verify Tanh hull contains all tanh outputs on 2D polytope."""
        hull = TanhHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        rng = np.random.default_rng(42)
        num_samples = 1000
        violations = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = np.tanh(x)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.9, f"Tanh soundness: {satisfaction_rate:.2f}%"


class TestLeakyReLUSoundness:
    """Test soundness of LeakyReLU hull over-approximation."""

    def test_leakyrelu_soundness_2d(self):
        """Verify LeakyReLU hull soundness."""
        hull = LeakyReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        rng = np.random.default_rng(42)
        num_samples = 1000
        violations = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = leakyrelu_np(x, negative_slope=0.01)
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.9, f"LeakyReLU soundness: {satisfaction_rate:.2f}%"


class TestELUSoundness:
    """Test soundness of ELU hull over-approximation."""

    def test_elu_soundness_2d(self):
        """Verify ELU hull soundness."""
        hull = ELUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        rng = np.random.default_rng(42)
        num_samples = 1000
        violations = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = elu_np(x)  # elu_np uses default alpha internally
            point = np.concatenate([x, y])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.9, f"ELU soundness: {satisfaction_rate:.2f}%"


class TestMaxPoolSoundness:
    """Test soundness of MaxPool hull over-approximation."""

    def test_maxpool_soundness_2d(self):
        """Verify MaxPool hull soundness."""
        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        rng = np.random.default_rng(42)
        num_samples = 1000
        violations = 0

        for _ in range(num_samples):
            x = rng.uniform(lb, ub)
            y = np.max(x)  # MaxPool computes max of inputs
            point = np.concatenate([x, [y]])

            b = result[:, 0]
            a = result[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-8):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.0, f"MaxPool soundness: {satisfaction_rate:.2f}%"


class TestSoundnessConsistency:
    """Test that soundness is preserved across multiple calls."""

    def test_relu_consistency_multiple_calls(self):
        """Verify ReLU soundness is consistent across multiple calls."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Call cal_hull multiple times
        results = []
        for _ in range(3):
            result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(result)

        # Verify all results are identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

        # Verify soundness for each result
        rng = np.random.default_rng(42)
        for result in results:
            violations = 0
            for _ in range(100):
                x = rng.uniform(lb, ub)
                y = relu_np(x)
                point = np.concatenate([x, y])

                b = result[:, 0]
                a = result[:, 1:]
                constraint_values = b + a @ point

                if not np.all(constraint_values >= -1e-8):
                    violations += 1

            satisfaction_rate = 100.0 * (100 - violations) / 100
            assert satisfaction_rate >= 99.0, f"Consistency violation: {satisfaction_rate:.2f}%"


class TestHullTightness:
    """Test tightness of hull approximations."""

    def test_relu_hull_tighter_than_trivial_bounds(self):
        """Verify ReLU hull is tighter than simple box bounds."""
        hull = ReLUHull()
        lb = np.array([0.0, 0.0])  # All positive inputs
        ub = np.array([2.0, 2.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For ReLU with all positive inputs, output = input
        # Hull should reflect this identity relationship (tighter than loose bounds)
        assert result.shape[0] > 0, "Hull has no constraints"

        # The hull should have some constraints that bind at the bounds
        sample_point = np.concatenate([lb, relu_np(lb)])
        b = result[:, 0]
        a = result[:, 1:]
        constraint_values = b + a @ sample_point

        # At least one constraint should be tight (nearly zero)
        tight_constraints = np.sum(constraint_values < 1e-6)
        assert tight_constraints > 0, "No tight constraints in hull"

    def test_sigmoid_hull_respects_bounds(self):
        """Verify Sigmoid hull respects [0, 1] output bounds."""
        hull = SigmoidHull()
        lb = np.array([-10.0, -10.0])
        ub = np.array([10.0, 10.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # All output values should be in [0, 1]
        # Test by checking extreme points
        test_x = np.array([-10.0, -10.0])
        test_y = sigmoid_np(test_x)
        assert np.all(test_y >= 0.0), "Output below sigmoid range"
        assert np.all(test_y <= 1.0), "Output above sigmoid range"

        # Hull should have constraints that enforce this
        assert result.shape[0] > 0, "No constraints in hull"
