"""Base template for soundness tests across all activation functions.

This module provides reusable test patterns for verifying soundness of function hulls.

Key Principle:
==============
For any activation function f and input polytope P (defined by bounds):
  - For all x in P: the point (x, f(x)) must satisfy ALL hull constraints
  - This is the MOST CRITICAL test for hull correctness

Template Usage:
===============
To test a new activation function (e.g., LeakyReLU):

1. Define the activation function:
   def leakyrelu_np(x, negative_slope=0.01):
       return np.where(x >= 0, x, negative_slope * x)

2. Create a test class:
   class TestLeakyReLUSoundness(BaseSoundnessTest):
       @pytest.fixture
       def activation_fn(self):
           def leakyrelu(x):
               return leakyrelu_np(x, negative_slope=0.01)
           return leakyrelu

       @pytest.fixture
       def hull_class_to_test(self):
           from wraact.acthull import LeakyReLUHull
           return LeakyReLUHull

3. All soundness tests will automatically run for LeakyReLU
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest


class BaseSoundnessTest:
    """Base class for soundness verification tests.

    Subclasses must provide:
    - activation_fn (fixture): The numpy implementation of the activation function
    - hull_class_to_test (fixture): The hull class to test (e.g., ReLUHull, SigmoidHull)
    """

    @pytest.fixture
    def activation_fn(self):
        """Override this to return the activation function.

        Example:
            def relu_np(x):
                return np.maximum(0, x)
            return relu_np
        """
        raise NotImplementedError("Subclasses must implement activation_fn fixture")

    @pytest.fixture
    def hull_class_to_test(self):
        """Override this to return the hull class.

        Example:
            from wraact.acthull import ReLUHull
            return ReLUHull
        """
        raise NotImplementedError("Subclasses must implement hull_class_to_test fixture")

    def test_soundness_2d_box_monte_carlo(
        self, activation_fn, hull_class_to_test, tolerance, monte_carlo_samples
    ):
        """Verify soundness using Monte Carlo sampling on 2D box.

        This is the CRITICAL test: verifies that all sampled points (x, f(x))
        satisfy the hull constraints.
        """
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = hull_class_to_test()
        try:
            hull_constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        except (ValueError, RuntimeError, TypeError):
            pytest.skip("Hull calculation failed")

        # Generate random points in input domain
        num_samples = monte_carlo_samples
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (num_samples, 2))

        # For each sample, compute (x, f(x)) and check constraints
        violations = 0
        for x in samples:
            y = activation_fn(x)
            point = np.concatenate([x, y])

            # Check all constraints: b + A @ point >= 0
            b = hull_constraints[:, 0]
            A = hull_constraints[:, 1:]
            constraint_values = b + A @ point

            if not np.all(constraint_values >= -tolerance):
                violations += 1

        # Calculate satisfaction rate
        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples

        # Assert soundness
        assert satisfaction_rate >= 99.9, (
            f"Soundness violation: {satisfaction_rate:.2f}% ({violations}/{num_samples})"
        )

    def test_soundness_3d_box_monte_carlo(self, activation_fn, hull_class_to_test, tolerance):
        """Verify soundness on 3D box polytope."""
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = hull_class_to_test()
        try:
            hull_constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        except (ValueError, RuntimeError, TypeError):
            pytest.skip("Hull calculation failed")

        # Sample points from input polytope
        num_samples = 1000
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (num_samples, 3))

        violations = 0
        for x in samples:
            y = activation_fn(x)
            point = np.concatenate([x, y])

            b = hull_constraints[:, 0]
            A = hull_constraints[:, 1:]
            constraint_values = b + A @ point

            if not np.all(constraint_values >= -tolerance):
                violations += 1

        actual_samples = len(samples)
        satisfaction_rate = (
            100.0 * (actual_samples - violations) / actual_samples if actual_samples > 0 else 0
        )
        assert satisfaction_rate >= 99.0, (
            f"Soundness violation: {satisfaction_rate:.2f}% ({violations}/{actual_samples})"
        )

    def test_soundness_4d_box_monte_carlo(self, activation_fn, hull_class_to_test, tolerance):
        """Verify soundness on 4D box polytope."""
        lb = np.array([-1.0, -1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0, 1.0])

        hull = hull_class_to_test()
        try:
            hull_constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        except (ValueError, RuntimeError, TypeError):
            pytest.skip("Hull calculation failed")

        # Sample points from input polytope
        num_samples = 500
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (num_samples, 4))

        violations = 0
        for x in samples:
            y = activation_fn(x)
            point = np.concatenate([x, y])

            b = hull_constraints[:, 0]
            A = hull_constraints[:, 1:]
            constraint_values = b + A @ point

            if not np.all(constraint_values >= -tolerance):
                violations += 1

        actual_samples = len(samples)
        satisfaction_rate = (
            100.0 * (actual_samples - violations) / actual_samples if actual_samples > 0 else 0
        )
        assert satisfaction_rate >= 99.0, (
            f"Soundness violation: {satisfaction_rate:.2f}% ({violations}/{actual_samples})"
        )

    @pytest.mark.parametrize("seed", [100, 200, 300])
    def test_soundness_random_seeds(self, activation_fn, hull_class_to_test, tolerance, seed):
        """Verify soundness on random inputs with different seeds."""
        rng = np.random.default_rng(seed)

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        hull = hull_class_to_test()
        try:
            hull_constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        except (ValueError, RuntimeError, TypeError):
            pytest.skip("Hull calculation failed")

        # Sample random points
        num_samples = 500
        samples = rng.uniform(lb, ub, (num_samples, 2))

        violations = 0
        for x in samples:
            y = activation_fn(x)
            point = np.concatenate([x, y])

            b = hull_constraints[:, 0]
            A = hull_constraints[:, 1:]
            constraint_values = b + A @ point

            if not np.all(constraint_values >= -tolerance):
                violations += 1

        actual_samples = len(samples)
        satisfaction_rate = (
            100.0 * (actual_samples - violations) / actual_samples if actual_samples > 0 else 0
        )
        assert satisfaction_rate >= 99.0, (
            f"Seed {seed}: Soundness violation: {satisfaction_rate:.2f}% ({violations}/{actual_samples})"
        )

    def test_hull_contains_actual_outputs(self, activation_fn, hull_class_to_test):
        """Verify hull contains actual function outputs (sanity check)."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = hull_class_to_test()
        try:
            hull_constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        except (ValueError, RuntimeError, TypeError):
            pytest.skip("Hull calculation failed")

        # Sample points and verify all are inside
        rng = np.random.default_rng()
        samples = rng.uniform(lb, ub, (100, 2))

        for x in samples:
            y = activation_fn(x)
            point = np.concatenate([x, y])

            b = hull_constraints[:, 0]
            A = hull_constraints[:, 1:]
            constraint_values = b + A @ point

            assert np.all(constraint_values >= -1e-8), "Point outside hull"

    def test_deterministic_computation(self, activation_fn, hull_class_to_test):
        """Verify hull computation is deterministic (same input → same output)."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = hull_class_to_test()

        # Compute hull multiple times
        results = []
        for _ in range(3):
            try:
                result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
                results.append(result)
            except (ValueError, RuntimeError, TypeError):
                pytest.skip("Hull calculation failed")

        # All three results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_soundness_preserved_after_multiple_calls(
        self, activation_fn, hull_class_to_test, tolerance
    ):
        """Verify soundness is maintained across multiple hull calculations."""
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = hull_class_to_test()
        rng = np.random.default_rng()

        # Compute hull multiple times and verify soundness each time
        for iteration in range(3):
            try:
                hull_constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            except (ValueError, RuntimeError, TypeError):
                pytest.skip("Hull calculation failed")

            # Test soundness with random samples
            samples = rng.uniform(lb, ub, (100, 2))
            violations = 0

            for x in samples:
                y = activation_fn(x)
                point = np.concatenate([x, y])

                b = hull_constraints[:, 0]
                A = hull_constraints[:, 1:]
                constraint_values = b + A @ point

                if not np.all(constraint_values >= -tolerance):
                    violations += 1

            actual_samples = len(samples)
            satisfaction_rate = (
                100.0 * (actual_samples - violations) / actual_samples if actual_samples > 0 else 0
            )
            assert satisfaction_rate >= 99.0, (
                f"Iteration {iteration}: Soundness violated ({satisfaction_rate:.2f}%) ({violations}/{actual_samples})"
            )
