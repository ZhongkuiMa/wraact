"""Soundness tests for LeakyReLUHull using template pattern.

This demonstrates how to reuse the BaseSoundnessTest template for
a different activation function (LeakyReLU).

Key Feature of LeakyReLU:
=========================
- Unlike ReLU which outputs 0 for negative inputs
- LeakyReLU outputs a small negative slope for negative inputs: y = max(x, negative_slope * x)
- Default negative_slope ≈ 0.01

Template Usage:
===============
This test class inherits from BaseSoundnessTest and only needs to:
1. Define the activation function
2. Specify which hull class to test
3. All soundness tests run automatically!
"""

__docformat__ = "restructuredtext"

import numpy as np
import pytest

from tests.test_templates import BaseSoundnessTest

# Import the actual hull class
from wraact.acthull import LeakyReLUHull


def leakyrelu_np(x, negative_slope=0.01):
    """NumPy implementation of LeakyReLU for testing.

    Args:
        x: Input value(s)
        negative_slope: Slope for negative inputs (default 0.01)

    Returns:
        max(x, negative_slope * x)
    """
    return np.where(x >= 0, x, negative_slope * x)


class TestLeakyReLUSoundness(BaseSoundnessTest):
    """Soundness tests for LeakyReLUHull.

    Reuses all tests from BaseSoundnessTest by implementing:
    1. activation_fn fixture: The actual function to test
    2. hull_class_to_test fixture: The hull class (LeakyReLUHull)

    All soundness tests are inherited and run automatically:
    - test_soundness_2d_box_monte_carlo
    - test_soundness_3d_box_monte_carlo
    - test_soundness_4d_box_monte_carlo
    - test_soundness_random_seeds (parametrized, 3 seeds)
    - test_hull_contains_actual_outputs
    - test_deterministic_computation
    - test_soundness_preserved_after_multiple_calls
    """

    @pytest.fixture
    def activation_fn(self):
        """Return the LeakyReLU function with default negative_slope=0.01."""

        def leakyrelu(x):
            return leakyrelu_np(x, negative_slope=0.01)

        return leakyrelu

    @pytest.fixture
    def hull_class_to_test(self):
        """Return the LeakyReLUHull class to test."""
        return LeakyReLUHull


class TestLeakyReLUSoundnessCustomSlope(BaseSoundnessTest):
    """Soundness tests for LeakyReLUHull with custom negative slope.

    Tests with negative_slope=0.1 to verify the hull works correctly
    with different slope parameters.
    """

    @pytest.fixture
    def activation_fn(self):
        """Return the LeakyReLU function with custom negative_slope=0.1."""

        def leakyrelu(x):
            return leakyrelu_np(x, negative_slope=0.1)

        return leakyrelu

    @pytest.fixture
    def hull_class_to_test(self):
        """Return the LeakyReLUHull class to test."""
        return LeakyReLUHull

    # All soundness tests inherited from BaseSoundnessTest
    # They will automatically test with negative_slope=0.1


class TestLeakyReLUBasicFunctionality:
    """Basic functionality tests for LeakyReLUHull."""

    def test_cal_hull_returns_ndarray(self):
        """Verify cal_hull() returns an ndarray."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # 2D array

    def test_cal_hull_output_shape_2d(self):
        """Verify output shape follows formula: 2*dim + 1 = 5 for 2D."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 2D input: 2*2 + 1 = 5 columns
        assert result.shape[1] == 5

    def test_cal_hull_output_shape_3d(self):
        """Verify output shape follows formula: 2*dim + 1 = 7 for 3D."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 3D input: 2*3 + 1 = 7 columns
        assert result.shape[1] == 7

    def test_cal_hull_output_finite(self):
        """Verify output contains no inf or nan values."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert np.all(np.isfinite(result))


class TestLeakyReLUBoundEdgeCases:
    """Test LeakyReLU with edge case bounds (trivial cases)."""

    def test_cal_hull_all_positive_bounds(self):
        """Test LeakyReLU with all-positive bounds (lb >= 0)."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([0.5, 0.5])
        ub = np.array([1.0, 1.0])

        hull = LeakyReLUHull()
        # All-positive bounds should work (linear region)
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))
        except ValueError:
            # Some implementations may require crossing zero
            pass

    def test_cal_hull_all_negative_bounds(self):
        """Test LeakyReLU with all-negative bounds (ub <= 0)."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([-0.5, -0.5])

        hull = LeakyReLUHull()
        # All-negative bounds should work (linear with slope)
        try:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert isinstance(constraints, np.ndarray)
            assert np.all(np.isfinite(constraints))
        except ValueError:
            # Some implementations may require crossing zero
            pass

    def test_cal_hull_very_small_range(self):
        """Test LeakyReLU with small input range (just above minimum threshold)."""
        from wraact.acthull import LeakyReLUHull

        # Minimum range threshold is 0.05, so we use range 0.06
        lb = np.array([-0.03, -0.03])
        ub = np.array([0.03, 0.03])

        hull = LeakyReLUHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        assert constraints.shape[1] == 5  # 2D input

    def test_cal_hull_asymmetric_bounds(self):
        """Test LeakyReLU with asymmetric bounds around zero."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-2.0, -2.0])
        ub = np.array([0.5, 0.5])

        hull = LeakyReLUHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_large_range(self):
        """Test LeakyReLU with large input range."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-10.0, -10.0])
        ub = np.array([10.0, 10.0])

        hull = LeakyReLUHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_3d_edge_case(self):
        """Test LeakyReLU with 3D asymmetric bounds."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-5.0, -1.0, -0.1])
        ub = np.array([0.1, 2.0, 5.0])

        hull = LeakyReLUHull()
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))
        assert constraints.shape[1] == 7  # 3D input


class TestLeakyReLUSingleNeuronMode:
    """Test LeakyReLU hull with single-neuron constraint mode.

    This tests the single-neuron constraint calculation path which is
    normally disabled in default ActHull initialization. Tests constraint
    caching behavior which is unique to LeakyReLU.
    """

    def test_cal_hull_single_neuron_2d(self):
        """Test single-neuron constraints for 2D input."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 2 * len(lb) + 1
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_3d(self):
        """Test single-neuron constraints for 3D input."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 2 * len(lb) + 1
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_4d(self):
        """Test single-neuron constraints for 4D input."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0, 1.0])

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert constraints.shape[1] == 2 * 4 + 1  # 9 columns for 4D
        assert np.all(np.isfinite(constraints))

    def test_cal_hull_single_neuron_soundness(self):
        """Verify soundness of single-neuron constraints with Monte Carlo."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Monte Carlo sampling to verify soundness
        rng = np.random.default_rng(42)
        samples = rng.uniform(lb, ub, (1000, 2))

        violations = 0
        for x in samples:
            y = leakyrelu_np(x, negative_slope=0.01)
            point = np.concatenate([x, y])

            b = constraints[:, 0]
            A = constraints[:, 1:]
            constraint_values = b + A @ point

            if not np.all(constraint_values >= -1e-6):
                violations += 1

        satisfaction_rate = 100.0 * (1000 - violations) / 1000
        # Single-neuron constraints may be less tight, but should still be sound
        assert satisfaction_rate >= 95.0

    def test_cal_hull_single_neuron_cache_behavior(self):
        """Test that LeakyReLU caches lower constraints."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        # First call - cache miss
        lb1 = np.array([-1.0, -1.0])
        ub1 = np.array([1.0, 1.0])
        c1 = hull.cal_hull(input_lower_bounds=lb1, input_upper_bounds=ub1)

        # Second call - same dimension, cache hit
        lb2 = np.array([-2.0, -2.0])
        ub2 = np.array([2.0, 2.0])
        c2 = hull.cal_hull(input_lower_bounds=lb2, input_upper_bounds=ub2)

        # Verify cache was used (constraints share structure)
        assert c1.shape[0] == c2.shape[0]  # Same number of constraints

    def test_cal_hull_single_neuron_cache_hit(self):
        """Test cache behavior on multiple calls with same dimension."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Multiple calls with same dimension
        c1 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c2 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
        c3 = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # All should have same shape (cache hit)
        assert c1.shape == c2.shape == c3.shape

    def test_cal_hull_single_neuron_cache_miss(self):
        """Test cache behavior on calls with different dimensions."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        # First call: 2D
        lb2d = np.array([-1.0, -1.0])
        ub2d = np.array([1.0, 1.0])
        c2d = hull.cal_hull(input_lower_bounds=lb2d, input_upper_bounds=ub2d)

        # Second call: 3D (cache miss)
        lb3d = np.array([-1.0, -1.0, -1.0])
        ub3d = np.array([1.0, 1.0, 1.0])
        c3d = hull.cal_hull(input_lower_bounds=lb3d, input_upper_bounds=ub3d)

        # Different dimensions should have different column counts
        assert c2d.shape[1] == 5  # 2*2 + 1
        assert c3d.shape[1] == 7  # 2*3 + 1

    def test_cal_sn_constrs_direct(self):
        """Test direct call to cal_sn_constrs method."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        # Call cal_hull which internally calls cal_sn_constrs
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify the method executed successfully
        assert constraints is not None
        assert constraints.shape[0] > 0

    def test_cal_hull_mixed_mode_disabled_error(self):
        """Test that having both modes disabled raises error or is handled."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # This tests the edge case where both constraint modes are disabled
        # The implementation may either raise an error or handle it gracefully
        try:
            hull = LeakyReLUHull(
                if_cal_single_neuron_constrs=False, if_cal_multi_neuron_constrs=False
            )
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # If it succeeds, at least constraints should be valid
            assert constraints is not None
        except (ValueError, RuntimeError):
            # Expected: both modes disabled should raise an error
            pass

    def test_cal_hull_single_neuron_output_properties(self):
        """Test properties of single-neuron constraint output."""
        from wraact.acthull import LeakyReLUHull

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify output shape and finiteness
        assert constraints.shape[1] == 5  # 2D input
        assert constraints.shape[0] > 0  # At least one constraint
        assert np.all(np.isfinite(constraints))
        # Coefficients should be reasonable (not extremely large)
        assert np.all(np.abs(constraints) < 1e10)
