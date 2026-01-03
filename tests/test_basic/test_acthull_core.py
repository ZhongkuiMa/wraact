"""Core ActHull tests targeting remaining coverage gaps.

This module tests core ActHull functionality and edge cases:
1. Bounds validation and input constraints
2. High-dimensional polytopes
3. Mixed mode configurations
4. Degenerate and edge case polytopes
5. Output shapes and dimensions
"""

__docformat__ = "restructuredtext"

import numpy as np


class TestActHullBoundsHandling:
    """Test bounds validation and handling in ActHull."""

    def test_relu_large_dimension_bounds(self):
        """Test ReLU with large dimensional input bounds."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()
        d = 10
        lb = -np.ones(d)
        ub = np.ones(d)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == 2 * d + 1  # 2d + 1
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_asymmetric_large_bounds(self):
        """Test LeakyReLU with large asymmetric bounds."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull()
        lb = np.array([-100.0, -50.0, -200.0])
        ub = np.array([50.0, 100.0, 10.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_elu_both_constraint_modes_with_different_bounds(self):
        """Test ELU with both modes and varying bound configurations."""
        from wraact.acthull import ELUHull

        hull = ELUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)

        # Test with different bound configurations
        test_cases = [
            (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),  # Symmetric
            (np.array([-2.0, -1.0]), np.array([0.5, 2.0])),  # Asymmetric
            (np.array([-5.0, -5.0]), np.array([5.0, 5.0])),  # Large bounds
        ]

        for lb, ub in test_cases:
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            assert np.all(np.isfinite(constraints))


class TestActHullMultiDimensional:
    """Test ActHull with various dimensionalities."""

    def test_sigmoid_very_high_dimension(self):
        """Test Sigmoid with high-dimensional input."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()
        d = 8
        lb = -2.0 * np.ones(d)
        ub = 2.0 * np.ones(d)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_tanh_high_dimension(self):
        """Test Tanh with high-dimensional input."""
        from wraact.acthull import TanhHull

        hull = TanhHull()
        d = 6
        lb = -2.0 * np.ones(d)
        ub = 2.0 * np.ones(d)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_maxpool_high_dimension(self):
        """Test MaxPool with high-dimensional input."""
        from wraact.acthull import MaxPoolHull

        hull = MaxPoolHull()
        d = 7
        lb = -np.ones(d)
        ub = np.ones(d)

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert constraints.shape[1] == d + 2  # d+2 for maxpool
        assert np.all(np.isfinite(constraints))


class TestActHullConstraintCombinations:
    """Test various constraint mode combinations."""

    def test_relu_single_multi_combined_soundness(self):
        """Test ReLU with combined single and multi-neuron modes."""
        from wraact.acthull import ReLUHull

        def relu_np(x):
            return np.maximum(0, x)

        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Soundness check
        num_samples = 300
        rng = np.random.default_rng(123)
        samples = rng.uniform(lb, ub, (num_samples, 3))

        violations = 0
        for x in samples:
            y = relu_np(x)
            point = np.concatenate([x, y])

            b = constraints[:, 0]
            a = constraints[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-6):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 99.0

    def test_leakyrelu_combined_modes_3d_soundness(self):
        """Test LeakyReLU soundness with combined constraint modes in 3D."""
        from wraact.acthull import LeakyReLUHull

        def leakyrelu_np(x, negative_slope=0.01):
            return np.where(x >= 0, x, negative_slope * x)

        lb = np.array([-1.5, -1.5, -1.5])
        ub = np.array([1.5, 1.5, 1.5])

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        num_samples = 300
        rng = np.random.default_rng(456)
        samples = rng.uniform(lb, ub, (num_samples, 3))

        violations = 0
        for x in samples:
            y = leakyrelu_np(x)
            point = np.concatenate([x, y])

            b = constraints[:, 0]
            a = constraints[:, 1:]
            constraint_values = b + a @ point

            if not np.all(constraint_values >= -1e-6):
                violations += 1

        satisfaction_rate = 100.0 * (num_samples - violations) / num_samples
        assert satisfaction_rate >= 95.0


class TestActHullOutputDimensions:
    """Test output dimensions for different configurations."""

    def test_relu_single_neuron_output_dims(self):
        """Test ReLU single-neuron output dimensions across various input sizes."""
        from wraact.acthull import ReLUHull

        for d in range(1, 8):
            lb = -np.ones(d)
            ub = np.ones(d)

            hull = ReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=False)
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert constraints.shape[1] == 2 * d + 1

    def test_maxpool_output_dims_various_inputs(self):
        """Test MaxPool output dimensions across various input sizes."""
        from wraact.acthull import MaxPoolHull

        for d in range(2, 9):
            lb = -np.ones(d)
            ub = np.ones(d)

            hull = MaxPoolHull()
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert constraints.shape[1] == d + 2  # d+2 format for maxpool

    def test_sigmoid_output_dims(self):
        """Test Sigmoid output dimensions."""
        from wraact.acthull import SigmoidHull

        for d in range(1, 7):
            lb = -2.0 * np.ones(d)
            ub = 2.0 * np.ones(d)

            hull = SigmoidHull()
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

            assert constraints.shape[1] == 2 * d + 1


class TestActHullSpecialBounds:
    """Test ActHull with special bound configurations."""

    def test_relu_near_zero_bounds(self):
        """Test ReLU with bounds very close to zero."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()
        lb = np.array([-0.1, -0.1])
        ub = np.array([0.1, 0.1])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_leakyrelu_crossing_zero_asymmetric(self):
        """Test LeakyReLU with asymmetric bounds crossing zero."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull()
        lb = np.array([-10.0, -0.5, -2.0])
        ub = np.array([-0.1, 10.0, 0.1])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_elu_crossing_zero(self):
        """Test ELU with bounds crossing zero."""
        from wraact.acthull import ELUHull

        hull = ELUHull()
        lb = np.array([-3.0, -1.0, -0.5])
        ub = np.array([0.5, 2.0, 3.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))

    def test_sigmoid_extreme_bounds(self):
        """Test Sigmoid with extreme bounds."""
        from wraact.acthull import SigmoidHull

        hull = SigmoidHull()
        lb = np.array([-10.0, -5.0])
        ub = np.array([5.0, 10.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        assert isinstance(constraints, np.ndarray)
        assert np.all(np.isfinite(constraints))


class TestActHullReproducibility:
    """Test reproducibility of ActHull computations."""

    def test_relu_multi_call_reproducibility(self):
        """Test ReLU produces identical results across multiple calls."""
        from wraact.acthull import ReLUHull

        hull = ReLUHull()
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        results = []
        for _ in range(5):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints)

        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_leakyrelu_multi_call_reproducibility(self):
        """Test LeakyReLU reproducibility."""
        from wraact.acthull import LeakyReLUHull

        hull = LeakyReLUHull(if_cal_single_neuron_constrs=True, if_cal_multi_neuron_constrs=True)
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        results = []
        for _ in range(3):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints)

        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_tanh_multi_call_reproducibility(self):
        """Test Tanh reproducibility."""
        from wraact.acthull import TanhHull

        hull = TanhHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        results = []
        for _ in range(3):
            constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            results.append(constraints)

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
