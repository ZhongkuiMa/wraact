"""End-to-end pipeline tests for hull computation.

This module tests complete workflows including constraint generation,
vertex computation, and multi-layer composition.

Key Tests:
==========
- Full cal_hull() pipeline correctness
- Constraint and vertex consistency
- Multi-layer composition
- Fraction vs float mode
"""

__docformat__ = "restructuredtext"

import numpy as np

from wraact._functions import relu_np, sigmoid_np
from wraact.acthull import MaxPoolHull, ReLUHull, SigmoidHull


class TestReLUPipeline:
    """Test complete ReLU hull computation pipeline."""

    def test_relu_full_pipeline_2d(self):
        """Test full cal_hull() pipeline for 2D ReLU."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Compute hull
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify output format
        assert isinstance(constraints, np.ndarray), "Output is not ndarray"
        assert constraints.ndim == 2, "Output is not 2D"
        assert constraints.shape[1] == 5, (
            f"Expected 5 columns (bias + 2 inputs + 2 outputs), got {constraints.shape[1]}"
        )

        # Verify constraints are well-formed
        assert np.all(np.isfinite(constraints)), "Constraints contain inf/nan"
        assert constraints.shape[0] > 0, "No constraints generated"

    def test_relu_pipeline_deterministic(self):
        """Verify ReLU pipeline is deterministic."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Compute hull multiple times
        results = [hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub) for _ in range(3)]

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_sigmoid_pipeline_3d(self):
        """Test complete Sigmoid hull computation pipeline for 3D."""
        hull = SigmoidHull()
        lb = np.array([-2.0, -2.0, -2.0])
        ub = np.array([2.0, 2.0, 2.0])

        # Compute hull
        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify output format
        assert isinstance(constraints, np.ndarray), "Output is not ndarray"
        assert constraints.ndim == 2, "Output is not 2D"
        assert constraints.shape[1] == 7, (
            f"Expected 7 columns (bias + 3 inputs + 3 outputs), got {constraints.shape[1]}"
        )

        # Verify constraints are well-formed
        assert np.all(np.isfinite(constraints)), "Constraints contain inf/nan"
        assert constraints.shape[0] > 0, "No constraints generated"


class TestMultiLayerComposition:
    """Test composing multiple activation layers."""

    def test_relu_then_sigmoid_composition(self):
        """Test composing ReLU output as Sigmoid input."""
        # Layer 1: ReLU
        relu_hull = ReLUHull()
        lb_layer1 = np.array([-1.0, -1.0])
        ub_layer1 = np.array([1.0, 1.0])
        relu_constraints = relu_hull.cal_hull(
            input_lower_bounds=lb_layer1, input_upper_bounds=ub_layer1
        )

        # The output of ReLU is [0, 1] for inputs in [-1, 1]
        # Use these as bounds for Layer 2
        lb_layer2 = np.array([0.0, 0.0])
        ub_layer2 = np.array([1.0, 1.0])

        # Layer 2: Sigmoid
        sigmoid_hull = SigmoidHull()
        sigmoid_constraints = sigmoid_hull.cal_hull(
            input_lower_bounds=lb_layer2, input_upper_bounds=ub_layer2
        )

        # Both should have valid constraints
        assert relu_constraints.shape[0] > 0, "ReLU layer has no constraints"
        assert sigmoid_constraints.shape[0] > 0, "Sigmoid layer has no constraints"
        assert np.all(np.isfinite(relu_constraints)), "ReLU constraints contain inf/nan"
        assert np.all(np.isfinite(sigmoid_constraints)), "Sigmoid constraints contain inf/nan"

    def test_relu_multi_layer_stack(self):
        """Test stacking multiple ReLU layers."""
        hull = ReLUHull()

        # First layer
        lb1 = np.array([-1.0, -1.0])
        ub1 = np.array([1.0, 1.0])
        constraints1 = hull.cal_hull(input_lower_bounds=lb1, input_upper_bounds=ub1)

        # For simplicity, ReLU with [-1, 1] input produces [0, 1] output
        # Second layer
        lb2 = np.array([0.0, 0.0])
        ub2 = np.array([1.0, 1.0])
        constraints2 = hull.cal_hull(input_lower_bounds=lb2, input_upper_bounds=ub2)

        # Third layer (ReLU input bounded by previous layer output)
        lb3 = np.array([0.0, 0.0])
        ub3 = np.array([1.0, 1.0])
        constraints3 = hull.cal_hull(input_lower_bounds=lb3, input_upper_bounds=ub3)

        # All should be valid
        for i, constraints in enumerate([constraints1, constraints2, constraints3], 1):
            assert constraints.shape[0] > 0, f"Layer {i} has no constraints"
            assert np.all(np.isfinite(constraints)), f"Layer {i} contains inf/nan"


class TestConstraintVectorConsistency:
    """Test consistency of constraint vectors across formats."""

    def test_constraint_format_consistency(self):
        """Verify constraint vector format is consistent."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Each row should be: [b, a1, a2, y1, y2]
        # Representing: b + a1*x1 + a2*x2 + y1*out1 + y2*out2 >= 0
        assert result.shape[1] == 5, "Inconsistent constraint format"

        # All values should be finite
        assert np.all(np.isfinite(result)), "Contains inf/nan"

        # Bias term (first column) should be bounded
        bias_terms = result[:, 0]
        assert np.all(np.abs(bias_terms) < 1e6), "Bias terms too large"

    def test_multioutput_constraints(self):
        """Test that constraints properly account for multiple outputs."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # For 3D input, should have 3D output
        # Constraint format: [b, a_x1, a_x2, a_x3, a_y1, a_y2, a_y3]
        expected_cols = 1 + 3 + 3  # bias + 3 inputs + 3 outputs
        assert result.shape[1] == expected_cols, (
            f"Expected {expected_cols} columns, got {result.shape[1]}"
        )


class TestSoundnessAcrossPipeline:
    """Test that soundness is maintained through the pipeline."""

    def test_relu_soundness_pipeline(self):
        """Verify soundness throughout ReLU pipeline."""
        hull = ReLUHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Sample and verify soundness
        test_points = [lb, ub, (lb + ub) / 2]

        for x in test_points:
            y = relu_np(x)
            point = np.concatenate([x, y])

            b = constraints[:, 0]
            a = constraints[:, 1:]
            constraint_values = b + a @ point

            assert np.all(constraint_values >= -1e-8), f"Soundness violated at {x}"

    def test_sigmoid_soundness_pipeline(self):
        """Verify soundness throughout Sigmoid pipeline."""
        hull = SigmoidHull()
        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        constraints = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Sample and verify soundness
        test_points = [lb, ub, (lb + ub) / 2]

        for x in test_points:
            y = sigmoid_np(x)
            point = np.concatenate([x, y])

            b = constraints[:, 0]
            a = constraints[:, 1:]
            constraint_values = b + a @ point

            assert np.all(constraint_values >= -1e-8), f"Soundness violated at {x}"


class TestBoundaryConditions:
    """Test behavior at boundary conditions during pipeline."""

    def test_pipeline_at_zero_bounds(self):
        """Test pipeline with bounds at origin."""
        hull = ReLUHull()
        lb = np.array([0.0, 0.0])
        ub = np.array([0.0, 0.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should still produce valid constraints
        assert result.shape[0] > 0, "No constraints at origin"
        assert np.all(np.isfinite(result)), "Constraints at origin contain inf/nan"

    def test_pipeline_at_extreme_point(self):
        """Test pipeline at extreme input point."""
        hull = SigmoidHull()
        lb = np.array([-100.0, -100.0])
        ub = np.array([-100.0, -100.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Should handle extreme points gracefully
        assert np.all(np.isfinite(result)), "Extreme point produced inf/nan"


class TestMaxPoolPipeline:
    """Test MaxPool hull pipeline."""

    def test_maxpool_pipeline_2d(self):
        """Test complete MaxPool pipeline for 2D."""
        hull = MaxPoolHull()
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

        # Verify output format: [b, a_x1, a_x2, a_y]
        expected_cols = 1 + 2 + 1  # bias + 2 inputs + 1 output (max of inputs)
        assert result.shape[1] == expected_cols, (
            f"Expected {expected_cols} columns, got {result.shape[1]}"
        )

        # Verify constraints are valid
        assert np.all(np.isfinite(result)), "MaxPool constraints contain inf/nan"
        assert result.shape[0] > 0, "No MaxPool constraints"
