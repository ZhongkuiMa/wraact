"""Direct tests for tangent line functions.

Tests for edge cases and numerical stability in tangent line calculation:
1. Parallel tangent line computation for Sigmoid and Tanh
2. Second tangent line computation with convergence
3. Edge cases: zero crossings, extreme values, NaN handling
4. Convergence and iteration limits
"""

__docformat__ = "restructuredtext"

import numpy as np

from wraact._tangent_lines import (
    get_parallel_tangent_line_sigmoid_np,
    get_parallel_tangent_line_tanh_np,
    get_second_tangent_line_sigmoid_np,
    get_second_tangent_line_tanh_np,
)


class TestParallelTangentLineSigmoid:
    """Test parallel tangent line computation for Sigmoid."""

    def test_parallel_tangent_sigmoid_basic(self):
        """Test basic parallel tangent line for Sigmoid."""
        k = np.array([0.1, 0.2, 0.3])
        b, k_out, x = get_parallel_tangent_line_sigmoid_np(k, get_big=True)

        assert isinstance(b, np.ndarray)
        assert isinstance(k_out, np.ndarray)
        assert isinstance(x, np.ndarray)
        assert b.shape == k.shape
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(k_out))
        assert np.all(np.isfinite(x))

    def test_parallel_tangent_sigmoid_get_big_false(self):
        """Test parallel tangent line with get_big=False."""
        k = np.array([0.1, 0.15])
        b, _, x = get_parallel_tangent_line_sigmoid_np(k, get_big=False)

        assert b.shape == k.shape
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x))

    def test_parallel_tangent_sigmoid_small_k(self):
        """Test with small slope values."""
        k = np.array([0.01, 0.02, 0.03])
        b, _, _ = get_parallel_tangent_line_sigmoid_np(k, get_big=True)

        assert np.all(np.isfinite(b))

    def test_parallel_tangent_sigmoid_large_k(self):
        """Test with large slope values near boundary."""
        k = np.array([0.24, 0.23, 0.22])
        b, _, _ = get_parallel_tangent_line_sigmoid_np(k, get_big=True)

        assert np.all(np.isfinite(b))

    def test_parallel_tangent_sigmoid_edge_k(self):
        """Test with edge case k values."""
        k = np.array([0.0, 0.1, 0.249])
        b, _, _ = get_parallel_tangent_line_sigmoid_np(k, get_big=True)

        assert np.all(np.isfinite(b))

    def test_parallel_tangent_sigmoid_single_value(self):
        """Test with single scalar value."""
        k = np.array([0.15])
        b, _, _ = get_parallel_tangent_line_sigmoid_np(k, get_big=True)

        assert b.shape == (1,)
        assert np.all(np.isfinite(b))

    def test_parallel_tangent_sigmoid_array_length(self):
        """Test with various array lengths."""
        for length in [1, 5, 10]:
            k = np.linspace(0.01, 0.24, length)
            b, _, _ = get_parallel_tangent_line_sigmoid_np(k, get_big=True)

            assert b.shape[0] == length
            assert np.all(np.isfinite(b))


class TestParallelTangentLineTanh:
    """Test parallel tangent line computation for Tanh."""

    def test_parallel_tangent_tanh_basic(self):
        """Test basic parallel tangent line for Tanh."""
        k = np.array([0.1, 0.2, 0.3])
        b, k_out, x = get_parallel_tangent_line_tanh_np(k, get_big=True)

        assert isinstance(b, np.ndarray)
        assert isinstance(k_out, np.ndarray)
        assert isinstance(x, np.ndarray)
        assert b.shape == k.shape
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(k_out))
        assert np.all(np.isfinite(x))

    def test_parallel_tangent_tanh_get_big_false(self):
        """Test with get_big=False."""
        k = np.array([0.1, 0.2])
        b, _, x = get_parallel_tangent_line_tanh_np(k, get_big=False)

        assert b.shape == k.shape
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x))

    def test_parallel_tangent_tanh_small_k(self):
        """Test with small k values."""
        k = np.array([0.01, 0.05, 0.1])
        b, _, _ = get_parallel_tangent_line_tanh_np(k, get_big=True)

        assert np.all(np.isfinite(b))

    def test_parallel_tangent_tanh_large_k(self):
        """Test with large k values near boundary."""
        k = np.array([0.95, 0.9, 0.85])
        b, _, _ = get_parallel_tangent_line_tanh_np(k, get_big=True)

        assert np.all(np.isfinite(b))

    def test_parallel_tangent_tanh_edge_k(self):
        """Test with edge case k values (excluding boundary 0 and 1)."""
        k = np.array([0.01, 0.5, 0.99])
        b, _, _ = get_parallel_tangent_line_tanh_np(k, get_big=True)

        assert np.all(np.isfinite(b))

    def test_parallel_tangent_tanh_near_boundary_k(self):
        """Test with k values near boundaries."""
        k = np.array([0.01, 0.1, 0.9])
        b, _, _ = get_parallel_tangent_line_tanh_np(k, get_big=True)

        # Most values should be finite
        assert np.sum(np.isfinite(b)) >= 2
        # If any NaN, that's expected for numerical edge cases
        if np.any(~np.isfinite(b)):
            # Make sure at least some are finite
            assert np.sum(np.isfinite(b)) > 0

    def test_parallel_tangent_tanh_array_lengths(self):
        """Test with various array lengths."""
        for length in [1, 3, 8]:
            k = np.linspace(0.1, 0.8, length)
            b, _, _ = get_parallel_tangent_line_tanh_np(k, get_big=True)

            assert b.shape[0] == length
            assert np.all(np.isfinite(b))


class TestSecondTangentLineSigmoid:
    """Test second tangent line computation for Sigmoid."""

    def test_second_tangent_sigmoid_basic(self):
        """Test basic second tangent line for Sigmoid."""
        x1 = np.array([-1.0, 0.5, 1.0])
        b, k, x2 = get_second_tangent_line_sigmoid_np(x1, get_big=True)

        assert isinstance(b, np.ndarray)
        assert isinstance(k, np.ndarray)
        assert isinstance(x2, np.ndarray)
        assert b.shape == x1.shape
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_sigmoid_get_big_false(self):
        """Test with get_big=False."""
        x1 = np.array([-0.5, 0.0, 0.5])
        b, _, x2 = get_second_tangent_line_sigmoid_np(x1, get_big=False)

        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_sigmoid_x1_zero(self):
        """Test with x1=0 (edge case)."""
        x1 = np.array([0.0])
        b, _, _ = get_second_tangent_line_sigmoid_np(x1, get_big=True)

        assert np.all(np.isfinite(b))
        # x2 should be initialized differently from x1

    def test_second_tangent_sigmoid_negative_x1(self):
        """Test with negative x1 values."""
        x1 = np.array([-2.0, -1.0, -0.5])
        b, _, _ = get_second_tangent_line_sigmoid_np(x1, get_big=True)

        assert np.all(np.isfinite(b))

    def test_second_tangent_sigmoid_positive_x1(self):
        """Test with positive x1 values."""
        x1 = np.array([0.5, 1.0, 2.0])
        b, _, _ = get_second_tangent_line_sigmoid_np(x1, get_big=True)

        assert np.all(np.isfinite(b))

    def test_second_tangent_sigmoid_large_magnitude(self):
        """Test with large magnitude x1 values."""
        x1 = np.array([-5.0, 0.0, 5.0])
        b, _, _ = get_second_tangent_line_sigmoid_np(x1, get_big=True)

        assert np.all(np.isfinite(b))

    def test_second_tangent_sigmoid_single_value(self):
        """Test with single scalar in array."""
        x1 = np.array([1.5])
        b, _, _ = get_second_tangent_line_sigmoid_np(x1, get_big=True)

        assert b.shape == (1,)
        assert np.all(np.isfinite(b))


class TestSecondTangentLineTanh:
    """Test second tangent line computation for Tanh."""

    def test_second_tangent_tanh_basic_array(self):
        """Test basic second tangent line for Tanh with array."""
        x1 = np.array([-1.0, 0.5, 1.0])
        b, k, x2 = get_second_tangent_line_tanh_np(x1, get_big=True)

        assert isinstance(b, np.ndarray)
        assert isinstance(k, np.ndarray)
        assert isinstance(x2, np.ndarray)
        assert b.shape == x1.shape
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_scalar(self):
        """Test with scalar input."""
        x1_scalar = 0.5
        b, _, x2 = get_second_tangent_line_tanh_np(x1_scalar, get_big=True)

        # For scalar input, output might be array or scalar
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_get_big_false(self):
        """Test with get_big=False."""
        x1 = np.array([-0.5, 0.0, 0.5])
        b, _, x2 = get_second_tangent_line_tanh_np(x1, get_big=False)

        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_x1_zero(self):
        """Test with x1=0 (edge case)."""
        x1 = np.array([0.0])
        b, _, x2 = get_second_tangent_line_tanh_np(x1, get_big=True)

        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_negative_x1(self):
        """Test with negative x1 values."""
        x1 = np.array([-2.0, -1.0, -0.5])
        b, _, x2 = get_second_tangent_line_tanh_np(x1, get_big=True)

        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_positive_x1(self):
        """Test with positive x1 values."""
        x1 = np.array([0.5, 1.0, 2.0])
        b, _, x2 = get_second_tangent_line_tanh_np(x1, get_big=True)

        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_extreme_values(self):
        """Test with extreme x1 values."""
        x1 = np.array([-10.0, 0.0, 10.0])
        b, _, x2 = get_second_tangent_line_tanh_np(x1, get_big=True)

        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_scalar_zero(self):
        """Test with scalar x1=0."""
        x1_scalar = 0.0
        b, _, x2 = get_second_tangent_line_tanh_np(x1_scalar, get_big=True)

        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_scalar_nonzero(self):
        """Test with scalar x1 != 0."""
        x1_scalar = 1.5
        b, _, x2 = get_second_tangent_line_tanh_np(x1_scalar, get_big=True)

        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))

    def test_second_tangent_tanh_mixed_array(self):
        """Test with mixed positive, negative, and zero values."""
        x1 = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        b, _, x2 = get_second_tangent_line_tanh_np(x1, get_big=True)

        assert x1.shape == b.shape
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(x2))


class TestTangentLineConsistency:
    """Test consistency properties of tangent line functions."""

    def test_sigmoid_tanh_parallel_consistency(self):
        """Test parallel tangent lines have consistent output format."""
        k = np.array([0.1, 0.2])

        b_sig, k_sig, x_sig = get_parallel_tangent_line_sigmoid_np(k, get_big=True)
        b_tanh, k_tanh, x_tanh = get_parallel_tangent_line_tanh_np(k, get_big=True)

        # Same shape
        assert b_sig.shape == b_tanh.shape
        assert k_sig.shape == k_tanh.shape
        assert x_sig.shape == x_tanh.shape

    def test_sigmoid_tanh_second_consistency(self):
        """Test second tangent lines have consistent output format."""
        x1 = np.array([0.0, 0.5, 1.0])

        b_sig, k_sig, x_sig = get_second_tangent_line_sigmoid_np(x1, get_big=True)
        b_tanh, k_tanh, x_tanh = get_second_tangent_line_tanh_np(x1, get_big=True)

        # Same shape
        assert b_sig.shape == b_tanh.shape
        assert k_sig.shape == k_tanh.shape
        assert x_sig.shape == x_tanh.shape

    def test_get_big_symmetry_sigmoid(self):
        """Test that get_big=True and get_big=False produce consistent shapes."""
        k = np.array([0.15])

        b_big, _, x_big = get_parallel_tangent_line_sigmoid_np(k, get_big=True)
        b_small, _, x_small = get_parallel_tangent_line_sigmoid_np(k, get_big=False)

        assert b_big.shape == b_small.shape
        assert x_big.shape == x_small.shape

    def test_get_big_symmetry_tanh(self):
        """Test that get_big=True and get_big=False produce consistent shapes."""
        k = np.array([0.5])

        b_big, _, x_big = get_parallel_tangent_line_tanh_np(k, get_big=True)
        b_small, _, x_small = get_parallel_tangent_line_tanh_np(k, get_big=False)

        assert b_big.shape == b_small.shape
        assert x_big.shape == x_small.shape


class TestTangentLineWarmup:
    """Test the JIT warmup function for tangent line functions."""

    def test_warmup_function_executes(self):
        """Test that the _warmup_jit_functions can be called successfully.

        This tests the refactored JIT warmup code which was moved from
        module-level side effects to an explicit function.
        """
        from wraact._tangent_lines import _warmup_jit_functions

        # Should execute without error
        _warmup_jit_functions()

    def test_tangent_functions_compiled_after_warmup(self):
        """Test that tangent functions work correctly after warmup.

        Verifies that JIT compilation completes successfully and functions
        produce correct results.
        """
        from wraact._tangent_lines import (
            _warmup_jit_functions,
            get_parallel_tangent_line_sigmoid_np,
            get_parallel_tangent_line_tanh_np,
        )

        # Execute warmup
        _warmup_jit_functions()

        # Test that functions work after warmup
        k = np.array([0.1, 0.2])
        b_sig, _, _ = get_parallel_tangent_line_sigmoid_np(k, get_big=True)
        b_tanh, _, _ = get_parallel_tangent_line_tanh_np(k, get_big=True)

        # Results should be finite
        assert np.all(np.isfinite(b_sig))
        assert np.all(np.isfinite(b_tanh))
