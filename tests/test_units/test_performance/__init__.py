"""Performance benchmarking tests for wraact (marked @pytest.mark.slow).

Tests cover:
- Runtime scaling with input dimension (2D, 3D, 4D only)
- Single-neuron vs multi-neuron method comparison
- Constraint count growth analysis
- Memory usage profiling
- Numba JIT compilation effectiveness
- Hull precision measurement (volume estimation)
"""
