"""Soundness verification and error handling tests for wraact.

Tests cover:
- Exception raising for invalid inputs (DegeneratedError, NotConvergedError)
- Edge cases (zero ranges, extreme bounds, degenerate polytopes)
- Over-approximation guarantees (Monte Carlo soundness verification)
- Constraint satisfaction validation
"""
