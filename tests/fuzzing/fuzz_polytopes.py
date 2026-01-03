#!/usr/bin/env python
"""Random fuzzing to discover edge cases in polytope hull calculation."""

__docformat__ = "restructuredtext"

import json
from pathlib import Path

import numpy as np

from wraact._exceptions import DegeneratedError
from wraact.acthull import (
    ELUHull,
    LeakyReLUHull,
    MaxPoolHullDLP,
    ReLUHull,
    SigmoidHull,
    TanhHull,
)

# Configuration
NUM_ITERATIONS = 10000  # Can increase for overnight runs
OUTPUT_DIR = Path(__file__).parent / "edge_cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Random number generator
rng = np.random.default_rng()


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.ndarray):
        return [float(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    return str(obj)


def generate_random_polytope(dim: int, strategy: str) -> tuple:
    """Generate random polytope using different strategies.

    Strategies:
    - "normal": Standard random bounds
    - "degenerate": Likely to create degenerate polytopes (collapsed dims)
    - "extreme": Extreme value ranges
    - "narrow": Very narrow bounds
    - "conflicting": Intentionally create lb > ub scenarios
    """
    if strategy == "normal":
        lb = rng.uniform(-10, 10, dim)
        ub = lb + rng.uniform(0.1, 10, dim)

    elif strategy == "degenerate":
        lb = rng.uniform(-5, 5, dim)
        ub = lb.copy()
        # Randomly collapse some dimensions
        collapse_mask = rng.random(dim) > 0.7
        ub[~collapse_mask] += rng.uniform(0.1, 5, (~collapse_mask).sum())

    elif strategy == "extreme":
        # Mix very large and very small values
        scales = rng.choice([1e-10, 1e-5, 1, 1e5, 1e10], dim)
        lb = rng.uniform(-1, 1, dim) * scales
        ub = lb + rng.uniform(0.01, 1, dim) * scales

    elif strategy == "narrow":
        lb = rng.uniform(-10, 10, dim)
        ub = lb + rng.uniform(1e-6, 1e-3, dim)

    elif strategy == "conflicting":
        # Intentionally create lb > ub for some dimensions
        lb = rng.uniform(-10, 10, dim)
        ub = rng.uniform(-10, 10, dim)
        # Don't enforce lb < ub!

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return lb, ub


def fuzz_activations() -> list:
    """Fuzz all activation hull types."""
    hulls = [
        ReLUHull,
        SigmoidHull,
        TanhHull,
        ELUHull,
        LeakyReLUHull,
        MaxPoolHullDLP,
    ]

    strategies = ["normal", "degenerate", "extreme", "narrow", "conflicting"]
    dims = [2, 3, 4, 5, 6]

    edge_cases = []
    exception_counts: dict[str, int] = {}

    for i in range(NUM_ITERATIONS):
        dim = rng.choice(dims)
        strategy = rng.choice(strategies)
        hull_class = rng.choice(hulls)

        lb, ub = generate_random_polytope(dim, strategy)

        hull = hull_class()

        try:
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
            # Success - not interesting for coverage

        except TypeError as e:
            # Catch TypeError from cdd.Error not inheriting from BaseException or from hull itself
            exc_type = "TypeError"
            exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1
            edge_case = {
                "iteration": i,
                "hull": hull_class.__name__,
                "dim": dim,
                "strategy": strategy,
                "lb": lb.tolist(),
                "ub": ub.tolist(),
                "exception": exc_type,
                "message": str(e),
            }
            edge_case = convert_to_json_serializable(edge_case)
            edge_cases.append(edge_case)
            filename = OUTPUT_DIR / f"edge_case_{len(edge_cases):04d}_{exc_type}.json"
            with filename.open("w") as f:
                json.dump(edge_case, f, indent=2)

        except RuntimeError as e:
            # This is what we want! Exception logging path
            exc_type = "RuntimeError"
            exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1

            edge_case = {
                "iteration": i,
                "hull": hull_class.__name__,
                "dim": dim,
                "strategy": strategy,
                "lb": lb.tolist(),
                "ub": ub.tolist(),
                "exception": exc_type,
                "message": str(e),
            }
            edge_case = convert_to_json_serializable(edge_case)
            edge_cases.append(edge_case)

            # Save immediately
            filename = OUTPUT_DIR / f"edge_case_{len(edge_cases):04d}_{exc_type}.json"
            with filename.open("w") as f:
                json.dump(edge_case, f, indent=2)

        except DegeneratedError as e:
            exc_type = "DegeneratedError"
            exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1

            edge_case = {
                "iteration": i,
                "hull": hull_class.__name__,
                "dim": dim,
                "strategy": strategy,
                "lb": lb.tolist(),
                "ub": ub.tolist(),
                "exception": exc_type,
                "message": str(e),
            }
            edge_case = convert_to_json_serializable(edge_case)
            edge_cases.append(edge_case)

            filename = OUTPUT_DIR / f"edge_case_{len(edge_cases):04d}_{exc_type}.json"
            with filename.open("w") as f:
                json.dump(edge_case, f, indent=2)

        except ValueError:
            exc_type = "ValueError"
            exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1
            # Usually input validation - less interesting

        if i % 1000 == 0:
            print(f"Iteration {i}: Found {len(edge_cases)} edge cases")
            print(f"Exception counts: {exception_counts}")

    # Summary
    print("\n=== Fuzzing Complete ===")
    print(f"Total iterations: {NUM_ITERATIONS}")
    print(f"Edge cases found: {len(edge_cases)}")
    print(f"Exception breakdown: {exception_counts}")
    print(f"Saved to: {OUTPUT_DIR}")

    return edge_cases


if __name__ == "__main__":
    edge_cases = fuzz_activations()
