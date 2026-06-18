"""Robustness regression tests from fuzzing-discovered edge cases.

Each JSON under ``tests/fuzzing/edge_cases/`` records a degenerate
``(hull, lb, ub)`` input that the fuzzer found. The library must DEGRADE
CLEANLY on every such input: either it returns constraints, or it raises one
of the *sanctioned* exceptions (input-validation / degeneracy / non-convergence).
A raw interpreter crash (``KeyError``, ``ZeroDivisionError``, ``TypeError``,
``AttributeError``, ``IndexError``, ...) is a real bug and fails the test.

The stale ``exception``/``message`` fields in the JSON are intentionally
ignored: they recorded a since-fixed bug (the ``except cdd.Error``
non-``BaseException`` defect that surfaced as
``TypeError: catching classes that do not inherit from BaseException``).
Only the input vectors are still meaningful, as a degenerate-input corpus.

Soundness of the *success* cases is out of scope here -- this file guards
crash-freedom, not hull tightness.
"""

__docformat__ = "restructuredtext"

import json
from pathlib import Path

import numpy as np
import pytest

from wraact._exceptions import DegeneratedError, NotConvergedError
from wraact.acthull import (
    ELUHull,
    LeakyReLUHull,
    MaxPoolHullDLP,
    ReLUHull,
    SigmoidHull,
    TanhHull,
)

# Exceptions the library is ALLOWED to raise on a degenerate input. Anything
# else escaping cal_hull is a raw crash and a real bug.
SANCTIONED_EXCEPTIONS = (ValueError, RuntimeError, DegeneratedError, NotConvergedError)

HULL_MAP: dict[
    str,
    type[ReLUHull | SigmoidHull | TanhHull | ELUHull | LeakyReLUHull | MaxPoolHullDLP],
] = {
    "ReLUHull": ReLUHull,
    "SigmoidHull": SigmoidHull,
    "TanhHull": TanhHull,
    "ELUHull": ELUHull,
    "LeakyReLUHull": LeakyReLUHull,
    "MaxPoolHullDLP": MaxPoolHullDLP,
}

# Load all edge cases from fuzzing
EDGE_CASES_DIR = Path(__file__).parent.parent.parent / "fuzzing" / "edge_cases"


def load_edge_cases():
    """Load all edge case JSON files."""
    edge_case_files = list(EDGE_CASES_DIR.glob("edge_case_*.json"))
    edge_cases = {}

    for filepath in sorted(edge_case_files):
        try:
            with filepath.open() as f:
                edge_case = json.load(f)
                edge_cases[filepath.name] = edge_case
        except (OSError, json.JSONDecodeError):
            # Skip malformed files
            pass

    return edge_cases


# Load edge cases at module level
EDGE_CASES = load_edge_cases()


@pytest.mark.parametrize(
    ("name", "edge_case"), list(EDGE_CASES.items()), ids=list(EDGE_CASES.keys())
)
def test_fuzzing_edge_case_regression(name, edge_case):
    """A degenerate fuzzing input degrades cleanly (no raw interpreter crash).

    cal_hull must either return constraints or raise a sanctioned exception
    (``ValueError`` / ``RuntimeError`` / ``DegeneratedError`` /
    ``NotConvergedError``). Any other escaping exception is a bug.
    """
    hull_name = edge_case.get("hull")
    if hull_name not in HULL_MAP:
        pytest.skip(f"Unknown hull type: {hull_name}")

    hull = HULL_MAP[hull_name]()
    lb = np.array(edge_case["lb"], dtype=np.float64)
    ub = np.array(edge_case["ub"], dtype=np.float64)

    try:
        hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
    except SANCTIONED_EXCEPTIONS:
        # Clean degradation on a degenerate input -- the contract holds.
        pass
    except Exception as exc:  # noqa: BLE001 -- the whole point is to catch raw crashes
        pytest.fail(
            f"{hull_name} cal_hull raised a non-sanctioned {type(exc).__name__} "
            f"on a degenerate input: {exc!r}. Sanctioned types: "
            f"{tuple(t.__name__ for t in SANCTIONED_EXCEPTIONS)}."
        )
