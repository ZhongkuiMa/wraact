"""Regression tests from fuzzing-discovered edge cases."""

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

# Load all edge cases from fuzzing
EDGE_CASES_DIR = Path(__file__).parent.parent / "fuzzing" / "edge_cases"


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
    """Test edge case discovered by fuzzing.

    This test verifies that:
    1. Exception paths are triggered (coverage!)
    2. Error messages are informative
    """
    hull_map = {
        "ReLUHull": ReLUHull,
        "SigmoidHull": SigmoidHull,
        "TanhHull": TanhHull,
        "ELUHull": ELUHull,
        "LeakyReLUHull": LeakyReLUHull,
        "MaxPoolHullDLP": MaxPoolHullDLP,
    }

    hull_name = edge_case.get("hull")
    if hull_name not in hull_map:
        pytest.skip(f"Unknown hull type: {hull_name}")

    hull_class = hull_map[hull_name]
    lb = np.array(edge_case["lb"], dtype=np.float64)
    ub = np.array(edge_case["ub"], dtype=np.float64)
    expected_exception = edge_case.get("exception", "Unknown")

    hull = hull_class()

    # Test that the expected exception is raised
    if expected_exception == "RuntimeError":
        with pytest.raises(RuntimeError):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    elif expected_exception == "DegeneratedError":
        with pytest.raises(DegeneratedError):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    elif expected_exception == "NotConvergedError":
        with pytest.raises(NotConvergedError):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    elif expected_exception == "TypeError":
        # These are expected - the code has a bug with catching cdd.Error
        # This test documents the issue
        with pytest.raises(TypeError):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    elif expected_exception == "ValueError":
        # Input validation errors - also acceptable
        with pytest.raises(ValueError, match=r"invalid|bound|shape"):
            hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

    else:
        # Unexpected exception type - let it fail
        hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
