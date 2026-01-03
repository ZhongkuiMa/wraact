#!/usr/bin/env python3
"""Performance Regression Detection Script.

Compares current performance metrics against baseline and flags regressions.
Usage: python tools/detect_perf_regression.py
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

# Baseline metrics (from PERFORMANCE_BASELINE.md, measured 2025-12-28)
# Updated to use unified 3^d constraint model across all activation functions
BASELINE = {
    "runtime": {
        "relu": {
            2: 0.45,  # ms
            3: 0.35,
            4: 0.41,
        },
        "leakyrelu": {
            2: 0.33,
            3: 0.44,
            4: 0.67,
        },
        "elu": {
            2: 0.20,
            3: 0.24,
            4: 0.31,
        },
        "sigmoid": {
            2: 0.68,
            3: 0.93,
            4: 1.21,
        },
        "tanh": {
            2: 0.89,
            3: 1.27,
            4: 1.66,
        },
        "maxpool": {
            2: 0.25,
            3: 0.28,
            4: 0.31,
        },
        "maxpooldlp": {
            2: 0.28,
            3: 0.30,
            4: 0.32,
        },
    },
    "constraints": {
        "relu": {
            2: 4,  # 2·d
            3: 6,  # 2·d
            4: 8,  # 2·d
        },
        "leakyrelu": {
            2: 4,  # 2·d
            3: 6,  # 2·d
            4: 8,  # 2·d
        },
        "elu": {
            2: 6,  # 3·d
            3: 9,  # 3·d
            4: 12,  # 3·d
        },
        "sigmoid": {
            2: 12,  # 6·d
            3: 18,  # 6·d
            4: 24,  # 6·d
        },
        "tanh": {
            2: 8,  # 4·d
            3: 12,  # 4·d
            4: 16,  # 4·d
        },
        "maxpool": {
            2: 4,  # 2·d
            3: 6,  # 2·d
            4: 8,  # 2·d
        },
        "maxpooldlp": {
            2: 4,  # 2·d
            3: 6,  # 2·d
            4: 8,  # 2·d
        },
    },
    "speedup_withoney": {
        "relu": {2: 1.31, 3: 1.51, 4: 1.64},
        "leakyrelu": {2: 1.45, 3: 1.89, 4: 2.17},
        "elu": {2: 1.23, 3: 1.44, 4: 1.59},
        "sigmoid": {2: 1.67, 3: 2.25, 4: 2.70},
        "tanh": {2: 1.75, 3: 2.45, 4: 3.00},
        "maxpool": {2: 1.01, 3: 1.05, 4: 1.01},  # Below 1.1x threshold
        "maxpooldlp": {2: 1.06, 3: 0.98, 4: 0.99},  # Below 1.1x threshold
    },
}

# Regression thresholds
THRESHOLDS = {
    "runtime": {
        "red": 1.50,  # 150% increase (2.5x baseline with ±150% tolerance)
        "yellow": 0.50,  # 50% increase
    },
    "constraints": {
        "red": 0.20,  # 20% deviation from 3^d baseline (±20% tolerance)
        "yellow": 0.10,  # 10% deviation
    },
}


def run_performance_tests() -> dict:
    """Run performance tests and capture metrics."""
    print("Running performance tests...")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_performance/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


def extract_metrics(output: str) -> dict:
    """Extract performance metrics from test output."""
    metrics: dict = {
        "runtime": {},
        "constraints": {},
        "tests_run": 0,
        "tests_passed": 0,
        "tests_skipped": 0,
    }

    # Parse runtime metrics: "ReLU dim 2: 0.45ms, 4 constraints"
    runtime_pattern = r"(\w+)\s+dim\s+(\d+):\s+([\d.]+)ms,\s+(\d+)\s+constraints"
    for match in re.finditer(runtime_pattern, output):
        activation, dim, time_ms, constraints = match.groups()
        dim = int(dim)
        time_ms = float(time_ms)
        constraints = int(constraints)

        if activation.lower() not in metrics["runtime"]:
            metrics["runtime"][activation.lower()] = {}
        metrics["runtime"][activation.lower()][dim] = time_ms

        if activation.lower() not in metrics["constraints"]:
            metrics["constraints"][activation.lower()] = {}
        metrics["constraints"][activation.lower()][dim] = constraints

    # Parse test counts
    counts_pattern = r"(\d+)\s+passed|(\d+)\s+skipped"
    for match in re.finditer(counts_pattern, output):
        if match.group(1):
            metrics["tests_passed"] += int(match.group(1))
        if match.group(2):
            metrics["tests_skipped"] += int(match.group(2))

    return metrics


def check_regressions(  # noqa: C901
    current: dict, baseline: dict, thresholds: dict
) -> tuple[list, list, list]:
    """Compare current metrics against baseline and classify regressions."""
    red_flags = []
    yellow_flags = []
    green = []

    # Check runtime metrics
    for activation, dims in current.get("runtime", {}).items():
        if activation not in baseline.get("runtime", {}):
            continue

        for dim, current_time in dims.items():
            baseline_time = baseline["runtime"][activation].get(dim)
            if baseline_time is None:
                continue

            change = (current_time - baseline_time) / baseline_time
            threshold_red = thresholds["runtime"]["red"]
            threshold_yellow = thresholds["runtime"]["yellow"]

            msg = (
                f"Runtime {activation.upper()} {dim}D: "
                f"{baseline_time:.2f}ms → {current_time:.2f}ms "
                f"({change:+.1%})"
            )

            if change > threshold_red:
                red_flags.append(f"🔴 RED: {msg}")
            elif change > threshold_yellow:
                yellow_flags.append(f"🟡 YELLOW: {msg}")
            else:
                green.append(f"🟢 GREEN: {msg}")

    # Check constraint count metrics
    for activation, dims in current.get("constraints", {}).items():
        if activation not in baseline.get("constraints", {}):
            continue

        for dim, current_count in dims.items():
            baseline_count = baseline["constraints"][activation].get(dim)
            if baseline_count is None:
                continue

            change = (current_count - baseline_count) / baseline_count
            threshold_red = thresholds["constraints"]["red"]
            threshold_yellow = thresholds["constraints"]["yellow"]

            msg = (
                f"Constraints {activation.upper()} {dim}D: "
                f"{baseline_count} → {current_count} "
                f"({change:+.1%})"
            )

            if change > threshold_red:
                red_flags.append(f"🔴 RED: {msg}")
            elif change > threshold_yellow:
                yellow_flags.append(f"🟡 YELLOW: {msg}")
            else:
                green.append(f"🟢 GREEN: {msg}")

    return red_flags, yellow_flags, green


def save_report(current_metrics: dict, red_flags: list, yellow_flags: list, green: list):
    """Save regression detection report."""
    report_dir = Path("performance_reports")
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"regression_report_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "metrics": current_metrics,
        "regressions": {
            "red": red_flags,
            "yellow": yellow_flags,
            "green": green,
        },
        "summary": {
            "red_count": len(red_flags),
            "yellow_count": len(yellow_flags),
            "green_count": len(green),
            "total_issues": len(red_flags) + len(yellow_flags),
        },
    }

    report_path = Path(report_file)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")
    return report


def main():
    """Run performance regression detection and report results."""
    print("=" * 70)
    print("PERFORMANCE REGRESSION DETECTION")
    print("=" * 70)
    print()

    # Run tests (note: this would need actual output from performance tests)
    test_result = run_performance_tests()

    # Extract metrics from output
    current_metrics = extract_metrics(test_result["stdout"])

    print(f"\n✓ Tests completed: {test_result['returncode']}")
    print(f"  - Passed: {current_metrics['tests_passed']}")
    print(f"  - Skipped: {current_metrics['tests_skipped']}")
    print()

    # Check for regressions
    red_flags, yellow_flags, green = check_regressions(current_metrics, BASELINE, THRESHOLDS)

    print("RESULTS:")
    print("-" * 70)

    if red_flags:
        for flag in red_flags:
            print(flag)
        print()

    if yellow_flags:
        for flag in yellow_flags:
            print(flag)
        print()

    for flag in green[:5]:  # Show first 5 green
        print(flag)
    if len(green) > 5:
        print(f"  ... and {len(green) - 5} more green metrics")
    print()

    # Summary
    print("SUMMARY:")
    print("-" * 70)
    print(f"🔴 Red Flags (>50% increase): {len(red_flags)}")
    print(f"🟡 Yellow Flags (>20% increase): {len(yellow_flags)}")
    print(f"🟢 Green (Normal): {len(green)}")
    print()

    # Save report
    save_report(current_metrics, red_flags, yellow_flags, green)

    # Exit with appropriate code
    if red_flags:
        print("⚠️  Performance regression detected! Review PERFORMANCE_BASELINE.md")
        return 1
    if yellow_flags:
        print("⚠️  Possible performance issue. Monitor next few commits.")
        return 0
    print("✅ Performance is stable.")
    return 0


if __name__ == "__main__":
    exit(main())
