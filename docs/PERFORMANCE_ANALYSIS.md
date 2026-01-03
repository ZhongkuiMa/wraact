# Performance Analysis Guide

## Overview

This guide explains how to interpret performance metrics, detect regressions, and make optimization decisions for the wraact library.

---

## 1. Key Performance Metrics

### 1.1 Runtime (Execution Time)

**What it measures**: How long cal_hull() takes to compute constraints

**Typical values** ([-1, 1]^d polytopes):
- 2D: 0.3-0.5 ms
- 3D: 0.3-0.5 ms
- 4D: 0.4-0.6 ms

**Why it matters**:
- Direct impact on user experience
- Larger regressions (>50%) indicate algorithmic inefficiency

**Example interpretation**:
```
Baseline: ReLU 2D = 0.45 ms
Current:  ReLU 2D = 0.68 ms
Change:   +51%  → 🔴 RED FLAG

Action: Investigate new code changes since last baseline
```

---

### 1.2 Constraint Count

**What it measures**: Number of linear constraints in output polytope

**Typical values** ([-1, 1]^d polytopes):
- ReLU: 2·d constraints (2D→4, 3D→6, 4D→8)
- Sigmoid/Tanh: 6·d constraints (2D→12, 3D→18, 4D→24)

**Why it matters**:
- Constraint count drives runtime (O(n log n) where n = constraint count)
- Affects downstream solver performance
- Indicates hull quality/tightness

**Example interpretation**:
```
Baseline: ReLU 2D = 4 constraints
Current:  ReLU 2D = 5 constraints
Change:   +25%  → 🔴 RED FLAG

Likely cause: Algorithm change added extra constraint
Investigation: Check git diff for changes to constraint generation logic
```

---

### 1.3 Precision (Coverage %)

**What it measures**: % of sample points inside computed hull (soundness)

**Typical values**:
- ReLU: 95-100% (tight, because it's piecewise linear)
- Sigmoid/Tanh: 85-98% (looser, because curves are smooth)

**Why it matters**:
- Must stay >85% to guarantee soundness
- Lower % means hull doesn't fully contain function values

**Example interpretation**:
```
Baseline: Sigmoid 2D = 92%
Current:  Sigmoid 2D = 78%
Change:   -14 points → 🔴 RED FLAG

Critical: Hull is no longer sound! Must fix immediately.
```

---

## 2. Interpreting Baseline Data

### 2.1 Linear Growth Patterns

**ReLU**:
```
Constraints = 2·d
Example: 2D→4, 3D→6, 4D→8

This is EXPECTED and GOOD
- Indicates tight, minimal hull
- Each dimension adds exactly 2 constraints
```

**Sigmoid**:
```
Constraints = 6·d
Example: 2D→12, 3D→18, 4D→24

This is EXPECTED
- Requires more constraints than ReLU (smooth vs piecewise)
- Still linear growth (good scalability)
```

### 2.2 Exponential Growth (BAD)

**Example**: If ReLU suddenly produces 2^d constraints:
```
2D→4 (baseline: 4)  ✓ Match
3D→8 (baseline: 6)  ✗ +33% (exponential growth detected!)
4D→16 (baseline: 8) ✗ +100% (clearly exponential)

Action: This indicates a serious algorithmic issue.
Investigate CDD vertex enumeration or constraint generation.
```

---

## 3. Decision Matrix

### 3.1 When to Investigate (and How Long to Spend)

| Metric Change | Category | Action | Time Budget |
|---------------|----------|--------|-------------|
| >50% runtime | 🔴 Red | STOP commits, investigate immediately | 2-4 hours |
| 20-50% runtime | 🟡 Yellow | Investigate in next 1-2 commits | 30 mins |
| <20% runtime | 🟢 Green | Normal variation, monitor trend | 0 |
| >20% constraints | 🔴 Red | STOP, check algorithm changes | 2-4 hours |
| 10-20% constraints | 🟡 Yellow | Review code changes | 30 mins |
| <10% constraints | 🟢 Green | Likely experimental variance | 0 |
| <85% precision | 🔴 Red | STOP, hull is unsound | 1-2 hours |
| 85-90% precision | 🟡 Yellow | Monitor, may need tuning | 30 mins |
| >90% precision | 🟢 Green | Good, acceptable | 0 |

### 3.2 Investigation Checklist

**Step 1: Identify What Changed**
```bash
git log --oneline | head -5
# Look at commits since last baseline

git diff HEAD~5...HEAD tools/acthull/_*.py
# Check core algorithm changes
```

**Step 2: Classify the Change**
- [ ] **Bug fix**: May justify small regression
- [ ] **Feature addition**: Expect some overhead
- [ ] **Optimization**: Should improve metrics
- [ ] **Refactoring**: Should have no impact

**Step 3: Benchmark Impact**
```python
# Create minimal reproducible example
import time
import numpy as np

for repeat in range(5):
    lb, ub = np.full(2, -1.0), np.full(2, 1.0)
    t0 = time.perf_counter()
    result = hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
    t1 = time.perf_counter()
    print(f"Run {repeat}: {(t1-t0)*1000:.2f}ms")
```

**Step 4: Make Decision**
- If regression is acceptable: Update PERFORMANCE_BASELINE.md
- If regression is unacceptable: Revert or optimize

---

## 4. Common Performance Scenarios

### Scenario A: Small Runtime Increase (+5-15%)

**Likely causes**:
- Added debug assertions
- New optional features
- Slightly different code path

**Decision**: Usually acceptable
```
Example: 0.45ms → 0.49ms (+8%)
Status: 🟢 GREEN - within normal variation
Action: Monitor, no action required
```

### Scenario B: Large Runtime Increase (+50%+)

**Likely causes**:
- Inefficient loop added
- New O(n²) algorithm where O(n) was before
- Unnecessary vertex enumeration

**Decision**: Must investigate
```
Example: 0.45ms → 0.72ms (+60%)
Status: 🔴 RED - significant regression
Action: STOP development, profile code

git bisect to find commit
Run profiler to find hot spot
Optimize or revert
```

### Scenario C: Constraint Count Explosion

**Example**:
```
Before: ReLU 3D → 6 constraints
After:  ReLU 3D → 18 constraints (+200%)

Likely cause: Algorithm now generating redundant constraints
Impact: 3× slower (O(n log n) where n increased 3×)

Investigation:
1. Check if hull is still sound (should be 100%)
2. If yes: Constraints are redundant, need to filter
3. If no: Algorithm bug, must fix
```

### Scenario D: Precision Loss

**Example**:
```
Before: ReLU 2D → 97% coverage
After:  ReLU 2D → 79% coverage

Status: 🔴 CRITICAL - hull is unsound!

Action: IMMEDIATE REVERT
This indicates the computed polytope doesn't contain all function values
This is a correctness bug, not just performance issue
```

---

## 5. Optimization Strategies

### 5.1 When Runtime is Slow

1. **Profile the code**:
   ```python
   import cProfile
   cProfile.run('hull.cal_hull(...)')
   ```

2. **Look for common bottlenecks**:
   - CDD vertex computation: O(n·d) where n=constraints
   - Tangent line calculation: Should be O(d²)
   - Constraint filtering: Should be O(n log n)

3. **Optimization techniques**:
   - Cache tangent line calculations
   - Use NumPy vectorization instead of loops
   - Early termination if hull is tight enough
   - Consider approximate algorithms for large polytopes

### 5.2 When Constraint Count Grows

1. **Identify source**:
   ```python
   # Before filtering
   print(f"Generated: {raw_constraints.shape[0]} constraints")

   # After filtering
   print(f"Final: {filtered_constraints.shape[0]} constraints")
   ```

2. **Check for redundancy**:
   - Plot constraints (2D case)
   - Check if any constraint is implied by others
   - Verify all constraints are "tight" (saturated at some vertex)

3. **Optimization**:
   - Implement constraint removal (linear programming)
   - Use CDD's H-representation minimization
   - Consider approximate constraints

---

## 6. Regression Reporting

### 6.1 Creating a Regression Report

```bash
# Run performance tests
pytest tests/test_performance/ -v > perf_run.log

# Manually create report
cat > REGRESSION_REPORT.md << EOF
# Performance Regression Report

Date: $(date)
Git Commit: $(git log -1 --oneline)

## Changes Detected

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| ReLU 2D Runtime | 0.45ms | 0.68ms | +51% 🔴 |
| ReLU 2D Constraints | 4 | 4 | - |

## Root Cause

[Your investigation here]

## Resolution

[What you did to fix it]
EOF
```

### 6.2 Updating Baseline

```bash
# After optimization confirmed
vim PERFORMANCE_BASELINE.md
# Update metrics section with new values

git add PERFORMANCE_BASELINE.md
git commit -m "Update performance baseline after optimization"
```

---

## 7. Continuous Monitoring

### 7.1 GitHub Actions Workflow

The `.github/workflows/performance-regression.yml` automatically:
- Runs performance tests on every push/PR
- Compares against baseline
- Posts results as PR comment
- Fails build if red flag detected

### 7.2 Local Testing

```bash
# Run before committing
python tools/detect_perf_regression.py

# Check specific category
pytest tests/test_performance/test_runtime.py -v

# Get detailed timing
pytest tests/test_performance/test_runtime.py::TestRuntimeScaling -v -s
```

---

## 8. Interpretation Examples

### Example 1: Expected Speedup After Optimization

```
Change: Switched to cached tangent lines
Before: ReLU 2D = 0.45ms
After:  ReLU 2D = 0.38ms
Change: -16% 🟢 GOOD

Interpretation: Optimization worked as intended
Action: Update baseline, document optimization
```

### Example 2: Unexpected Regression

```
Change: Refactored constraint filtering
Before: ReLU 3D = 0.35ms
After:  ReLU 3D = 0.52ms
Change: +49% 🟡 YELLOW

Interpretation: Refactoring introduced inefficiency
Investigation: Profiling shows new O(n²) loop

Decision: Revert refactor and re-approach differently
```

### Example 3: Feature Addition Trade-off

```
Change: Added advanced polytope validation
Before: ReLU 2D = 0.45ms, 4 constraints
After:  ReLU 2D = 0.48ms, 4 constraints
Change: +7% runtime, 0% constraints 🟢

Interpretation: Validation adds small overhead but no constraint growth
Cost-benefit: +7% is worth the improved robustness
Action: Accept and monitor
```

---

## 9. Quick Reference

### Acceptable Performance Variance
- **Normal**: ±10% (measurement noise)
- **Acceptable**: ±20% (minor code changes)
- **Investigate**: ±30-50% (potential issue)
- **Critical**: >50% or <85% precision (must fix)

### Constraint Count Guidelines
- **ReLU**: 2·d (exactly linear)
- **Sigmoid**: 6·d (exactly linear)
- **Any deviation**: Investigate
- **Exponential growth**: Critical bug

### Scaling Expectations
- Time complexity: O(n log n) where n = constraints
- Constraint growth: Linear (good)
- Memory: O(n·d)

---

## 10. Related Documentation

- `PERFORMANCE_BASELINE.md` - Current baseline values
- `tools/detect_perf_regression.py` - Automated regression detection
- `.github/workflows/performance-regression.yml` - CI/CD integration
- `tests/test_performance/` - Performance test suite

---

## Summary

**Key Takeaways:**
1. Compare against baseline, not absolute values
2. Use 50% threshold for runtime, 20% for constraints
3. Precision <85% is critical (must fix)
4. Profile before optimizing
5. Document why regressions are acceptable (if they are)
6. Update baseline after confirmed improvements
