---
type: DESCRIPTION
note: "Descriptive. Mirrors current code; update to follow code changes. < functional code."
---
> **This file IS**: architecture overview of wraact. **It is NOT**: implementation code or detailed conventions.

# WRAACT Architecture

Convex hull computation for neural network activation functions using pycddlib (double description method) with NumPy/Numba.

## Package Tree

```
src/wraact/
├── __init__.py          Re-exports all hull classes (modify when: adding new hull)
├── _constants.py        Numerical tolerances and defaults (modify when: tuning precision)
├── _exceptions.py       DegeneratedError, NotConvergedError (modify when: new failure modes)
├── _functions.py        Activation functions + derivatives in NumPy (modify when: adding activation)
├── _tangent_lines.py    Tangent line solvers for S-shaped functions (modify when: convergence issues)
├── acthull/             Multi-output hull classes (modify when: adding/changing hull logic)
│   ├── _act.py          ActHull base class — core API: cal_hull(), cal_constrs()
│   ├── _sshape.py       SShapeHull — base for sigmoid/tanh (DLP bounding)
│   ├── _relulike.py     ReLULikeHull — base for piecewise-linear activations
│   ├── _relu.py         ReLUHull
│   ├── _sigmoid.py      SigmoidHull
│   ├── _tanh.py         TanhHull
│   ├── _elu.py          ELUHull
│   ├── _leakyrelu.py    LeakyReLUHull
│   ├── _maxpool.py      MaxPoolHull, MaxPoolHullDLP
│   └── _utils.py        cal_mn_constrs_with_one_y_dlp (DLP constraint builder)
└── oney/                Single-output optimized variants (modify when: adding *WithOneY class)
    ├── _act.py          ActHullWithOneY base class
    ├── _relu.py         ReLUHullWithOneY
    ├── _sigmoid.py      SigmoidHullWithOneY
    ├── _tanh.py         TanhHullWithOneY
    ├── _elu.py          ELUHullWithOneY
    ├── _leakyrelu.py    LeakyReLUHullWithOneY
    ├── _maxpool.py      MaxPoolHullWithOneY, MaxPoolHullDLPWithOneY
    ├── _relulike.py     ReLULikeHullWithOneY
    └── _sshape.py       SShapeHullWithOneY
```

## Modification Map

| Intent | Primary Modify | Follow-ups | Avoid | Constraints | Failure Signal |
|--------|---------------|------------|-------|-------------|----------------|
| Add activation function | `_functions.py` | `acthull/_new.py`, `oney/_new.py`, all 3 `__init__.py` | Existing hull files | Must provide derivative (enforced) | `ImportError` on missing export |
| Add piecewise-linear hull | `acthull/_new.py` (inherit `ReLULikeHull`) | `oney/_new.py`, `__init__.py` files | `_sshape.py`, `_tangent_lines.py` | Override `cal_constrs` (enforced) | `TypeError` abstract method |
| Add S-shaped hull | `acthull/_new.py` (inherit `SShapeHull`) | `_tangent_lines.py`, `oney/_new.py`, `__init__.py` files | `_relulike.py` | Must add tangent solver (observed) | `NotConvergedError` |
| Tune numerical precision | `_constants.py` | None | Hull class logic | Soundness tests must pass (enforced) | `pytest tests/test_soundness` failures |
| Fix convergence issue | `_tangent_lines.py` | None | `_functions.py` | numba `@njit` compatible (enforced) | `NotConvergedError` at runtime |

## Dependency Rules

| Rule | Source | Failure |
|------|--------|---------|
| Absolute imports only | (enforced) ruff TID `ban-relative-imports = "all"` | `ruff check` error |
| No PyTorch — NumPy + pycddlib + numba only | (observed) | Import error in CI (no torch in deps) |
| `acthull/` and `oney/` import from top-level `_*.py` modules | (observed) | Circular import |
| `oney/` mirrors `acthull/` 1:1 (same activation, `WithOneY` suffix) | (observed) | Missing export |

## Common Mistakes

| Mistake | Detection Signal | Fix |
|---------|-----------------|-----|
| Adding hull without `oney/` variant | Missing `*WithOneY` in `__all__` | Create parallel class in `oney/` |
| Using `fraction` dtype by default in pycddlib | Extreme slowdown | Use `"float"` first, fall back to `"fraction"` on numerical error |
| Forgetting `__all__` in new module | `ruff check` warning, import failures | Add `__all__` with all public names |

## Class Hierarchy

```
ActHull (ABC)
├── ReLULikeHull (ABC) — piecewise linear
│   ├── ReLUHull
│   ├── LeakyReLUHull
│   └── ELUHull
├── SShapeHull (ABC) — smooth S-shaped
│   ├── SigmoidHull
│   └── TanhHull
├── MaxPoolHull
└── MaxPoolHullDLP
```

`oney/` mirrors this with `WithOneY` suffix on every class.

## Conventions

- H-representation: all constraints encoded as `b + A @ x >= 0`
- Output shape: `(m, 2d+1)` as `[b | A_x | A_y]` for d-dimensional input
- Mathematical variable names permitted (`N806` suppressed): `Axb`, `c`, `v`
- Tangent line solvers use numba `@njit` for performance

## Related Documents

- [README.md](README.md) — usage examples, API reference
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, adding hulls
- [Root ARCHITECTURE.md](../ARCHITECTURE.md) — rover project structure (wraact is Layer 1 submodule)
