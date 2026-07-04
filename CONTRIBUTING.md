---
type: DESCRIPTION
note: "Descriptive. Mirrors current code; update to follow code changes. < functional code."
---
> **This file IS**: contributor workflow for wraact. **It is NOT**: implementation code or detailed conventions.

# Contributing to WRAACT

Shared conventions (imports, formatting, docstrings, pre-commit) are in the
[root CONTRIBUTING.md](../CONTRIBUTING.md). This file covers wraact-specific workflow only.

## Setup

Install system libraries (pycddlib 3.x depends on cddlib + GMP):

- **macOS:**
  ```bash
  brew install cddlib gmp
  export CPATH="$(brew --prefix)/include"
  export LIBRARY_PATH="$(brew --prefix)/lib"
  ```
- **Ubuntu:** `sudo apt-get install libcdd-dev libgmp-dev`

```bash
cd wraact
pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
pre-commit run --all-files  # lint, format, type-check
pytest tests/ -v            # tests
```

## Workflow

1. Create branch from `main`
2. Make changes following the domain workflow below
3. Run checks (above)
4. Commit and push

## Domain Workflow: Adding an Activation Hull

Most contributions add a new activation function hull. Follow this pattern:

1. Add activation + derivative in `src/wraact/_functions.py` (e.g., `myfunc_np`, `dmyfunc_np`)
2. Choose base class:
   - Piecewise linear (kink at zero) -> inherit `ReLULikeHull` in `acthull/_relulike.py` pattern
   - S-shaped (smooth, monotone) -> inherit `SShapeHull` in `acthull/_sshape.py` pattern
   - Multi-input (pooling) -> inherit `ActHull` directly
3. Create `src/wraact/acthull/_myfunc.py` with hull class
4. Create `src/wraact/oney/_myfunc.py` with single-output variant (`WithOneY`)
5. Export from `acthull/__init__.py`, `oney/__init__.py`, and top-level `__init__.py`
6. Add to `__all__` in all three `__init__.py` files (alphabetically sorted)
7. Add tests in `tests/` covering soundness and edge cases
8. Validate: `pytest tests/ -v`

## Constraints

| Rule | Details |
|------|---------|
| Absolute imports only | `from wraact.acthull._act import ActHull` (no relative) |
| `__docformat__` + `__all__` | Required in every module |
| NumPy only (no PyTorch) | All computation uses numpy + pycddlib + numba |
| H-representation format | Constraints as `b + A @ x >= 0` |
| McCabe complexity <= 10 | Enforced by ruff C90 |
| Mathematical var names allowed | `N806` ignored for `src/wraact/*` |
