# Wraact Conventions

## Docstrings

Module: 1-line summary + optional 1-2 paragraph extended description.
Follow `__docformat__ = "restructuredtext"`.
Concrete hull (`_relu.py`): 1 line. Base class (`_act.py`): summary.
`__init__.py`: subpackage public API.

Class: `:param name:` for constructor args; `.. tip::` for guidance;
`.. attention::` for warnings; `.. seealso::` for references.

Method: imperative mood, 1-line summary. `:param name:`, `:return:`, `:raises:`.
Shape annotations in backtick format: `` Shape: ``(n, d)``. ``
Abstract methods document the subclass contract.

No non-ASCII in docstrings. Use LaTeX commands or ASCII (`->`, `<=`).

## Architecture

```
Root _*.py modules (no internal deps)
    ▲
    ├── acthull/ (imports from root _*.py)
    │       ▲
    │       └── oney/ (imports from acthull/ and root _*.py)
```

- `oney/` may import from `acthull/`; reverse forbidden.
- Root modules (`_functions.py`, `_tangent_lines.py`) must NOT import from subpackages.
- New activation: add to `_functions.py` → `acthull/_new.py` → `oney/_new.py` → export from all 3 `__init__.py`.
- H-representation: `b + A @ x >= 0`, shape `(m, 2d+1)` as `[b | A_x | A_y]`.

## Naming

| Category | Convention | Example |
|----------|-----------|---------|
| Classes | PascalCase | `ActHull`, `ReLUHull` |
| Functions/methods | snake_case | `cal_hull`, `cal_mn_constrs` |
| Private methods | `_` prefix | `_check_inputs` |
| Private modules | `_` prefix | `_act.py`, `_relu.py` |
| Constants | UPPER_CASE | `TOLERANCE`, `DEBUG` |
| Bool constructor params | `if_` prefix | `if_cal_single_neuron_constrs` |
| Activation functions | `_np` suffix | `relu_np`, `sigmoid_np` |
| Derivatives | `d` prefix | `drelu_np`, `dsigmoid_np` |

Mathematical single-letter names (`c`, `v`, `l`, `u`, `k`) permitted in `src/wraact/*`.

## Argument Patterns

- Bool flags: `if_` prefix.
- Enumerated strings: `Literal["float", "fraction"]` not bare `str`.
- Bound order: lower then upper — `lb, ub`, `l, u`.
- Hull computation: `input_constrs, input_lower_bounds, input_upper_bounds` (all optional).
- Return: single `ndarray` or `tuple[ndarray, ...]`.

## Exception Handling

- Two-tier fallback: try `"float"` first; on error retry with `"fraction"`.
- Custom exceptions: `DegeneratedError`, `NotConvergedError` (extend `Exception`).
- `_record_and_raise_exception`: writes to `.temp/acthull_{timestamp}.log` then raises `RuntimeError`.
- Validate early: `_check_inputs`, `_check_input_bounds`, `_check_input_constrs`.
- `DEBUG=True` skips catch logic, surfaces raw errors.
- Catch specific types: `cdd.Error, RuntimeError, ArithmeticError, ValueError, NotConvergedError`.
- Minimize try-except scope; extract shared fallback into `_cdd_retry`.

## Tangent Lines (Numba JIT)

- `@njit` functions in `_tangent_lines.py`; pure-NumPy in `_functions.py`.
- JIT params: only `ndarray`, `bool`, `float` — no `None`, no `Literal`.
- Convergence: `_MAX_ITER=100`, `_CONVERGE_TOL=1e-4`; raise `NotConvergedError`.
- Clamp log inputs: `np.maximum(x, _LOG_MIN)` before `np.log()`.
- Suppress Numba logging: `logging.getLogger("numba").setLevel(logging.CRITICAL)`.

## DLP Construction

- DLP = Double Linear Piece: piecewise function y = max/min(line1, line2).
- `cal_sn_constrs` builds single-neuron constraints; `cal_mn_constrs` builds multi-neuron.
- Single-piece constraint: `(alpha_l, beta_l, alpha_u, beta_u, x1)` — 5 scalars.
- Constraint matrix (H-rep): `[b | A_x | A_y]` — `(m, 2d+1)`.
- Upper/lower constraints built separately then stacked.
- `cal_mn_constrs_with_one_y_dlp` in `_utils.py` is the core engine.

## 8-step Workflow for New Activation

1. Add `_np` + `d_np` to `_functions.py`
2. Create `acthull/_new.py` with hull class
3. Create `oney/_new.py` with `WithOneY` variant
4. Export from `acthull/__init__.py`
5. Export from `oney/__init__.py`
6. Export from root `__init__.py`
7. Update `__all__` in all 3 init files
8. Add tests

## Test Style

- File: `test_<topic>.py`. Class: `Test<Domain><Aspect>`.
- Method: `test_<action>_<subject>_<condition>`.
- Fixtures in `conftest.py`; no `_helpers.py` files.
- `@pytest.mark.parametrize` for 3+ cases.
- Deterministic RNG: `np.random.default_rng(seed)` with fixed seeds.
- Soundness: sample points uniformly, check `b + A @ point >= -tolerance`, ≥99.9%.
- `pytest.raises(ExceptionType, match=r"regex")` with `match`.
- Test via public API (`cal_hull()`) primarily.
