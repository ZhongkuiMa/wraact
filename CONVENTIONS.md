# Wraact Conventions

This file defines style and documentation conventions for the wraact package.
Use it as a **checklist** — when writing or reviewing code, check each item below
one by one.

---

## 1. Module Docstrings

Every `.py` file begins with a module docstring.

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 1.1 | **First line**: short summary of the module's purpose (one sentence) | ☐ |
| 1.2 | **Extended description** (optional): 1-2 paragraphs after a blank line, covering the module's role or usage guidance | ☐ |
| 1.3 | **Format**: ReST plain text, no `:param:` or `:return:` tags at module level | ☐ |
| 1.4 | Always followed by `__docformat__ = "restructuredtext"` | ☐ |
| 1.5 | **No author, date, or version lines** — git history is authoritative | ☐ |
| 1.6 | **No non-ASCII characters** in docstrings — use LaTeX commands (`\text`, `\land`) or ASCII equivalents for symbols (e.g., `->` not `→`, `<=` not `≤`) | ☐ |

### Patterns

| File type | Style | Example |
|-----------|-------|---------|
| Concrete hull module (`_relu.py`, `_sigmoid.py`) | One line | `"""ReLU activation hull computation."""` |
| Base class module (`_act.py`, `_sshape.py`) | One line | `"""Base class for activation function convex hull computation."""` |
| Utility module (`_constants.py`, `_exceptions.py`) | One line | `"""Custom exceptions for wraact hull computations."""` |
| Complex module (`_functions.py`, `_tangent_lines.py`) | Summary + 1-2 paragraph extended description | See `_tangent_lines.py` or `_functions.py` |
| `__init__.py` | Summary of subpackage public API with listed classes | See `acthull/__init__.py` |

---

## 2. Class Docstrings

### 2.1 Structure

```python
class MyHull(BaseClass):
    """
    One-line summary of what the class computes.

    Extended description (optional) — the algorithm or approach.

    :param param1: Description of constructor parameter (capitalized, ends with period).
    :param param2: Description of constructor parameter.

    .. tip::
        Usage guidance or performance notes for callers.

    .. attention::
        Warnings about cost, precision trade-offs, or restrictive conditions.
    """
```

### 2.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 2.1 | **First line**: describes what the class computes, ends with period | ☐ |
| 2.2 | Constructor parameters documented in class docstring with `:param name:` — capitalized, ends with period | ☐ |
| 2.3 | `:param` descriptions: describe **semantics**, not Python types (types go in annotations) | ☐ |
| 2.4 | `__init__` may have its own docstring for parameter details when the class docstring is long; `:raises ValueError:` describes validation failures in `__init__` | ☐ |
| 2.5 | Use `.. tip::` for usage guidance, performance notes, or mode recommendations | ☐ |
| 2.6 | Use `.. attention::` for warnings about cost, precision trade-offs, or restrictive conditions | ☐ |
| 2.7 | Use `.. seealso::` for paper citations and related references (with `:cite:` role) | ☐ |
| 2.8 | No docstring on `__init__` of a simple dataclass-like class (class docstring covers it) | ☐ |

### 2.3 Good examples

```python
class ActHull(ABC):
    """
    An object used to calculate the function hull of the activation function.

    :param if_cal_single_neuron_constrs: Whether to calculate single-neuron
        constraints.
    :param if_cal_multi_neuron_constrs: Whether to calculate multi-neuron
        constraints.

    .. tip::
        The multi-neuron constraints here means those constraints that cannot obtained
        by trivial methods or the properties of the activation function.

    .. attention::
        When enabled, it cost more time and generate (almost double) constraints.
    """
```

```python
class ReLUHull(ReLULikeHull):
    """
    This is to calculate the function hull for the rectified linear unit (ReLU)
    activation function.

    .. tip::
        This is an ad hoc implementation for ReLU to obtain the function hull
        considering high efficiency and accuracy based on the two linear pieces
        (:math:`y=x` and :math:`y=0`) of ReLU.
    """
```

```python
class DegeneratedError(Exception):
    """
    An exception for degenerated input polytope when calculating function hull.

    It means the number of vertices is fewer than the dimension.
    """
```

---

## 3. Method Docstrings

### 3.1 Structure

```
def method_name(self, param1: type, param2: type) -> return_type:
    """
    Short imperative description of what the method computes.

    Extended description (optional) — the algorithm or formula.

    .. tip::
        Optional guidance for callers or subclasses.

    :param param1: Description of param1 (capitalized, ends with period).
        Shape: ``(n, d)``.
    :param param2: Description of param2.

    :return: Description of return value(s) (capitalized, ends with period).
        Shape: ``(num_constraints, 2*d+1)``.
    :raises ValueError: When and why this exception is raised.
    :raises DegeneratedError: When the input polytope is degenerate.
    """
```

### 3.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 3.1 | **First line**: imperative mood, describes what the method computes, ends with period | ☐ |
| 3.2 | Use `:param name:`, `:return:`, and `:raises ExceptionType:` tags — no `:type:` tags (types go in annotations) | ☐ |
| 3.3 | `:param` descriptions: **capitalized, end with period**, describe semantics; include shape in backtick-formatted inline like `` Shape: ``(n, d)``. `` | ☐ |
| 3.4 | `:return` description: **capitalized, end with period**; include output shape | ☐ |
| 3.5 | `:raises` descriptions: **capitalized, end with period**; describe the condition that triggers the exception | ☐ |
| 3.6 | Use `r"""` (raw triple-quotes) when docstring contains `:math:` | ☐ |
| 3.7 | Abstract methods must have docstrings describing the contract subclasses must fulfill | ☐ |
| 3.8 | Static methods and class methods use the same docstring format as instance methods | ☐ |
| 3.9 | **No non-ASCII characters** in docstrings — same as 1.6 | ☐ |

### 3.3 Good examples

```python
def cal_hull(
    self,
    input_constrs: ndarray | None = None,
    input_lower_bounds: ndarray | None = None,
    input_upper_bounds: ndarray | None = None,
) -> ndarray | None:
    """
    Calculate the function hull of an activation function.

    There are two usage modes:

    1. **Single-neuron mode**: Provide only ``input_lower_bounds`` and
       ``input_upper_bounds``. Requires ``if_cal_single_neuron_constrs=True``.

    2. **Multi-neuron mode**: Provide ``input_constrs`` with optionally
       ``input_lower_bounds`` and ``input_upper_bounds``.

    :param input_constrs: Input polytope constraints in H-representation.
        Shape: ``(n, d+1)``. Format: ``[b | A]`` where each row represents
        ``b + A @ x >= 0``.
    :param input_lower_bounds: Lower bounds for each input variable.
        Shape: ``(d,)``.
    :param input_upper_bounds: Upper bounds for each input variable.
        Shape: ``(d,)``.
    :return: Constraint matrix in H-representation defining the function hull.
        Shape: ``(num_constraints, 2*d+1)``.
    :raises ValueError: If parameters are invalid or bounds don't match dimensions.
    :raises DegeneratedError: If the input polytope is degenerate.
    """
```

```python
@staticmethod
def _build_input_bounds_constraints(s: ndarray, is_lower: bool = True) -> ndarray:
    """
    Build the constraints based on the lower or upper bounds of the input variables.

    :param s: The lower or upper bounds of the input variables.
    :return: The constraints based on the lower or upper bounds of the input
        variables.
    """
```

```python
def get_second_tangent_line_sigmoid_np(
    x1: ndarray, get_big: bool
) -> tuple[ndarray, ndarray, ndarray]:
    """Find second tangent line to sigmoid passing through point x1.

    Uses iterative method to find a tangent line that passes through
    the point (x1, sigmoid(x1)) on the sigmoid curve.

    :param x1: First tangent point x-coordinates. Shape: (n,).
    :param get_big: If True, return upper tangent; else lower tangent.
    :return: Tuple of (b, k, x2) where b is intercept, k is slope, x2 is
        second tangent point. Each has shape (n,).
    :raises NotConvergedError: If iteration does not converge.
    """
```

---

## 4. Inline Comments

| # | Rule | Pass/Fail |
|---|------|-----------|
| 4.1 | Comment **why**, not what — the code already says what | ☐ |
| 4.2 | Only add comments when the reasoning is non-obvious (algorithm rationale, workarounds) | ☐ |
| 4.3 | **No inline shape comments** on function signatures — shapes belong in `:param:`/`:return:` docstrings with backtick format `` Shape: ``(n, d)``. `` | ☐ |
| 4.4 | `#:` prefix for attribute docstrings on module-level constants and `ClassVar` fields | ☐ |
| 4.5 | `# pragma: no cover` on defensive branches that cannot be reached in normal execution | ☐ |
| 4.6 | No commented-out code — delete it | ☐ |
| 4.7 | `# TODO:` comments require an associated issue reference (enforced by ruff TD001); `task-tags = ["TODO", "FIXME"]` in pyproject.toml accepts both tags | ☐ |
| 4.8 | Multi-line block comments inside functions use triple-quoted strings for algorithm rationale (e.g., float→fraction fallback explanation) | ☐ |

---

## 5. Naming Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 5.1 | **Classes**: PascalCase — `ActHull`, `ReLUHull`, `SigmoidHullWithOneY` | ☐ |
| 5.2 | **Methods/functions**: snake_case — `cal_hull`, `cal_constrs`, `get_parallel_tangent_line_sigmoid_np` | ☐ |
| 5.3 | **Private methods**: `_` prefix — `_check_inputs`, `_cal_hull_with_mn_constrs`, `_build_input_bounds_constraints` | ☐ |
| 5.4 | **Private modules**: `_` prefix — `_act.py`, `_constants.py`, `_relu.py`. Exception: `__init__.py` | ☐ |
| 5.5 | **Constants**: UPPER_CASE — `TOLERANCE`, `MIN_BOUNDS_RANGE_ACTHULL`, `DEBUG` | ☐ |
| 5.6 | **Boolean constructor parameters**: `if_` prefix — `if_cal_single_neuron_constrs`, `if_use_double_orders`, `if_return_input_bounds_by_vertices` | ☐ |
| 5.7 | **Abstract methods**: short names describing the contract — `_f`, `_df`, `cal_constrs`, `cal_sn_constrs`, `cal_mn_constrs`, `_construct_dlp` | ☐ |
| 5.8 | **Activation functions**: `_np` suffix for NumPy implementations — `relu_np`, `sigmoid_np`. Derivatives: `d` prefix — `drelu_np`, `dsigmoid_np`. Second derivatives: `dd` prefix — `ddsigmoid_np` | ☐ |
| 5.9 | **Mathematical variable names** permitted: single letters `c`, `v`, `d`, `l`, `u`, `k`, `b` (N806 suppressed for `src/wraact/*`) | ☐ |
| 5.10 | **Module-level privacy**: modules prefixed `_` are internal; public modules are exported via `__init__.py` | ☐ |

---

## 6. Argument Patterns

| # | Rule | Pass/Fail |
|---|------|-----------|
| 6.1 | **Boolean flags**: use `if_` prefix — `if_cal_single_neuron_constrs`, `if_use_double_orders` | ☐ |
| 6.2 | **Enumerated string args**: use `Literal["float", "fraction"]`, not bare `str` | ☐ |
| 6.3 | **Bound order**: lower then upper — `lb, ub`, `input_lower_bounds, input_upper_bounds`, `l, u` — never reversed | ☐ |
| 6.4 | **Constraint methods**: `c, v, lb, ub` or `c, v, l, u` (constraints, vertices, lower, upper) | ☐ |
| 6.5 | **Hull computation**: `input_constrs, input_lower_bounds, input_upper_bounds` (all optional, at least one required) | ☐ |
| 6.6 | **Shape annotations**: include array shapes in `:param:`/`:return:` docstrings — no inline `# (n, d)` comments on signatures | ☐ |
| 6.7 | **Return type**: single `ndarray` or `tuple[ndarray, ...]`; use `ndarray | None` when computation can fail gracefully | ☐ |
| 6.8 | **Numba JIT functions**: all parameters must be Numba-compatible types (`ndarray`, `bool`, `float`); no `None` defaults in `@njit` functions | ☐ |
| 6.9 | **Constructor defaults**: `if_cal_single_neuron_constrs=False`, `if_cal_multi_neuron_constrs=True`, `if_use_double_orders=False`, `dtype_cdd="float"` | ☐ |

---

## 7. Code Style

| # | Rule | Pass/Fail |
|---|------|-----------|
| 7.1 | **100-char line length** (enforced by ruff) | ☐ |
| 7.2 | **Double quotes** for strings and docstrings (enforced by ruff) | ☐ |
| 7.3 | **Absolute imports only** — `from wraact.acthull._act import ActHull` (enforced by ruff `ban-relative-imports = "all"`) | ☐ |
| 7.4 | `__docformat__ = "restructuredtext"` after module docstring, before imports | ☐ |
| 7.5 | `__all__` in every source module, alphabetically sorted, listing all public names. Not required in test files. | ☐ |
| 7.6 | **Import order**: stdlib → third-party (`cdd`, `numpy`, `numba`) → first-party (`wraact.*`). Groups separated by blank lines. | ☐ |
| 7.7 | `import numpy as np`; `from numpy import ndarray` (for type annotations) | ☐ |
| 7.8 | **No in-place array ops** that mutate inputs — copy first with `np.array(x, dtype=np.float64)` | ☐ |
| 7.9 | **Explicit float64**: convert input arrays to `np.float64` at entry points for numerical stability | ☐ |
| 7.10 | **McCabe complexity ≤ 10** (enforced by ruff C90) — split complex methods | ☐ |
| 7.11 | **NumPy only** — no PyTorch, no TensorFlow. Dependencies: `numpy`, `pycddlib`, `numba`. | ☐ |
| 7.12 | **No `cast(Tensor, ...)`** — use `cast(ndarray | float, expr)` only when needed for type-narrowing `@overload` or `ndarray | float` disambiguation | ☐ |
| 7.13 | **Only import what you use** — clean up unused imports (enforced by ruff F401) | ☐ |
| 7.14 | `from typing import ClassVar, Literal, NoReturn` — use typing imports at module level, not inline | ☐ |
| 7.15 | **No string annotations** when type is already imported — write `-> ndarray` not `-> "ndarray"` | ☐ |

---

## 8. Class Design Patterns

| # | Rule | Pass/Fail |
|---|------|-----------|
| 8.1 | **ABC hierarchy**: abstract bases use `ABC` and `@abstractmethod`; concrete classes implement all abstract methods | ☐ |
| 8.2 | **`__slots__`**: define on base classes to constrain attribute creation; subclasses extend via `[*Parent.__slots__, "_new_attr"]` | ☐ |
| 8.3 | **`ClassVar`**: use for class-level caches (`_reversed_orders`, `_lower_constraints`) with type annotation | ☐ |
| 8.4 | **Template method pattern**: public `cal_hull()` calls private `_cal_hull_with_mn_constrs()` / `_cal_hull_with_sn_constrs()` which call abstract `cal_constrs()` / `cal_sn_constrs()` / `cal_mn_constrs()` | ☐ |
| 8.5 | **Static methods**: pure computation (no `self`/`cls` state) — `cal_vertices`, `_check_vertices`, `_f`, `_df` | ☐ |
| 8.6 | **Class methods**: polymorphic dispatch that may vary by subclass but doesn't need instance state — `cal_sn_constrs`, `cal_mn_constrs` | ☐ |
| 8.7 | **Properties**: use `@property` for derived attributes (e.g., `dtype_cdd`) | ☐ |
| 8.8 | **Private attributes**: store constructor args with `_` prefix — `self._add_sn_constrs`, `self._dtype_cdd` | ☐ |
| 8.9 | **Deferred imports**: use local imports inside methods to avoid circular imports between sibling modules (e.g., `from wraact._tangent_lines import ...` inside a method) | ☐ |

---

## 9. Exception Handling

| # | Rule | Pass/Fail |
|---|------|-----------|
| 9.1 | **Two-tier fallback**: try `"float"` arithmetic first; on `cdd.Error` / `RuntimeError` / `ArithmeticError` / `ValueError`, retry with `"fraction"` (exact rational arithmetic) | ☐ |
| 9.2 | **Custom exceptions**: extend `Exception` directly — `DegeneratedError`, `NotConvergedError`; include `__init__` with default message and `__str__` for formatting | ☐ |
| 9.3 | **Error logging**: `_record_and_raise_exception` writes failure details to `.temp/acthull_{timestamp}.log` before re-raising as `RuntimeError` | ☐ |
| 9.4 | **Validate early**: `_check_inputs`, `_check_input_bounds`, `_check_input_constrs` called at the top of `cal_hull()` | ☐ |
| 9.5 | **Defensive asserts**: use `assert` for invariants that indicate bugs (e.g., `assert d is not None` after input parsing) | ☐ |
| 9.6 | **`DEBUG` flag**: when `DEBUG=True` in `_constants.py`, skip exception handling to surface raw errors during development | ☐ |
| 9.7 | Catch **specific** exception types (`cdd.Error, RuntimeError, ArithmeticError, ValueError`), never bare `except:` or `except Exception:` | ☐ |
| 9.8 | **Minimize try-except scope**: wrap only the specific operation that can fail, not entire function bodies. Do not nest identical try-except blocks; extract shared fallback logic into helpers | ☐ |

---

## 10. Tangent Lines and Numba JIT

| # | Rule | Pass/Fail |
|---|------|-----------|
| 10.1 | Numba `@njit` functions go in `_tangent_lines.py`; pure-NumPy activation functions go in `_functions.py` | ☐ |
| 10.2 | JIT functions take only Numba-compatible types: `ndarray`, `bool`, `float` — no `None`, no `Literal`, no `ndarray | float` unions | ☐ |
| 10.3 | **Warmup**: call `_warmup_jit_functions()` at module import time to trigger JIT compilation before first use | ☐ |
| 10.4 | **Convergence**: iterative solvers use module-level constants `_MAX_ITER`, `_CONVERGE_TOL`; raise `NotConvergedError` if exceeded | ☐ |
| 10.5 | **Numerical stability**: use `np.maximum(x, _LOG_MIN)` before `np.log()`; use `np.errstate(divide="ignore", invalid="ignore")` for division-by-zero guards | ☐ |
| 10.6 | Suppress Numba logging at module level: `logging.getLogger("numba").setLevel(logging.CRITICAL)` | ☐ |

---

## 11. Test Style

### 11.1 Directory Layout

```
tests/
├── conftest.py              # shared fixtures and helpers
├── test_arch/               # architecture/import enforcement
├── test_units/
│   ├── test_basic/          # per-activation unit tests
│   ├── test_integration/    # pipeline tests
│   ├── test_soundness/      # correctness guarantees
│   └── test_performance/    # benchmarks (opt-in)
└── fuzzing/                 # random edge-case generation
```

### 11.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 11.1 | **Test file naming**: `test_<topic>.py` — `test_relu.py`, `test_sigmoid.py`, `test_tangent_lines.py` | ☐ |
| 11.2 | **Test class naming**: `Test<Domain><Aspect>` — `TestReLUHullBasic`, `TestActHullInputValidation` | ☐ |
| 11.3 | **Test method naming**: `test_<action>_<subject>_<condition>` — `test_cal_hull_returns_ndarray`, `test_cal_hull_output_shape` | ☐ |
| 11.4 | **Fixtures in conftest**: shared polytope fixtures, hull class fixtures, tolerance fixtures, sampling helpers go in `tests/conftest.py` | ☐ |
| 11.5 | **Parametrize**: use `@pytest.mark.parametrize` for 3+ similar test cases; fixture-level `params=` for dimension sweeps | ☐ |
| 11.6 | **No `_helpers.py` files**: shared helpers are standalone functions in `conftest.py` or imported from `tests.conftest` | ☐ |
| 11.7 | **Deterministic randomness**: use `np.random.default_rng(seed)` with fixed seeds (`42, 43, 44`) | ☐ |
| 11.8 | **Monte Carlo soundness**: sample points uniformly in bounds, check `b + A @ point >= -tolerance`, assert satisfaction rate ≥ 99.9% | ☐ |
| 11.9 | **Exception testing**: use `pytest.raises(ExceptionType, match=r"regex")` with `match` pattern | ☐ |
| 11.10 | **Graceful skip**: use `pytest.skip("reason")` inside `try/except` for cases where hull computation may legitimately fail on degenerate inputs | ☐ |
| 11.11 | **Test via public API**: prefer testing through `cal_hull()`; test `cal_sn_constrs` / `cal_mn_constrs` only for edge-case coverage | ☐ |
| 11.12 | **Module docstrings in tests**: 1-3 lines summarizing what the file validates; `Key Learning for Template` section for shared patterns | ☐ |
| 11.13 | **`pytest.skip` on import errors**: guard hull class fixtures with `try/except ImportError` + `pytest.skip` | ☐ |
| 11.14 | **Performance tests excluded by default**: run explicitly with `pytest tests/test_units/test_performance` | ☐ |
| 11.15 | **Default test suite**: `pytest` runs `tests/test_units/` by default. Performance tests in `test_performance/` are excluded | ☐ |
| 11.16 | **No `@pytest.mark.skip`** in committed code — use conditional early return with `[REVIEW]` comment | ☐ |

---

## 12. Constants Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 12.1 | **Naming**: `UPPER_SNAKE_CASE`, 2-4 words. Use prefixes (`DEFAULT_`, `MAX_`, `MIN_`, `DEBUG_`) and suffixes for clarity | ☐ |
| 12.2 | **Scope levels**: Place at narrowest scope — function-level → file-level → subfolder `_constants.py` → package-level. Promote when a second consumer at broader scope appears | ☐ |
| 12.3 | **Extraction trigger**: Extract a literal when it appears 2+ times. Never duplicate a constant across files | ☐ |
| 12.4 | **When NOT to extract**: Self-documenting single-use values, test data, function defaults already named by the parameter | ☐ |
| 12.5 | **Type annotations**: Annotate only when the type is not obvious from the literal | ☐ |
| 12.6 | **Frozen collections**: Use `frozenset` or `tuple` for constant collections — never mutable | ☐ |

---

## 13. Architecture Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 13.1 | **`oney/` may import from `acthull/`** — `from wraact.acthull import ReLUHull` for inheritance | ☐ |
| 13.2 | **`acthull/` must NOT import from `oney/`** — enforced by `test_arch/test_imports.py` via AST analysis | ☐ |
| 13.3 | **Root modules (`_functions.py`, `_tangent_lines.py`, `_constants.py`, `_exceptions.py`) must NOT import from `acthull/` or `oney/`** | ☐ |
| 13.4 | **Subpackages import from root `_*.py` modules** — `from wraact._constants import TOLERANCE` | ☐ |
| 13.5 | **New activation follows 8-step workflow**: add to `_functions.py` → create `acthull/_new.py` → create `oney/_new.py` → export from all 3 `__init__.py` → add to `__all__` → add tests → validate | ☐ |
| 13.6 | **H-representation format**: all constraints encoded as `b + A @ x >= 0` with shape `(m, 2d+1)` as `[b | A_x | A_y]` | ☐ |

---

## 14. Module Organization

wraact source is organized into a three-tier structure.

### 13.1 Package structure

| Tier | Location | Contents |
|------|----------|----------|
| Root utilities | `_functions.py`, `_tangent_lines.py`, `_constants.py`, `_exceptions.py` | Shared activation functions, tangent line solvers, constants, custom exceptions |
| Activation hulls | `acthull/` | Per-activation convex hull computation classes |
| Single-output constraints | `oney/` | Single-neuron output constraint variants |

### 13.2 Dependency flow

```
Root _*.py modules (no internal deps)
    ▲
    ├── acthull/ (imports from root _*.py)
    │       ▲
    │       └── oney/ (imports from acthull/ and root _*.py)
```

Dependency rules are enforced in §12 (Architecture Rules).

| # | Rule | Pass/Fail |
|---|------|-----------|
| 14.1 | Each activation has modules in both `acthull/` and `oney/` — `_relu.py`, `_sigmoid.py`, etc. | ☐ |
| 14.2 | All three `__init__.py` files (root, acthull, oney) re-export their public symbols | ☐ |
| 14.3 | New modules follow the existing file naming: `_<activation>.py` in both `acthull/` and `oney/` | ☐ |

---

## 15. DLP (Double Linear Piece) Conventions

The DLP is the central data structure for hull constraints. Each constraint
is a double linear piece: two linear functions joined at a breakpoint.

### 14.1 Data structures

| Structure | Contents | Shape |
|-----------|----------|-------|
| Single-piece constraint | `(alpha_l, beta_l, alpha_u, beta_u, x1)` | 5 scalars per constraint |
| Constraint matrix (H-rep) | `[b \| A_x \| A_y]` | `(m, 2d+1)` |

`alpha_l, beta_l` are the lower linear piece coefficients; `alpha_u, beta_u`
are the upper linear piece coefficients; `x1` is the breakpoint.

### 14.2 Construction

| # | Rule | Pass/Fail |
|---|------|-----------|
| 15.1 | DLP constraints are built from tangent lines computed by functions in `_tangent_lines.py` | ☐ |
| 15.2 | Upper and lower constraints are built separately then stacked | ☐ |
| 15.3 | Degenerate inputs (fewer vertices than dimension) raise `DegeneratedError` before DLP construction | ☐ |
| 15.4 | Multi-neuron constraints combine per-neuron DLPs with cross-neuron terms | ☐ |
| 15.5 | `cal_sn_constrs` builds single-neuron constraints; `cal_mn_constrs` builds multi-neuron constraints | ☐ |

