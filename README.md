# WRAACT: Precise Activation Function Over-Approximation for Neural Network Verification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/wraact/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/wraact/actions/workflows/unit-tests.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Tests](https://img.shields.io/badge/tests-1876%20passed-success)](https://github.com/ZhongkuiMa/wraact/actions/workflows/unit-tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](https://github.com/ZhongkuiMa/wraact)
[![Version](https://img.shields.io/badge/version-2026.1.0-blue.svg)](https://github.com/ZhongkuiMa/wraact/releases)

Precise over-approximation of neural network activation functions for sound abstract interpretation during verification.

## Features

- **Precise Hull Computation**: Compute tight convex hulls for activation functions (ReLU, Sigmoid, Tanh, ELU, LeakyReLU, MaxPool)
- **Sound Over-Approximation**: Guarantee soundness through careful constraint derivation
- **Dual Backend Support**: PyTorch tensors and NumPy arrays
- **Efficient Constraint Generation**: Optimized constraint matrices for verification
- **Comprehensive Activation Coverage**: Support for 10+ activation variants and pooling operations
- **Robust Error Handling**: Proper handling of degenerate polytopes and numerical edge cases
- **WithOneY Optimization**: Fast approximation for single-output scenarios

## Quality Metrics

- **Test Suite**: 1876 comprehensive tests with 100% pass rate
- **Code Coverage**: 92% statement coverage (1279 statements, 99 uncovered)
- **Type Safety**: Fully typed with mypy type checking
- **Code Quality**: Enforced with ruff linter (comprehensive ruleset)
- **Dual Backend**: Verified with both PyTorch and NumPy implementations
- **Lines of Code**: ~1,300 lines (source only, excluding tests)

### Test Coverage Summary

**Test Results: ✅ 1876 passed, 22 warnings**

```
============================= test session starts ==============================
Platform: linux, Python 3.11.6, pytest-9.0.2
Test Coverage: 92% (1279 statements analyzed, 99 uncovered)
Test Execution Time: ~7.89 seconds
============================= 1876 passed in 7.89s ==============================
```

**Module Coverage:**

```
src/wraact/__init__.py                   100% (4/4 statements)
src/wraact/_constants.py                 100% (9/9 statements)
src/wraact/_exceptions.py                100% (12/12 statements)
src/wraact/acthull/__init__.py          100% (11/11 statements)
src/wraact/acthull/_sigmoid.py          100% (20/20 statements)
src/wraact/acthull/_tanh.py             100% (20/20 statements)
src/wraact/oney/__init__.py             100% (10/10 statements)
src/wraact/acthull/_utils.py            100% (38/38 statements)
src/wraact/acthull/_sshape.py            96% (196/196 statements, 8 uncovered)
src/wraact/acthull/_elu.py               99% (67/67 statements, 1 uncovered)
src/wraact/acthull/_relulike.py          97% (38/38 statements, 1 uncovered)
src/wraact/acthull/_relu.py              93% (72/72 statements, 5 uncovered)
src/wraact/acthull/_maxpool.py           92% (168/168 statements, 13 uncovered)
src/wraact/acthull/_leakyrelu.py         88% (59/59 statements, 7 uncovered)
src/wraact/acthull/_act.py               83% (230/230 statements, 39 uncovered)
src/wraact/_functions.py                 91% (43/43 statements, 4 uncovered)
src/wraact/_tangent_lines.py             78% (65/65 statements, 14 uncovered)
```

**TOTAL: 1279 statements, 99 uncovered (7.7% uncovered) = 92% coverage**

### Running All Tests

To run the complete test suite:

```bash
# Run all tests (1876 total)
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/wraact --cov-report=term-missing -v

# Run specific test categories
pytest tests/test_basic/ -v           # Core functionality
pytest tests/test_soundness/ -v       # Sound abstract interpretation
pytest tests/test_performance/ -v     # Performance and scaling
```

## Installation

WRAACT is currently a component of the Rover verification framework and is **not available on PyPI**. Install from source:

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/ZhongkuiMa/wraact.git
cd wraact

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check version
python -c "import wraact; print(wraact.__version__)"
# Expected: 2026.1.0

# Run test suite
pytest tests/ -v
# Expected: 1876 passed

# Run linting
ruff check src/wraact tests
```

### Installation Options

**Development mode (recommended for contributors):**
```bash
pip install -e ".[dev]"  # Includes pytest, ruff, mypy
```

**Minimal install (runtime only):**
```bash
pip install -e .  # Dependencies: pycddlib, numpy, numba
```

### Requirements

- **Python**: 3.11 or higher
- **Runtime dependencies**: pycddlib (CDD library for polytope operations), numpy, numba
- **Development dependencies**: pytest, pytest-cov, ruff, mypy

## Quick Start

### Computing a Hull for ReLU

```python
from wraact import ReLUHull
import numpy as np

# Create a ReLU hull calculator
relu_hull = ReLUHull()

# Define input bounds
lb = np.array([0.0])
ub = np.array([1.0])

# Compute constraints (hyperplane representation of polytope)
constraints = relu_hull.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
# Returns: (A, b) where A*y <= b describes the convex hull

print(f"Hull constraints shape: {constraints.shape}")
```

### Computing with Multiple Dimensions

```python
import wraact
import numpy as np

# 2D ReLU
relu_2d = wraact.ReLUHull()
lb = np.array([-1.0, -2.0])
ub = np.array([3.0, 1.0])
hull = relu_2d.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

# Sigmoid with bounds
sigmoid = wraact.SigmoidHull()
lb = np.array([-5.0])
ub = np.array([5.0])
constraints = sigmoid.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)
```

### Available Activation Functions

**Standard Variants:**
- `ReLUHull` - Rectified Linear Unit
- `SigmoidHull` - Sigmoid activation
- `TanhHull` - Hyperbolic tangent
- `ELUHull` - Exponential Linear Unit
- `LeakyReLUHull` - Leaky ReLU (parametric)
- `MaxPoolHull` - Max pooling operation
- `MaxPoolHullDLP` - Max pooling with DLP optimization

**WithOneY Variants (faster approximations for single output):**
- `ReLUHullWithOneY`
- `SigmoidHullWithOneY`
- `TanhHullWithOneY`
- `ELUHullWithOneY`
- `LeakyReLUHullWithOneY`
- `MaxPoolHullWithOneY`
- `MaxPoolHullDLPWithOneY`

## Core Concepts

### ActHull - Hull Computation

`ActHull` is the base class for all activation hull computations:

- **`cal_hull(input_lower_bounds, input_upper_bounds)`** - Compute polytope constraints
  - Returns constraint matrix where `A*y <= b`
  - `A` has shape `(num_constraints, num_inputs)`
  - Last column contains intercept `b`

### Sound Over-Approximation

All hull computations guarantee **soundness**: the computed polytope is a valid outer approximation of the true activation function graph. This is critical for verification:

- Forward bounds: If `x` is in input bounds, then `f(x)` is in output bounds
- Constraint satisfaction: All feasible points satisfy all constraints
- Tightness: Constraints are chosen to minimize over-approximation

## API Overview

### Main Classes

**Activation Hull Classes**
- All inherit from `ActHull`
- Implement `cal_hull()` method
- Support both NumPy and PyTorch tensors via `TensorLike` type

**Public API Functions**
```python
from wraact import (
    # Standard hulls
    ActHull,
    ReLUHull,
    SigmoidHull,
    TanhHull,
    ELUHull,
    LeakyReLUHull,
    ReLULikeHull,
    MaxPoolHull,
    MaxPoolHullDLP,

    # WithOneY variants
    ActHullWithOneY,
    ReLUHullWithOneY,
    SigmoidHullWithOneY,
    TanhHullWithOneY,
    ELUHullWithOneY,
    LeakyReLUHullWithOneY,
    ReLULikeHullWithOneY,
    MaxPoolHullWithOneY,
    MaxPoolHullDLPWithOneY,

    # Utility function
    cal_mn_constrs_with_one_y_dlp,

    # Version
    __version__,
)
```

### Exception Classes

```python
from wraact import DegeneratedError, NotConvergedError

# Raised when polytope is degenerate (collapsed dimensions)
try:
    hull = ReLUHull()
    constraints = hull.cal_hull(lb, ub)
except DegeneratedError as e:
    print(f"Degenerate polytope: {e}")
```

## Examples

### Example 1: Verify ReLU Bounds

```python
import numpy as np
from wraact import ReLUHull

relu = ReLUHull()

# Input in [-2, 2]
lb = np.array([-2.0])
ub = np.array([2.0])

# Compute hull
constraints = relu.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

# Verify: max(0, -2) to max(0, 2) = 0 to 2
print(f"Hull computed successfully with {constraints.shape[0]} constraints")
```

### Example 2: Handle Degenerate Cases

```python
from wraact import ReLUHull, DegeneratedError
import numpy as np

relu = ReLUHull()

# Degenerate case: single point
try:
    constraints = relu.cal_hull(
        input_lower_bounds=np.array([1.0]),
        input_upper_bounds=np.array([1.0])
    )
    print("Single point is non-degenerate")
except DegeneratedError:
    print("Single point polytope is degenerate")
```

### Example 3: Multi-dimensional Activation

```python
from wraact import SigmoidHull
import numpy as np

sigmoid = SigmoidHull()

# 3D sigmoid
lb = np.array([-1.0, 0.0, 1.0])
ub = np.array([1.0, 2.0, 3.0])

constraints = sigmoid.cal_hull(input_lower_bounds=lb, input_upper_bounds=ub)

print(f"Constraints shape: {constraints.shape}")
print(f"Number of constraints: {constraints.shape[0]}")
```

## Project Structure

```
wraact/
├── src/wraact/
│   ├── __init__.py              # Main exports
│   ├── _constants.py            # Shared constants
│   ├── _exceptions.py           # DegeneratedError, NotConvergedError
│   ├── _functions.py            # Activation function definitions
│   ├── _tangent_lines.py        # Tangent line computation
│   ├── acthull/                 # Activation hull implementations
│   │   ├── __init__.py
│   │   ├── _act.py              # Base ActHull class
│   │   ├── _relu.py             # ReLU implementation
│   │   ├── _sigmoid.py          # Sigmoid implementation
│   │   ├── _tanh.py             # Tanh implementation
│   │   ├── _elu.py              # ELU implementation
│   │   ├── _leakyrelu.py        # LeakyReLU implementation
│   │   ├── _relulike.py         # ReLU-like activations
│   │   ├── _maxpool.py          # MaxPool implementation
│   │   ├── _sshape.py           # S-shaped functions (Sigmoid/Tanh)
│   │   └── _utils.py            # Shared utilities
│   └── oney/                    # WithOneY optimized variants
│       ├── __init__.py
│       ├── _act.py              # Base ActHullWithOneY
│       ├── _relu.py
│       ├── _sigmoid.py
│       ├── _tanh.py
│       ├── _elu.py
│       ├── _leakyrelu.py
│       ├── _relulike.py
│       ├── _maxpool.py
│       └── _sshape.py
├── tests/
│   ├── test_basic/              # Core functionality tests
│   ├── test_soundness/          # Soundness verification tests
│   └── test_performance/        # Performance and scalability tests
├── pyproject.toml               # Package configuration
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .github/workflows/           # GitHub Actions CI/CD
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Contributing Guidelines

We welcome contributions! Please follow these guidelines to ensure smooth collaboration.

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/wraact.git
cd wraact

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify setup
pytest tests/ -v
ruff check src/wraact tests
```

### Branch Naming Conventions

Use descriptive branch names with prefixes:
- `feature/` - New features (e.g., `feature/add-new-activation`)
- `fix/` - Bug fixes (e.g., `fix/degenerate-polytope-handling`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-constraints`)
- `docs/` - Documentation updates (e.g., `docs/improve-examples`)
- `test/` - Test improvements (e.g., `test/add-edge-cases`)

### Commit Message Format

Write clear, concise commit messages:

```
<type>: <short summary in present tense>

<optional detailed description>

<optional footer with issue references>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `test:` - Add or update tests
- `docs:` - Documentation changes
- `style:` - Code style/formatting
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### Pull Request Guidelines

1. **Before creating a PR:**
   ```bash
   pytest tests/ -v
   ruff check src/wraact tests
   ruff format src/wraact tests
   python -m mypy
   ```

2. **Create PR with:**
   - Clear title following commit message format
   - Description explaining changes and why
   - Tests for new features/fixes
   - Documentation updates if needed

3. **Review process:**
   - Maintainers review within 48-72 hours
   - Address feedback by pushing to PR branch
   - Once approved, maintainers will merge

### Running CI Locally

Before pushing, run the same checks as GitHub Actions:

```bash
pytest tests/ --cov=src/wraact --cov-report=term-missing -v
ruff check src/wraact tests
ruff format --check src/wraact tests
python -m mypy
```

### Code Style

WRAACT follows strict code quality standards:

- **Formatter**: `ruff format` (100 char line length)
- **Linter**: `ruff check` (comprehensive ruleset)
- **Type checker**: `mypy`
- **Docstrings**: PEP 257 style with type hints

### Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_basic/ -v

# Run with coverage
pytest tests/ --cov=src/wraact --cov-report=term-missing -v
```

### Pre-commit Hooks (Optional)

Install pre-commit hooks to automatically check code before commits:

```bash
pip install pre-commit
pre-commit install

# Hooks will run automatically on git commit
# Or run manually:
pre-commit run --all-files
```

## License

MIT License - see [LICENSE](LICENSE)

## Contact

For questions, bug reports, or feature requests, please open an issue on GitHub.
