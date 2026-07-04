# WRAACT

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/wraact/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/wraact/actions/workflows/unit-tests.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

Compute convex hulls over neural network activation functions with sound over-approximation guarantees for abstract interpretation.

Supported activations: ReLU, Sigmoid, Tanh, ELU, LeakyReLU, MaxPool.

## Installation

### System Dependencies

pycddlib 3.x depends on system-level cddlib + GMP libraries:

- **macOS:**
  ```bash
  brew install cddlib gmp
  export CPATH="$(brew --prefix)/include"
  export LIBRARY_PATH="$(brew --prefix)/lib"
  ```
- **Ubuntu/Debian:** `sudo apt-get install libcdd-dev libgmp-dev`
- **Fedora:** `sudo dnf install cddlib-devel gmp-devel`

### Python Package

```bash
git clone https://github.com/ZhongkuiMa/wraact.git
cd wraact
pip install -e ".[dev]"  # or pip install -e . for runtime only
```

**Requirements:** Python 3.11+, pycddlib, numpy, numba

## Quick Start

```python
import numpy as np
from wraact import SigmoidHull

hull = SigmoidHull()

# Input region in H-representation: b + a1*x1 + a2*x2 >= 0
input_constrs = np.array([
    [0.0,  1.0,  0.0],   # x1 >= 0
    [0.0,  0.0,  1.0],   # x2 >= 0
    [1.0, -1.0, -1.0],   # x1 + x2 <= 1
])
lb = np.array([0.0, 0.0])
ub = np.array([1.0, 1.0])

# Returns (m, 2d+1) array: b + A_x @ x + A_y @ y >= 0
output_constrs = hull.cal_hull(
    input_constrs=input_constrs,
    input_lower_bounds=lb,
    input_upper_bounds=ub,
)
print(f"Generated {output_constrs.shape[0]} hull constraints")
```

## Usage Guide

### Hull Classes

Each activation has a full hull class and a `WithOneY` variant that extends one output dimension for multi-neuron constraint computation.

| Full hull | Single-output variant | Activation |
|-----------|----------------------|------------|
| `ReLUHull` | `ReLUHullWithOneY` | max(0, x) |
| `SigmoidHull` | `SigmoidHullWithOneY` | 1/(1+exp(-x)) |
| `TanhHull` | `TanhHullWithOneY` | tanh(x) |
| `ELUHull` | `ELUHullWithOneY` | ELU |
| `LeakyReLUHull` | `LeakyReLUHullWithOneY` | LeakyReLU |
| `MaxPoolHull` | `MaxPoolHullWithOneY` | max pooling |
| `MaxPoolHullDLP` | `MaxPoolHullDLPWithOneY` | max pooling (DLP) |

Base classes `ActHull` and `ActHullWithOneY` support subclassing for custom activations. Intermediate bases `ReLULikeHull`, `SShapeHull` (and their `WithOneY` counterparts) handle piecewise-linear and S-shaped families. Use `cal_mn_constrs_with_one_y_dlp` for direct multi-neuron constraint computation with the DLP method.

### Two Usage Modes

1. **Single-neuron** — provide `input_lower_bounds` and `input_upper_bounds` with `if_cal_single_neuron_constrs=True`
2. **Multi-neuron** — provide `input_constrs` with bounds for joint constraints over multiple neurons

### Constraint Format

All constraints use H-representation: `b + A @ x >= 0`

- **Input**: `input_constrs` shape `(n, d+1)`, bounds shape `(d,)`
- **Output**: shape `(m, 2d+1)` as `[b | A_x | A_y]`

### Constructor Options

```python
hull = SigmoidHull(
    if_cal_single_neuron_constrs=False,    # include per-neuron constraints
    if_cal_multi_neuron_constrs=True,      # include multi-neuron constraints
    if_use_double_orders=False,            # doubled variable orderings (slower)
    dtype_cdd="float",                     # "float" or "fraction" for exact arithmetic
)
```

## Project Structure

```
src/wraact/
├── __init__.py          # Public API re-exports
├── _constants.py        # Numerical constants and tolerances
├── _exceptions.py       # DegeneratedError and custom exceptions
├── _functions.py        # Activation function implementations
├── _tangent_lines.py    # Tangent line computation utilities
├── acthull/             # Full hull classes (ActHull -> per-activation)
└── oney/                # Single-output WithOneY variants
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License — see [LICENSE](LICENSE)
