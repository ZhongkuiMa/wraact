# WRAACT: Precise Activation Function Over-Approximation for Neural Network Verification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/wraact/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/wraact/actions/workflows/unit-tests.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Tests](https://img.shields.io/badge/tests-1876%20passed-success)](https://github.com/ZhongkuiMa/wraact/actions/workflows/unit-tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](https://github.com/ZhongkuiMa/wraact)
[![Version](https://img.shields.io/github/v/tag/ZhongkuiMa/wraact?sort=semver)](https://github.com/ZhongkuiMa/wraact/releases)

Compute tight convex hulls for neural network activation functions (ReLU, Sigmoid, Tanh, ELU, LeakyReLU, MaxPool) with sound over-approximation guarantees for abstract interpretation during verification.

## Installation

```bash
git clone https://github.com/ZhongkuiMa/wraact.git
cd wraact
pip install -e ".[dev]"  # or `pip install -e .` for minimal install
```

**Requirements:** Python 3.11+, pycddlib, numpy, numba

## Quick Start

```python
import numpy as np
from wraact import SigmoidHull

# Create hull calculator
hull = SigmoidHull()

# Define a triangular input region in 2D using H-representation
# Each row: [b, a1, a2] represents b + a1*x1 + a2*x2 >= 0
input_constrs = np.array([
    [0.0,  1.0,  0.0],   # x1 >= 0
    [0.0,  0.0,  1.0],   # x2 >= 0
    [1.0, -1.0, -1.0],   # x1 + x2 <= 1
])
lb = np.array([0.0, 0.0])
ub = np.array([1.0, 1.0])

# Compute convex hull constraints for sigmoid over this region
output_constrs = hull.cal_hull(
    input_constrs=input_constrs,
    input_lower_bounds=lb,
    input_upper_bounds=ub
)
# Output shape: (n, 5) as [b, a_x1, a_x2, a_y1, a_y2]
# Each row: b + a_x1*x1 + a_x2*x2 + a_y1*y1 + a_y2*y2 >= 0
```

## Usage Guide

### Available Hull Classes

| Class | Description |
|-------|-------------|
| `ReLUHull` | ReLU: max(0, x) |
| `SigmoidHull` | Sigmoid: 1/(1+exp(-x)) |
| `TanhHull` | Hyperbolic tangent |
| `ELUHull` | Exponential Linear Unit |
| `LeakyReLUHull` | Leaky ReLU with configurable slope |
| `MaxPoolHull` | Max pooling operation |
| `*WithOneY` variants | Faster single-output approximations |

### Input/Output Format

All constraints use **H-representation**: `b + A @ x >= 0`

- **Input**: `input_constrs` shape `(n, d+1)`, bounds shape `(d,)`
- **Output**: shape `(m, 2d+1)` as `[b | A_x | A_y]`

## Code Quality

```bash
pytest tests/ -v          # 1876 tests, 92% coverage
ruff check src/wraact     # Linting
mypy src/wraact           # Type checking
```

## Project Structure

```
wraact/
├── src/wraact/
│   ├── __init__.py          # Package exports
│   ├── _constants.py        # Numerical constants
│   ├── _exceptions.py       # Custom exceptions
│   ├── _functions.py        # Activation functions
│   ├── _tangent_lines.py    # Tangent line utilities
│   ├── acthull/             # Hull implementations
│   └── oney/                # Single-output optimized variants
├── tests/                   # Test suite
└── pyproject.toml           # Package configuration
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing procedures, code quality standards, and pull request guidelines.

## License

MIT License - see [LICENSE](LICENSE)
