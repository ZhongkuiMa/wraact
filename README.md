# WraAct: Precise Activation Function Over-Approximation for Neural Network Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy 2.2](https://img.shields.io/badge/NumPy-2.2-green.svg)](https://numpy.org/)
[![Numba 0.61](https://img.shields.io/badge/Numba-0.61-orange.svg)](https://numba.pydata.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**WraAct** constructs tight convex hull approximations of activation functions for sound and efficient neural network verification.

> 📖 **See also**:  
> This repo is based on the following papers and provides implementations for the algorithms described therein. **This is a regularly maintained and updated repo for the algorithm part.**
>
> [ReLU Hull Approximation](https://dl.acm.org/doi/pdf/10.1145/3632917) (POPL'24) (Ma et al., 2024).
>
> [Convex Hull Approximation for Activation Functions]() (OOPSLA'25) (Ma et al., 2025)

## ✨ Key Features

- 🔍 **Precise Approximations** - Generates mathematically sound convex hull constraints for various activation functions
- 🚀 **Performance Optimized** - Uses Numba JIT compilation for fast constraint generation
- 🧮 **Multiple Activation Types** - Supports ReLU, Sigmoid, Tanh, LeakyReLU, ELU, and many more
- 🔄 **V-H Representation** - Efficiently converts between vertex and halfspace representations with pycddlib
- 🌐 **Multi-Dimensional Support** - Handles both unary and multi-variable activation functions

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ZhongkuiMa/rover_alpha.git
cd rover_alpha/wraact

# Install dependencies
pip install pycddlib==2.1.8.post1 numpy==2.2.4 numba==0.61.2
```

---

# 📚 Quick Learning Function Hull

This tutorial introduces the concept of the **function hull** and the algorithm to calculate the function hull of an activation function. The function hull, represented by a set of **linear constraints**, provides **sound constraints**
for neural network verification.

## ✨ Two Representations of Polytope

A **polytope** (high-dimensional polyhedron) can be defined by:

- A set of **halfspaces** (linear constraints), called **H-representation**.
- A set of **vertices**, called **V-representation**.

> ℹ️ **Note**:  
> These are basic concepts in *computational geometry*. See any computational geometry textbook for more.

> 💡 **Tip**:  
> Here, we only discuss **bounded convex polytopes** (no unbounded ones).  
> Formal definitions are available in computational geometry literature.

### 🧩 Halfspace Representation (H-representation)

A **halfspace** is the set of points satisfying a linear inequality:

$$
\boldsymbol{b} + \boldsymbol{A}x \geq 0,
$$

where:

- $\boldsymbol{b} \in \mathbb{R}^m$ is the bias term.
- $\boldsymbol{A} \in \mathbb{R}^{m \times d}$ is the matrix of coefficients.

All constraints are stored together as:

$$
\boldsymbol{C} = [\boldsymbol{b}, \boldsymbol{A}] = \begin{bmatrix} b_1 & a_{11} & a_{12} & \cdots & a_{1d} \\\\ b_2 & a_{21} & a_{22} & \cdots & a_{2d} \\\\ \vdots & \vdots & \vdots & \ddots & \vdots \\\\ b_m & a_{m1} & a_{m2} & \cdots & a_{md} \end{bmatrix}
$$

Shape: **$(m, d+1)$**.

- First column: bias terms $b$.
- Remaining columns: variable coefficients.

### 🧩 Vertices Representation (V-representation)

**Vertices** directly define the polytope.

Vertices are stored as:

$$
\boldsymbol{V} = \begin{bmatrix} 1 & v_{11} & v_{12} & \cdots & v_{1d} \\\\ 1 & v_{21} & v_{22} & \cdots & v_{2d} \\\\ \vdots & \vdots & \vdots & \ddots & \vdots \\\\ 1 & v_{n1} & v_{n2} & \cdots & v_{nd} \end{bmatrix}
$$

Shape: **$(n, d+1)$**.

- First column: all ones (to indicate vertices).
- Rest: vertex coordinates.

> 🔗 **Reference**:  
> We use the same format as [pycddlib](https://pycddlib.readthedocs.io/).

## 🚀 Activation Function Taxonomy

Activation functions are **non-linear functions** applied to neuron outputs.  
We classify them by:

1. **Input dimension**: Unary vs Multi-variable.
2. **Shape**: ReLU-like vs S-shaped.

### 1️⃣ Unary and Multi-Variable Activation Functions

- **Unary Activation Functions**:
    - Form: $f: \mathbb{R} \rightarrow \mathbb{R}$ (scalar to scalar).
    - Examples: ReLU, LeakyReLU, ELU, SiLU, Sigmoid, Tanh.
    - Function hull calculated for **multiple neurons**.

$$
\mathcal{M} \supseteq \{(\boldsymbol{x}, f(\boldsymbol{x})) \mid \boldsymbol{x} \in \mathcal{X}\}.
$$

- **Multi-Variable Activation Functions**:
    - Form: $f: \mathbb{R}^n \rightarrow \mathbb{R}$ (vector to scalar).
    - Examples: MaxPool.
    - Function hull calculated for **a single neuron**.

$$
\mathcal{M} \supseteq \{(\boldsymbol{x}, f(\boldsymbol{x})) \mid \boldsymbol{x} \in \mathcal{X}\}.
$$

> 💬 **Note**:  
> Strictly speaking, we are computing **function hull over-approximations**, but we simply call them **function hulls**.

### 2️⃣ ReLU-like and S-shaped Activation Functions

- **ReLU-like Activation Functions**:
    - Examples: ReLU, LeakyReLU, ELU.
    - Construct a **DLP (double linear piece)** upper bound.
    - Behave like:
        - In negative region: almost constant (e.g., 0).
        - In positive region: close to identity ($y = x$).

- **S-shaped Activation Functions**:
    - Examples: Sigmoid, Tanh.
    - Construct **two DLPs** (upper and lower bounds).
    - Behave like:
        - Negative region: close to constant (e.g., 0).
        - Positive region: close to constant (e.g., 1).
        - Monotonically increasing or decreasing.

> 💡 **Tip**:  
> No strict mathematical definition for "ReLU-like" or "S-shaped"; it's based on behavior.

## 🧠 What is Function Hull?

The **Function Hull** is a **polytope** in the input-output space that **encloses** the graph of an activation function over a given input domain.

We only consider **bounded convex polytopes** and focus on their **H-representation**.

## 🛠️ Algorithm Overview

The goal is to construct a polytope that **wraps** the graph of the activation function.

### 📥 Input

- $\boldsymbol{C} \in \mathbb{R}^{m \times (n+1)}$: Input polytope constraints.
- $\boldsymbol{l}, \boldsymbol{u} \in \mathbb{R}^n$: (Optional) Lower and upper bounds of input variables.

### 📤 Output

- Constraints defining the **function hull** of the activation function.

### ⚙️ Core Computation Steps

- Extend output dimension one by one.
- Construct convex/concave **piece-wise linear bounds**.
- Use **DLP (double linear piece)** functions where needed.

#### Details:

- Input constraints:

$$
\boldsymbol{b} + \boldsymbol{A}x \geq 0
$$

- Constructed linear pieces:

$$
y - \boldsymbol{B} \boldsymbol{x} = 0
$$

- Compute the quotient:

$$
\beta_{ij} = \frac{b_i + \boldsymbol{A}_i \boldsymbol{x}}{y - \boldsymbol{B}_j \boldsymbol{x}}
$$

> 🔍 **Efficient Calculation**:  
> Enumerate all vertices — efficient for low dimensions (2–10).

- Then take:

$$
\beta_i = \max_j (\beta_{ij})
$$

- And add constraints depending on convexity:

$$
b_i + \boldsymbol{A}_i \boldsymbol{x} \geq \beta_i (y - \boldsymbol{B}_j \boldsymbol{x})
$$

or

$$
b_i + \boldsymbol{A}_i \boldsymbol{x} \leq \beta_i (y - \boldsymbol{B}_j \boldsymbol{x})
$$

---

# 🤝 Contributing

We warmly welcome contributions from everyone! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or just sharing ideas 💡—your input is appreciated!

📌 NOTE: Direct pushes to the `main` branch are restricted. Make sure to fork the repository and submit a Pull Request for any changes!
