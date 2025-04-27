# WraAct: Wrapping Activation Functions

WraAct is a tool to construct the convex hull of various activation functions.

## 🛠️ Installation

You need to use 🐍 Python 3.10+ (We are using Python 3.12).  
You need the following dependencies to run **wraact**: 📦

- 📦 pycddlib (2.1.8.post1): A library for computing convex hulls and polyhedra. We use it to compute the vertices of the polytope.
- 🧮 numpy (2.2.4): A library for numerical computations in Python. We use it to handle arrays and matrices.
- ⚡ numba (0.61.2): A library for just-in-time compilation of Python code. We use it to speed up some computations.

You can install the dependencies using pip:

```bash
pip install pycddlib==2.1.8.post1 numpy==2.2.4 numba==0.61.2
```

---

# 📚 Quick Learning Function Hull

This tutorial introduces the concept of the **function hull** and the algorithm to calculate
the function hull of an activation function.
The function hull, represented by a set of **linear constraints**, provides **sound constraints**
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
\boldsymbol{C} = [\boldsymbol{b}, \boldsymbol{A}] =
\begin{bmatrix}
b_1 & a_{11} & a_{12} & \cdots & a_{1d} \\\\
b_2 & a_{21} & a_{22} & \cdots & a_{2d} \\\\
\vdots & \vdots & \vdots & \ddots & \vdots \\\\
b_m & a_{m1} & a_{m2} & \cdots & a_{md}
\end{bmatrix}
$$

Shape: **$(m, d+1)$**.  
- First column: bias terms $b$.
- Remaining columns: variable coefficients.



### 🧩 Vertices Representation (V-representation)

**Vertices** directly define the polytope.

Vertices are stored as:

$$
\boldsymbol{V} =
\begin{bmatrix}
1 & v_{11} & v_{12} & \cdots & v_{1d} \\\\
1 & v_{21} & v_{22} & \cdots & v_{2d} \\\\
\vdots & \vdots & \vdots & \ddots & \vdots \\\\
1 & v_{n1} & v_{n2} & \cdots & v_{nd}
\end{bmatrix}
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



> 📖 **See also**:  
> More theoretical details in the paper:  
> [ReLU Hull Approximation](https://dl.acm.org/doi/pdf/10.1145/3632917) (Ma et al., 2024).


---

# 🤝 Contributing

We warmly welcome contributions from everyone! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or just sharing ideas 💡—your input is appreciated!

📌 NOTE: Direct pushes to the `main` branch are restricted. Make sure to fork the repository and submit a Pull Request for any changes!
