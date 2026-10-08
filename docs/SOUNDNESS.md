# Soundness Gates and Conservative Fallbacks

WRAACT returns constraints in the form `b + A @ z >= 0`. A returned hull is
sound only if every concrete graph point in the input domain satisfies every
constraint. A high pass percentage is not enough: one reproducible concrete
violation is a correctness bug.

## Required gates

- Deterministic graph-containment tests require a minimum margin of `-1e-8`.
- The public-surface matrix covers full-output, single-neuron, double-order,
  `WithOneY`, exact MaxPool, and DLP-named MaxPool APIs on asymmetric domains.
- Shape and finiteness checks are useful diagnostics, but they are not
  substitutes for graph containment.
- Sampled containment is a regression oracle, not a mathematical proof. New
  geometry still needs a derivation or proof before it replaces a conservative
  fallback.

Run the focused gate with:

```bash
pytest -q tests/test_units/test_soundness/test_public_hull_containment.py
```

## Current conservative choices

- Double-order evaluation permutes constraints, vertices, bounds, input
  coefficients, and output coefficients as one parent state.
- Sigmoid and Tanh single-neuron constraints use monotonic interval bounds.
  Saturated numerical regimes fall back to the known activation codomain.
- ELU's two-piece upper approximation joins the left and right interval chords
  at the concrete graph point `(m, ELU(m))`; derivative values are slopes, not
  admissible output coordinates.
- `MaxPoolHullDLP` and `MaxPoolHullDLPWithOneY` are compatibility names that
  currently delegate to the exact MaxPool hull. The former group-sum DLP is not
  valid for arbitrary signed inputs.
- MaxPool semantic shortcuts require exact vertex dominance; approximate
  equality must not change which coordinate computes the maximum.

These fallbacks may be looser or slower. Tightness and performance improvements
are welcome only when the strict containment gate remains green and the new
construction has a defensible soundness argument.
