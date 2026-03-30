# Architecture

gaussx is organized as a **four-layer stack** that extends lineax. Each layer builds on the one below. Users can enter at any layer depending on their needs.

## Four-Layer Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3 — Recipes                                      (v0.3+) │
│  Kalman filter/smoother, ensemble covariance,                    │
│  natural ↔ expectation params, Kalman gain                       │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2 — Distributions + Sugar                        (v0.2+) │
│  MultivariateNormal, Schur complement, project,                  │
│  conditional variance, exponential family                        │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1 — Operators                                     (v0.0) │
│  Kronecker, BlockDiag, LowRankUpdate                             │
│  Extend lineax.AbstractLinearOperator                            │
├──────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives                                    (v0.0) │
│  solve, logdet, cholesky, diag, trace, sqrt, inv                 │
│  Dispatch on operator type + tags                                │
└──────────────────────────────────────────────────────────────────┘
                 │                          │
          ┌──────▼──────┐           ┌───────▼──────┐
          │   lineax    │           │   matfree    │
          │  (solvers)  │           │  (Lanczos,   │
          │             │           │   SLQ, etc.) │
          └─────────────┘           └──────────────┘
```

### Layer 0 --- Primitives

Pure functions that match the equations in papers. Every function takes an operator and returns arrays or operators:

```python
x = gaussx.solve(A, b)      # solve Ax = b
ld = gaussx.logdet(A)        # log|det(A)|
L = gaussx.cholesky(A)       # A = LL^T
d = gaussx.diag(A)           # diagonal entries
t = gaussx.trace(A)          # tr(A)
S = gaussx.sqrt(A)           # S such that SS = A
A_inv = gaussx.inv(A)        # lazy A^{-1}
```

Each primitive uses **isinstance dispatch** to select the efficient code path based on operator type:

| Operator | `solve` | `logdet` | `cholesky` |
|----------|---------|----------|------------|
| Diagonal | O(n) divide | O(n) sum(log) | O(n) sqrt |
| BlockDiag | per-block | sum of logdets | per-block |
| Kronecker | Roth's lemma | scaled sum | per-factor |
| LowRankUpdate | Woodbury | det lemma | --- |
| Dense | lineax solver | slogdet | scipy |

For large unstructured operators, Layer 0 delegates to **matfree** for iterative/stochastic algorithms (Lanczos for logdet, Hutchinson for trace).

### Layer 1 --- Operators

Structured operators extending `lineax.AbstractLinearOperator`. Each is an `equinox.Module` (immutable PyTree), supports `mv`, `as_matrix`, `transpose`, and carries structural tags.

| Operator | Represents | Efficient `mv` |
|----------|-----------|----------------|
| `Kronecker(A, B, ...)` | $A \otimes B \otimes \cdots$ | Roth's column lemma via einops |
| `BlockDiag(A, B, ...)` | $\mathrm{diag}(A, B, \ldots)$ | Per-block, concatenate |
| `LowRankUpdate(L, U, d, V)` | $L + U \mathrm{diag}(d) V^\top$ | Base mv + rank-k update |

Arithmetic (`+`, `@`, `*`) composes with lineax's built-in operators:

```python
K = gaussx.Kronecker(A, B)
perturbed = K + 0.1 * lx.IdentityLinearOperator(...)
```

### Layer 1.5 --- Solver Strategies

Pair `solve` + `logdet` into reusable strategy objects:

| Strategy | Algorithm | Best for |
|----------|-----------|----------|
| `DenseSolver` | Structural dispatch (Cholesky for PSD, etc.) | Small-medium, structured |
| `CGSolver` | Iterative CG + stochastic Lanczos logdet | Large PSD, matrix-free |

Strategies decouple the distribution from the solver --- a `MultivariateNormal` doesn't know or care whether it's doing dense Cholesky or iterative CG.

### Layer 2 --- Distributions + Sugar *(planned, v0.2+)*

- `MultivariateNormal(loc, cov_operator, solver=...)` --- accepts any covariance operator and any solver strategy
- Compound operations: `project`, `unwhiten`, `schur_complement`, `conditional_variance`
- Gaussian exponential family: natural/expectation parameters, Fisher information

### Layer 3 --- Recipes *(planned, v0.3+)*

Cross-library patterns: Kalman filter, RTS smoother, ensemble covariance, natural gradient updates. Thin wiring of Layer 0--2 operations into domain-specific sequences.

## Structural Tags

Operators carry tags that drive dispatch. gaussx extends lineax's tag set:

| Tag | Source | Used by |
|-----|--------|---------|
| `symmetric_tag` | lineax | solve, logdet |
| `positive_semidefinite_tag` | lineax | cholesky, sqrt, CG |
| `diagonal_tag` | lineax | all primitives (O(n) paths) |
| `kronecker_tag` | gaussx | all primitives (per-factor) |
| `block_diagonal_tag` | gaussx | all primitives (per-block) |
| `low_rank_tag` | gaussx | solve (Woodbury), logdet (det lemma) |

Query functions (`is_kronecker`, `is_block_diagonal`, `is_low_rank`, plus all lineax queries) let you inspect operator properties without knowing the concrete type.

## Dependencies

| Package | Role | Required |
|---------|------|----------|
| `jax` / `jaxlib` | Array backend | Yes |
| `equinox` | Module system, PyTrees | Yes |
| `lineax` | Linear operators, solvers | Yes |
| `matfree` | Krylov methods, stochastic trace | Yes |
| `jaxtyping` | Array type annotations | Yes |
| `einops` | Tensor reshaping (Principle 5) | Yes |
| `numpyro` | Distributions (Layer 2) | Planned |

## Package Layout

```
src/gaussx/
├── __init__.py              # Public API
├── _tags.py                 # Structural tags + queries
├── _operators/              # Layer 1
│   ├── _kronecker.py
│   ├── _block_diag.py
│   └── _low_rank_update.py
├── _primitives/             # Layer 0
│   ├── _solve.py
│   ├── _logdet.py
│   ├── _cholesky.py
│   ├── _diag.py
│   ├── _trace.py
│   ├── _sqrt.py
│   └── _inv.py
├── _strategies/             # Layer 1.5
│   ├── _base.py
│   ├── _dense.py
│   └── _cg.py
└── _testing.py              # Test utilities
```
