# Sparse Operators

`SparseOperator` is a sparse matrix whose sparsity pattern is a static,
host-side, hashable `SparsityPattern`; only the non-zero `values` are traced.
Every matrix a GMRF / INLA workflow factorises — a prior precision, a graph
Laplacian, the Laplace Hessian $\boldsymbol{\Lambda} + \mathbf{A}^\top \mathbf{W} \mathbf{A}$ — has a pattern known before
any value is, so all symbolic work (canonical ordering, transposition, pattern
unions, the pattern of a congruence) runs once per pattern on the host and is
reused across `jit` calls, Newton steps, hyperparameter values and `vmap`-ped
datasets. Changing the values never retraces; changing the pattern does.

The pattern of the Laplace Hessian is known in advance:

$$
[\mathbf{A}^\top \mathbf{W} \mathbf{A}]_{ij} = \sum_k [\mathbf{A}]_{ki} w_k [\mathbf{A}]_{kj} \neq 0 \Rightarrow i, j \in \mathrm{supp}([\mathbf{A}]_{k,:})
$$

$$
\mathrm{pattern}(\boldsymbol{\Lambda} + \mathbf{A}^\top \mathbf{W} \mathbf{A}) = \mathrm{pattern}(\boldsymbol{\Lambda}) \cup \bigcup_k \mathrm{supp}([\mathbf{A}]_{k,:})^2
$$

A FEM projector row touches the vertices of one triangle, which are already
neighbours in $\boldsymbol{\Lambda}$, so the pattern does not grow; each fixed effect adds one
dense row and column.

**Storage.** Patterns are canonical: sorted row-major, duplicates merged, and
the diagonal of a square matrix always present. `symmetric=True` stores the
lower triangle only; `from_coo(..., symmetric=True)` takes each off-diagonal
pair once (edge-once graph storage) and mirrors it.

**Dispatch.**

| Primitive | Behaviour |
|---|---|
| `diag` | Exact, read from the pattern |
| `solve` | `SparseCholeskySolver` when the caller passes it (strategies, `dispatch_solve`); an explicit lineax `solver` wins; otherwise CG when PSD-tagged and larger than `AutoSolver.size_threshold`, dense below |
| `logdet` | `SparseCholeskySolver` when passed; otherwise `SLQLogdet` when PSD-tagged and large, dense below |
| `diag_inv` | Takahashi through the sparse factor with `solver=SparseCholeskySolver(...)` (any size) or `method="cholesky"`; `"auto"` uses it for `N ≤ 2048` and Hutchinson above |
| `eig(rank=)` | Lanczos on the matvec |
| `cholesky` | A `SparseCholeskyFactor` (RCM ordering, cached symbolic analysis) |

There is no size heuristic for the exact path: the caller knows `N` and
chooses `SparseCholeskySolver`; `AutoSolver` keeps CG for large PSD
operators.

`JacobiPreconditioner` works through the exact diagonal, which is usually
enough for a graph Laplacian plus a diagonal shift.

**Example.** An ICAR structure matrix from an edge list, then the Laplace
Hessian rebuilt on its precomputed pattern at each Newton step:

```python
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np

import gaussx as gx

# Path graph 0 - 1 - 2 - 3, each edge once, host-side indices
N = 4
senders, receivers = np.array([1, 2, 3]), np.array([0, 1, 2])
w = np.ones(3)
deg = np.bincount(senders, w, N) + np.bincount(receivers, w, N)
R = gx.SparseOperator.from_coo(
    np.r_[np.arange(N), senders],
    np.r_[np.arange(N), receivers],
    jnp.asarray(np.r_[deg, -w]),
    (N, N),
    symmetric=True,
    tags=frozenset({lx.positive_semidefinite_tag}),
)
tau = 2.0
precision = eqx.tree_at(lambda op: op.values, R, tau * R.values)  # same static pattern

# Observation projector A (2 observations) and Newton weights w_t
A = gx.SparseOperator.from_coo(
    np.array([0, 0, 1]), np.array([0, 1, 3]), jnp.array([0.5, 0.5, 1.0]), (2, N)
)
w_t = jnp.array([1.0, 2.0])
H = precision.union(
    precision.congruence(A, w_t), tags=lx.positive_semidefinite_tag
)  # Λ + Aᵀ diag(w_t) A
x = gx.solve(H, jnp.ones(N))
```

## Sparse Cholesky

Cholesky is Gaussian elimination: eliminating node $j$ connects its
not-yet-eliminated neighbours, so the column patterns of $\mathbf{P} \mathbf{A} \mathbf{P}^\top = \mathbf{L} \mathbf{L}^\top$
follow the **elimination tree**,

$$
\mathrm{struct}([\mathbf{L}]_{:,j}) = \mathrm{struct}([\mathbf{P} \mathbf{A} \mathbf{P}^\top]_{j:,j}) \cup \bigcup_{\mathrm{parent}(c) = j} \mathrm{struct}([\mathbf{L}]_{:,c}) \setminus \{c\}
$$

$$
\mathrm{parent}(j) = \min\{i > j : [\mathbf{L}]_{ij} \neq 0\}
$$

That depends only on the pattern and the ordering, so `symbolic_cholesky`
runs once on the host (NumPy / SciPy) and is cached per `SparsityPattern`.
`sparse_cholesky` then traces only the values: it `jit`s, `vmap`s over
values and is differentiable.

- **Ordering.** `"rcm"` (reverse Cuthill–McKee, the default) minimises the
  bandwidth; `"natural"` keeps the given order; `"amd"` (approximate minimum
  degree, through CHOLMOD) minimises fill.
- **Numeric phase.** After RCM on a mesh the factor fills a narrow band, so
  $\mathbf{L}$ is block tridiagonal in bandwidth-sized blocks and dense block kernels
  do the work. Otherwise a left-looking `lax.scan` over columns gathers
  fixed-size windows from the CSC arrays (columns bucketed by length, so a
  few long separator columns do not pad all the short ones). Either way
  `SparseCholeskyFactor.values` holds $\mathbf{L}$ on its exact CSC pattern.
- **Takahashi.** The backward recursion
  $[\mathbf{Z}]_{ij} = \delta_{ij} / [\mathbf{L}]_{jj}^2 - [\mathbf{L}]_{jj}^{-1} \sum_{k > j, k \in \mathrm{struct}([\mathbf{L}]_{:,j})} [\mathbf{L}]_{kj} [\mathbf{Z}]_{ki}$
  evaluates $\mathbf{Z} = (\mathbf{P} \mathbf{A} \mathbf{P}^\top)^{-1}$ exactly on $\mathrm{pattern}(\mathbf{L} + \mathbf{L}^\top)$,
  which contains $\mathrm{pattern}(\mathbf{P} \mathbf{A} \mathbf{P}^\top)$, at about the cost of the
  factorisation. `selected_inverse()` returns it as a `SparseOperator` in
  the original order; `diag_inv()` gives the marginal variances.
- **Gradients.** $\mathrm{d} \log \det \mathbf{A} = \mathrm{tr}(\mathbf{A}^{-1} \mathrm{d}\mathbf{A})$, so the log-determinant's cotangent is $\mathbf{Z}$ on $\mathrm{pattern}(\mathbf{A})$: one Takahashi sweep, never $\mathbf{A}^{-1}$. For $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$: $\bar{\mathbf{b}} = \mathbf{A}^{-1}\bar{\mathbf{x}}$ and $\bar{\mathbf{A}} = -\bar{\mathbf{b}} \mathbf{x}^\top$, symmetrised. With `symmetric=True` storage an off-diagonal stored value sets $[\mathbf{A}]_{ij}$ and $[\mathbf{A}]_{ji}$, so its gradient is doubled ($2 [\mathbf{Z}]_{ij}$); general storage is factored as $\frac{1}{2}(\mathbf{A} + \mathbf{A}^\top)$ and each stored value gets $[\mathbf{Z}]_{ij}$.
  `solve_lower_transpose` (sampling, $\mathbf{x} = \mathbf{P}^\top \mathbf{L}^{-\top} \mathbf{z}$) and
  `diag_inv` are differentiated by JAX through the factorisation.
- **CHOLMOD backend** (opt-in, `pip install scikit-sparse`, which needs
  SuiteSparse; not a dependency of gaussx). `backend="cholmod"` runs only
  the numeric factorisation in CHOLMOD through `jax.pure_callback`, on the
  same symbolic pattern; the solves, Takahashi and the gradients are the
  same JAX code, so both backends give identical gradients. CPU only, and
  `vmap` calls CHOLMOD once per batch element.

**Scale.** Measured on triangulated square meshes (the P1 FEM 7-point
stencil), CPU, float64, `jit`-compiled, after compilation:

| Nodes | Ordering / backend | nnz(L) | Fill vs tril(Λ) | `logdet` | `value_and_grad(logdet)` | `diag_inv` |
|---|---|---|---|---|---|---|
| 10,000 | RCM / JAX (banded) | 681,550 | 17.2 | 1.0 s | 1.1 s | 0.9 s |
| 40,000 | RCM / JAX (banded) | 5,393,100 | 33.9 | 1.1 s | 1.6 s | 1.5 s |
| 99,856 | RCM / JAX (banded) | 21,185,746 | 53.2 | 3.3 s | 5.7 s | 11 s |
| 10,000 | AMD / JAX (windows) | 295,884 | 7.5 | 1.7 s | 3.4 s | 3.2 s |
| 10,000 | AMD / CHOLMOD | 295,884 | 7.5 | 0.07 s | 1.7 s | 1.6 s |
| 40,000 | AMD / CHOLMOD | 1,569,916 | 9.9 | 0.5 s | 15 s | 17 s |
| 99,856 | AMD / CHOLMOD | 4,748,667 | 11.9 | 1.8 s | ~2 min | ~2 min |

Timings are from a shared 16-core machine and are indicative only. The
symbolic analysis is a one-off host cost (about 0.3 s at 10⁴ nodes and
8 s at 10⁵ with RCM).

- 2-D meshes up to about $10^5$ nodes: RCM with the JAX backend.
- Beyond that, or when the fill of RCM is prohibitive: AMD through
  CHOLMOD (much less fill; its `logdet` is fast, but gradients and
  marginal variances run the JAX Takahashi on the AMD pattern, which
  gathers windows and is the slower path).
- Beyond that: the iterative backend (CG, SLQ, Hutchinson).

Supernodes and level scheduling (GPU parallelism over independent
subtrees) are follow-ups.

**Example.** A Matérn-like SPDE precision $\boldsymbol{\Lambda}(\kappa) = \kappa^2 \mathbf{I} + \mathbf{G}$ on a
40k-node mesh: analyse once, factor for many $\kappa$, differentiate the
log-determinant.

```python
import jax
import jax.numpy as jnp
import numpy as np

import gaussx as gx

# Stiffness-like graph Laplacian G of a triangulated 200 x 200 square
side = 200
n = side * side
i = np.arange(n)
right = i[i % side != side - 1]  # nodes with a right neighbour
down = i[i < n - side]  # nodes with a neighbour below
diagonal = right[right < n - side]  # one diagonal per cell
senders = np.r_[right + 1, down + side, diagonal + side + 1]
receivers = np.r_[right, down, diagonal]
w = np.ones(senders.size)
deg = np.bincount(senders, w, n) + np.bincount(receivers, w, n)
G = gx.SparseOperator.from_coo(
    np.r_[np.arange(n), senders],
    np.r_[np.arange(n), receivers],
    jnp.asarray(np.r_[deg, -w]),
    (n, n),
    symmetric=True,
)
sym = gx.symbolic_cholesky(G.pattern)  # host, once: RCM, banded layout


def logdet(log_kappa):
    precision = G.add_diagonal(jnp.full(n, jnp.exp(2 * log_kappa)))  # same pattern
    return gx.sparse_cholesky(precision, sym).logdet()


jax.vmap(logdet)(jnp.linspace(-1.0, 1.0, 16))  # 16 factorisations, one analysis
jax.grad(logdet)(0.0)  # d log det Λ / d log κ, through one Takahashi sweep
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SparseOperator, SparsityPattern, symbolic_cholesky, SymbolicCholesky, sparse_cholesky, SparseCholeskyFactor]
