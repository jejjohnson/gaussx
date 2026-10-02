# Sparse Operators

`SparseOperator` is a sparse matrix whose sparsity pattern is a static,
host-side, hashable `SparsityPattern`; only the non-zero `values` are traced.
Every matrix a GMRF / INLA workflow factorises — a prior precision, a graph
Laplacian, the Laplace Hessian $Q + A^\top W A$ — has a pattern known before
any value is, so all symbolic work (canonical ordering, transposition, pattern
unions, the pattern of a congruence) runs once per pattern on the host and is
reused across `jit` calls, Newton steps, hyperparameter values and `vmap`-ped
datasets. Changing the values never retraces; changing the pattern does.

The pattern of the Laplace Hessian is known in advance:

$$
(A^\top W A)_{ij} = \sum_k A_{ki}\,w_k\,A_{kj} \neq 0 \;\Rightarrow\;
i, j \in \operatorname{supp}(A_{k,:}),\qquad
\operatorname{pattern}(Q + A^\top W A) = \operatorname{pattern}(Q)\cup
\textstyle\bigcup_k \operatorname{supp}(A_{k,:})^2 .
$$

A FEM projector row touches the vertices of one triangle, which are already
neighbours in $Q$, so the pattern does not grow; each fixed effect adds one
dense row and column.

**Storage.** Patterns are canonical: sorted row-major, duplicates merged, and
the diagonal of a square matrix always present. `symmetric=True` stores the
lower triangle only; `from_coo(..., symmetric=True)` takes each off-diagonal
pair once (edge-once graph storage) and mirrors it.

**Dispatch.**

| Primitive | Behaviour |
|---|---|
| `diag` | Exact, read from the pattern |
| `solve` | An explicit `solver` wins; otherwise CG when PSD-tagged and larger than `AutoSolver.size_threshold`, dense below |
| `logdet` | `SLQLogdet` when PSD-tagged and large, dense below |
| `diag_inv` | The existing paths (dense Cholesky for small `N`, Hutchinson otherwise) |
| `eig(rank=)` | Lanczos on the matvec |
| `cholesky` | Dense below `AutoSolver.size_threshold`; `NotImplementedError` above, until a sparse Cholesky lands |

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
Q = eqx.tree_at(lambda op: op.values, R, tau * R.values)  # same static pattern

# Observation projector A (2 observations) and Newton weights w_t
A = gx.SparseOperator.from_coo(
    np.array([0, 0, 1]), np.array([0, 1, 3]), jnp.array([0.5, 0.5, 1.0]), (2, N)
)
w_t = jnp.array([1.0, 2.0])
H = Q.union(Q.congruence(A, w_t), tags=lx.positive_semidefinite_tag)  # Q + Aᵀ diag(w_t) A
x = gx.solve(H, jnp.ones(N))
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SparseOperator, SparsityPattern]
