# Sparse operators

Role: ⓪ structure (`role-ladder`, `references/structure.md`).
Unit: structures and primitives.

Open this when a precision matrix is sparse and arbitrary, such as a graph or a mesh, and a banded or Kronecker operator does not fit.
The pattern is fixed on the host and only the values are traced, so one analysis serves every `jit` call, Newton step and `vmap` batch.

## Names

| Name | What it is |
|---|---|
| `SparsityPattern(rows, cols, shape, *, symmetric=False)` | A static, hashable set of index pairs. Duplicates are merged, entries are sorted, and a square pattern always stores its diagonal. |
| `SparseOperator(values, pattern, *, tags=())` | A `lineax` operator whose only leaf is `values`, shape `(nnz,)`, in the pattern's canonical order. |
| `SparseOperator.from_coo(rows, cols, values, shape, *, symmetric=False)` | Builds from triplets. Duplicates are summed. With `symmetric=True` give each off-diagonal pair once. |
| `SparseOperator.add_diagonal(d)`, `.union(other)`, `.congruence(A, w)` | `A + diag(d)`, `A + B` on the union pattern, and `Aᵀ diag(w) A`. |
| `SymbolicCholesky` | The host-side analysis of a pattern: ordering, elimination tree and the pattern of the factor. Build it with `symbolic_cholesky`. |
| `symbolic_cholesky(pattern, *, ordering="rcm", backend="jax")` | The cached symbolic analysis. Orderings are `"rcm"`, `"natural"` and `"amd"`; backends are `"jax"` and `"cholmod"`. |
| `sparse_cholesky(op, symbolic=None)` | The numeric factorisation of a symmetric positive-definite `SparseOperator`. It `jit`s, `vmap`s and differentiates. |
| `SparseCholeskyFactor` | The factor, with `.solve(b)`, `.logdet()`, `.diag_inv()`, `.selected_inverse()` and `.solve_lower_transpose(z)`. |

## Use it

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

import gaussx

n = 8
rows = np.r_[np.arange(n), np.arange(1, n)]  # diagonal, then the lower edges once
cols = np.r_[np.arange(n), np.arange(n - 1)]
vals = jnp.r_[jnp.full(n, 3.0), -jnp.ones(n - 1)]
Q = gaussx.SparseOperator.from_coo(rows, cols, vals, (n, n), symmetric=True)
Q = Q.add_diagonal(jnp.zeros(n), tags=lx.positive_semidefinite_tag)  # (8, 8), PSD claim

sym = gaussx.symbolic_cholesky(Q.pattern)  # host, once per pattern
factor = gaussx.sparse_cholesky(Q, sym)
dense = Q.as_matrix()
b = jnp.arange(1.0, n + 1)
assert jnp.allclose(factor.solve(b), jnp.linalg.solve(dense, b), atol=1e-3)
assert jnp.allclose(factor.logdet(), jnp.linalg.slogdet(dense)[1], atol=1e-3)
assert jnp.allclose(factor.diag_inv(), jnp.diag(jnp.linalg.inv(dense)), atol=1e-3)


def logdet(scale):  # new values on the same pattern: no retrace, no new analysis
    Qs = eqx.tree_at(lambda op: op.values, Q, scale * Q.values)
    return gaussx.sparse_cholesky(Qs, sym).logdet()


assert jnp.isfinite(jax.grad(logdet)(1.0))
```

## Rules

- Build the pattern on the host from concrete integer arrays; only `values` may be traced.
- With `symmetric=True` store the lower triangle and give each edge once; giving both `(i, j)` and `(j, i)` sums them.
- Call `symbolic_cholesky` once and pass the result to every `sparse_cholesky` on that pattern.
- Change values with `eqx.tree_at(lambda op: op.values, Q, new_values)`; changing the pattern retraces.
- `add_diagonal` keeps only the symmetric tag unless you pass `tags=`, because an arbitrary `d` can break definiteness.
- `gaussx.cholesky` on a `SparseOperator` returns a `SparseCholeskyFactor`; for solves and log-determinants through it, pass `SparseCholeskySolver` as described in the skill `gaussx:solvers-and-preconditioners`.
- The `"amd"` ordering and the `"cholmod"` backend need `scikit-sparse`.

## Goes wrong

- `as_matrix()` on a large `SparseOperator` builds the dense array; `diag`, `mv` and `to_bcoo` do not.
- A non-positive-definite operator gives `NaN` from `logdet`, not an error.
- A `symbolic` computed for another pattern raises `ValueError` in `sparse_cholesky`.
- `selected_inverse()` holds entries of `Q⁻¹` on the pattern of `L + Lᵀ`; it is not equal to `Q⁻¹` as an operator.

## Next

- A banded precision of a time series is cheaper as a `BlockTriDiag`: [operators](operators.md).
- Ready-made sparse precisions for fields and areal data: [gmrf](gmrf.md).
- The verbs on these operators: [solves](solves.md), [identities](identities.md).
- Tagging the operator so dispatch can use it: [tags](tags.md).
