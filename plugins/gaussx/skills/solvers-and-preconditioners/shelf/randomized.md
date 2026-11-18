# Randomized factorisations and sketches

Role: setup, the factorisations a strategy is built from (`role-ladder`, `references/strategy.md`).
Unit: strategies.

Open this when you need a low-rank factor, a truncated SVD or eigendecomposition, or a sketch of an operator that is too big to factor densely.
The operators and the verbs live in the skill `gaussx:structured-gaussian-linalg`.

## Names

The factorisations touch the operator only through `mv`, so it may be matrix-free, and take `key=` (`None` means `PRNGKey(0)`).

| Name | What it is |
|---|---|
| `randomized_svd(op, rank, *, oversample=10, n_power_iter=2, key=None)` | Truncated SVD from a randomized QB; returns `(U, s, Vt)` with `s` descending. |
| `randomized_eigh(op, rank, *, oversample=10, n_power_iter=2, which="largest", key=None)` | Rayleigh-Ritz eigenpairs of a symmetric operator; returns `(eigenvalues, eigenvectors)` ascending. |
| `randomized_nystrom(op, rank, *, oversample=0, key=None)` | Nyström approximation `U Λ̂ Uᵀ` of a PSD operator, returned as a `LowRankUpdate`. |
| `rp_cholesky(diagonal, column, rank, *, pivoting="random", key=None)` | Randomly pivoted partial Cholesky; returns `(F, pivots)` with `F Fᵀ ≈ A`, from the diagonal and `column(s)` alone. |
| `range_finder(op, rank, *, oversample=10, n_power_iter=2, sketch=None, key=None)` | An orthonormal basis `Q` for the dominant range of `A`. |
| `qb(op, rank, *, oversample=10, n_power_iter=2, key=None)` | The factorisation `A ≈ QB` with `B = QᵀA`; returns `(Q, B)`. |
| `column_id(op, rank, ...)`, `ColumnID` | Column interpolative decomposition `A ≈ A[:, J] X`; the result holds `columns` and `interpolation`. |
| `cur(op, rank, ...)`, `CUR` | `A ≈ C U R` from actual columns and rows of `A`; the result holds `columns`, `rows`, `C`, `U`, `R`. |
| `AbstractSketch` | A random matrix `S` of shape `(d, m)`, sampled once, with `apply`, `apply_transpose`, `sketch_operator(op)` and `as_operator()`. |
| `GaussianSketch`, `OrthonormalSketch` | Dense sketches, built with `.sample(key, d, m)`; the second has orthonormal rows and needs `d <= m`. |
| `SparseSignSketch` | Sparse sketch with `nnz` signed entries per column, `.sample(key, d, m, nnz=8)`. |
| `SRHTSketch` | Subsampled randomized Hadamard transform, `.sample(key, d, m)`. |
| `RowSamplingSketch` | Weighted row sampling, `.sample(key, d, m, probabilities=None)`. |
| `hadamard_transform(x)` | The unnormalised Walsh-Hadamard transform along the last axis. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

X = jr.normal(jr.key(0), (60, 5)) @ jr.normal(jr.key(1), (5, 40))  # rank 5
U, s, Vt = gaussx.randomized_svd(lx.MatrixLinearOperator(X), 5, key=jr.key(2))
assert jnp.allclose((U * s) @ Vt, X, atol=1e-3)

t = jnp.linspace(0.0, 1.0, 100)
K = jnp.exp(
    -0.5 * ((t[:, None] - t[None, :]) / 0.2) ** 2
)  # (n, n), fast spectral decay
K_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
approx = gaussx.randomized_nystrom(K_op, 25, key=jr.key(3))  # a LowRankUpdate
assert jnp.allclose(approx.as_matrix(), K, atol=1e-2)

F, pivots = gaussx.rp_cholesky(jnp.diag(K), lambda s: K[:, s], 25, key=jr.key(4))
assert jnp.allclose(F @ F.T, K, atol=1e-2)  # K is never passed whole to the factoriser

S = gaussx.SparseSignSketch.sample(jr.key(5), d=30, m=60, nnz=4)  # (30, 60)
SX = S.sketch_operator(lx.MatrixLinearOperator(X))  # (30, 40)
Q = gaussx.range_finder(
    lx.MatrixLinearOperator(X),
    5,
    sketch=gaussx.SRHTSketch.sample(jr.key(6), d=15, m=40),
)
assert SX.shape == (30, 40) and jnp.allclose(Q @ (Q.T @ X), X, atol=1e-3)
```

## Rules

- Randomized methods find the top of the spectrum; for the small end use Lanczos or LOBPCG.
- Use `n_power_iter >= 2` for slowly decaying spectra, such as Matérn-½ Gram matrices and most geophysical fields.
- For a PSD operator `randomized_nystrom` is more accurate than `randomized_eigh` for the same number of matvecs.
- `randomized_eigh` accepts an indefinite operator; use `which="magnitude"` when large negative eigenvalues dominate.
- `rp_cholesky` needs only the diagonal and `rank` columns, so a kernel can be evaluated on the fly; its pivots double as landmark indices.
- Filter `rp_cholesky` pivots with `pivots[pivots >= 0]`: pivot `-1` marks columns past the numerical rank, which are exactly zero.
- `range_finder` takes a sketch whose `in_size` equals the operator's column count and uses `sketch.out_size` as the number of columns.
- A sketch's random draws live in the module, so `apply` and `apply_transpose` always refer to the same `S`.
- `randomized_nystrom` is what `NystromPreconditioner` and `NystromLogdet` build on, see [preconditioners](preconditioners.md) and [logdet](logdet.md).

## Goes wrong

- Randomness is drawn from `key`; leaving it as `None` reuses `PRNGKey(0)` on every call.
- `randomized_nystrom` needs a square operator, and `rank` must be at least 1.
- `OrthonormalSketch.sample` raises `ValueError` when `d > m`, and `rp_cholesky` raises `NotImplementedError` for `block_size != 1`.
- `SRHTSketch` pads the input to a power of two, which costs up to twice the memory while the transform runs.
- `column_id` and `cur` raise `ValueError` when `rank` is outside `[1, min(m, n)]`.

## Next

- Strategies that use these: [preconditioners](preconditioners.md), [logdet](logdet.md), [solvers](solvers.md).
- Sketch-based least squares, `SketchAndPrecondLSMR` and `sketch_and_solve`: [solvers](solvers.md).
- Handing a strategy down through your own library: [pass-through](pass-through.md).
