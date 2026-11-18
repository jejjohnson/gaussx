# Solves and factorisations

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: structures and primitives.

Open this when you need `A⁻¹ b`, `log|A|`, a Cholesky factor, a square root, a diagonal, a trace or eigenvalues of an operator.
Each verb takes any `lineax` operator and picks a rule from its structure, so build the operator first with [operators](operators.md).

## Names

| Name | What it is |
|---|---|
| `solve(op, b, *, solver=None)` | Solves `A x = b`. A `lineax` solver or a gaussx strategy in `solver=` overrides the default. |
| `logdet(op)` | `log|det A|`, exact for structured operators. |
| `pseudo_logdet(op, *, null_space=None, structure=None, ...)` | The log of the product of the non-zero eigenvalues of a singular PSD operator, such as a random-walk structure matrix. |
| `cholesky(op)` | An operator `L` with `A = L Lᵀ`; a `SparseOperator` gives a `SparseCholeskyFactor`. |
| `cholesky_logdet(L)` | `log|A|` from a dense factor array `L`. |
| `sqrt(op, *, lanczos_order=None)` | An operator `S` with `S S = A`, for PSD `A`. |
| `sqrt_matmul(op, rhs)`, `sqrt_inv_matmul(op, rhs)` | `A^{1/2} B` and `A^{-1/2} B` for a `(N, C)` matrix `B`, by contour quadrature with shifted solves. |
| `SumOfKroneckersSqrt` | The lazy Lanczos root that `sqrt` returns for a `SumOfKroneckers`. |
| `SumKroneckerSqrt` | Deprecated alias for `SumOfKroneckersSqrt`. |
| `inv(op, *, solver=None)` | A lazy inverse operator that applies `solve`; a `Kronecker` keeps its structure. |
| `inv_quad_logdet(op, rhs, *, strategy=None)` | The pair `tr(Rᵀ A⁻¹ R)` and `log|A|` in one call. |
| `diag(op, *, stochastic=False, ...)` | The diagonal as a 1-D array, read from the structure; `stochastic=True` estimates it from matvecs. |
| `trace(op, *, stochastic=False, algorithm=...)`, `trace_and_diag(op, ...)` | The trace, exact or by Hutchinson, Hutch++ or XTrace; the second estimates trace and diagonal from one probe pass. |
| `frobenius_norm(op, *, stochastic=False)` | `‖A‖_F` without densifying a diagonal, block-diagonal or Kronecker operator. |
| `eig(op, *, rank=None, method="lanczos")`, `eigvals(op, *, rank=None)` | Eigenpairs or eigenvalues; with `rank=` the top of the spectrum. |
| `eigh_generalized(A, B, *, rank=None, which="smallest")` | Solves `A v = λ B v` with `B`-orthonormal eigenvectors. |
| `svd(op, *, rank=None, method="lanczos")` | `(U, s, Vt)`, truncated to `rank` when given. |
| `estimate_spectral_bounds(op)` | An estimate of `(λ_min, λ_max)` of a symmetric operator. |
| `root_decomposition(op, rank=50, method="lanczos")` | A `RootDecomposition` whose tall `root` satisfies `R Rᵀ ≈ A`. |
| `root_inv_decomposition(op, rank=50, method="lanczos")` | A `RootDecomposition` with `R⁻ (R⁻)ᵀ ≈ A⁻¹`. |
| `RootDecomposition` | The result type; its `root` has shape `(N, k)`. |
| `submatrix(op, row_idx, col_idx)` | `A[row_idx, col_idx]` as a dense block without forming the whole matrix. |
| `mixture_quantile(cdf_fn, q, lower, upper)`, `Chandrupatla(rtol, atol)` | Inverts a CDF by bracketed root finding; `Chandrupatla` is the default root finder. |
| `mixture_quantile_gaussian_approx(means, stds, q)` | Quantiles of the Gaussian matched to an ensemble's mean and variance. |
| `DenseFallbackWarning` | The warning a structured verb raises when it has to build the dense matrix. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag


def rbf(n, ls=0.5):  # (n, n)
    x = jnp.linspace(0, 1, n)
    return jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / ls) ** 2) + 0.1 * jnp.eye(n)


K = gaussx.Kronecker(
    lx.MatrixLinearOperator(rbf(3), PSD), lx.MatrixLinearOperator(rbf(4), PSD)
)  # (12, 12)
dense = K.as_matrix()
b = jnp.arange(1.0, 13.0)

assert jnp.allclose(
    gaussx.solve(K, b), jnp.linalg.solve(dense, b), rtol=1e-3, atol=1e-3
)
assert jnp.allclose(gaussx.logdet(K), jnp.linalg.slogdet(dense)[1], rtol=1e-3)
L = gaussx.cholesky(K)  # a Kronecker of two Cholesky factors
assert type(L).__name__ == "Kronecker"
assert jnp.allclose((L @ L.T).as_matrix(), dense, atol=1e-3)
assert jnp.allclose(gaussx.diag(K), jnp.diag(dense), atol=1e-4)
assert jnp.allclose(gaussx.trace(K), jnp.trace(dense), rtol=1e-4)
assert jnp.allclose(gaussx.inv(K).mv(b), gaussx.solve(K, b), atol=1e-3)
top = gaussx.eigvals(K, rank=3)
assert jnp.allclose(jnp.sort(top), jnp.sort(jnp.linalg.eigvalsh(dense))[-3:], rtol=1e-3)
R = gaussx.root_decomposition(K, method="cholesky").root  # (12, 12), exact
assert jnp.allclose(R @ R.T, dense, atol=1e-3)
```

## Rules

- Pass the structured operator, not `as_matrix()`; `solve`, `logdet`, `cholesky`, `inv` and `sqrt` use the structure only when they see it.
- Tag leaf operators PSD so the dense path uses Cholesky; see [tags](tags.md).
- Use `diag`, `trace` and `submatrix` instead of building the matrix and indexing it.
- Use `inv_quad_logdet` when you need both the quadratic form and the log-determinant, as in a Gaussian likelihood.
- For a singular precision, such as a random-walk structure matrix, use `pseudo_logdet`, which takes the log of the non-zero eigenvalues only.
- Choose an iterative method with `solver=`; the strategies are in the skill `gaussx:solvers-and-preconditioners`.
- Pass `rank=` to `eig`, `eigvals` and `svd` on a large operator; `svd` without `rank=` computes the full decomposition and needs the dense matrix.

## Goes wrong

- With the default `solver=None`, a singular operator gives `nan` or `inf` from `solve`, not an error; an explicit `solver=` raises instead.
- `cholesky` and `sqrt` need a PSD operator, and `-A` for a PSD `A` raises `ValueError`.
- A `DenseFallbackWarning` means the verb materialised the operator; check the structure and the tags.
- `cholesky_logdet` takes a dense array `L`, not an operator.
- `sqrt` of a `SumOfKroneckers` is a Lanczos approximation, not an exact root.
- `diag(op, stochastic=True)` and `trace(op, stochastic=True)` return estimates, not exact values.

## Next

- Identities built from these: [identities](identities.md).
- Operators with structure to exploit: [operators](operators.md), [sparse](sparse.md), [gmrf](gmrf.md).
- Choosing CG, BBMM, preconditioners or sparse Cholesky: the skill `gaussx:solvers-and-preconditioners`.
