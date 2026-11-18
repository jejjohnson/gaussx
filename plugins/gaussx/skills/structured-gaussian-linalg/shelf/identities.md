# Linear-algebra identities

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: structures and primitives.

Open this when a formula you are about to code is a known identity: a Woodbury solve, a Schur complement, a Lyapunov equation, a tridiagonal solve, a marginal variance of a precision, or a covariance pushed through a linear map.
These functions use a cheaper route than the dense formula; the general verbs are in [solves](solves.md).

## Names

| Name | What it is |
|---|---|
| `woodbury_solve(base, U, D, b)` | `(L + U diag(D) Uᵀ)⁻¹ b` from the parts, without building a `LowRankUpdate`. |
| `schur_complement(K_XX, K_XZ, K_ZZ)` | `K_XX − K_XZ K_ZZ⁻¹ K_ZX`, returned as a `LowRankUpdate`. |
| `conditional_variance(K_XX_diag, K_XZ, A_X, S_u=None)` | The diagonal of the Schur complement, plus `diag(A_X S_u A_Xᵀ)` when `S_u` is given. Negative base variances are clamped to zero. |
| `diag_conditional_variance(K_XX_diag, K_XZ, A_X)` | `diag(K_XX − A_X K_ZX)`, the same without the variational term. |
| `diag_inv(op, *, method="auto", ...)` | `diag(A⁻¹)`, the marginal variances of a precision, without forming the inverse. Structured operators take an exact path. |
| `selected_inverse(op)` | The band of `Q⁻¹` for a symmetric positive-definite `BlockTriDiag`, as a `BlockTriDiag`. |
| `discrete_lyapunov_solve(G, Q)` | Solves `P − G P Gᵀ = Q` through the eigendecomposition of `G`. |
| `EigenDecomposition.from_matrix(A, symmetric=None)` | A precomputed `A = V diag(λ) V⁻¹`; `.solve_shifted(b, shift)` solves `(A − σI) x = b` for any shift. |
| `kronecker_sum_solve(factors, rhs, shift=0.0)` | Solves `(A₀ ⊕ A₁ ⊕ … − σI) vec(X) = vec(R)` on a tensor `R`, given one `EigenDecomposition` per axis. |
| `tridiagonal_solve(lower, diag, upper, rhs)` | Solves a tridiagonal system from its three bands. |
| `tridiagonal_solve_batched(lower, diag, upper, rhs)` | The same over leading batch dimensions. |
| `solve_matrix(op, B)` | `A⁻¹ B` for a matrix `B`, with one Cholesky factorisation when `A` is plain PSD. |
| `solve_columns(op, B)`, `solve_rows(op, B)` | `A⁻¹ B` column by column, and `A x = bᵢ` for each row `bᵢ` of a matrix, each with a structured solve per vector. |
| `cov_transform(J, cov)`, `sandwich(A, P)` | `J Σ Jᵀ` and `A P Aᵀ`, keeping matched `Kronecker` or `BlockDiag` structure. |
| `trace_product(A, B)` | `tr(A B)` without forming the product. |
| `safe_cholesky(op, *, initial_jitter=1e-8, max_jitter=1e-2, ...)` | A dense Cholesky factor that retries with growing diagonal jitter when the first attempt gives `NaN`. |
| `symmetrize(M)` | `½(M + Mᵀ)` on the last two axes. |
| `stable_squared_distances(X, Z, *, compute_dtype=..., accumulate_dtype=None)` | Squared Euclidean distances that are never negative, for building kernels. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

n, k = 12, 2
U = jr.normal(jr.key(0), (n, k))  # (n, k)
D = jnp.array([1.0, 2.0])
base = lx.DiagonalLinearOperator(jnp.full(n, 2.0))
b = jnp.arange(1.0, n + 1)
dense = base.as_matrix() + U @ jnp.diag(D) @ U.T
assert jnp.allclose(
    gaussx.woodbury_solve(base, U, D, b), jnp.linalg.solve(dense, b), atol=1e-3
)

Q = gaussx.ar1_precision(8, rho=0.6, tau=2.0)  # BlockTriDiag, (8, 8)
band = gaussx.selected_inverse(Q)  # only the band of Q⁻¹
Sigma = jnp.linalg.inv(Q.as_matrix())
assert jnp.allclose(band.diagonal[:, 0, 0], jnp.diag(Sigma), atol=1e-3)
assert jnp.allclose(gaussx.diag_inv(Q), jnp.diag(Sigma), atol=1e-3)

G = jnp.array([[0.5, 0.1], [0.0, 0.3]])  # stable: spectral radius below one
P = gaussx.discrete_lyapunov_solve(G, jnp.eye(2))
assert jnp.allclose(P - G @ P @ G.T, jnp.eye(2), atol=1e-4)

A0, A1 = (
    jnp.array([[2.0, 0.5], [0.5, 1.0]]),
    jnp.array([[3.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 1.0]]),
)
facs = [gaussx.EigenDecomposition.from_matrix(A, symmetric=True) for A in (A0, A1)]
R = jr.normal(jr.key(1), (2, 3))
X = gaussx.kronecker_sum_solve(facs, R, shift=0.5)  # (2, 3), no (6, 6) matrix formed
dense_ks = jnp.kron(A0, jnp.eye(3)) + jnp.kron(jnp.eye(2), A1) - 0.5 * jnp.eye(6)
assert jnp.allclose(dense_ks @ X.ravel(), R.ravel(), atol=1e-3)
```

## Rules

- Use `woodbury_solve` when you hold `base`, `U` and `D` separately; if you keep the operator around, build a `LowRankUpdate` and call `solve`.
- Take marginal variances of a precision with `diag_inv`, never `jnp.diag(jnp.linalg.inv(...))`; for a `BlockTriDiag` it reads the diagonal blocks of `selected_inverse`.
- `selected_inverse` holds only the diagonal and first off-diagonal blocks of `Q⁻¹`; the first are the marginal variances and the second the lag-one covariances.
- Build an `EigenDecomposition` once, outside `jit`, then reuse it for any number of shifts; `shift` may be traced.
- Prefer `solve_matrix` over `solve_columns` for a plain PSD operator, and `solve_columns` for a `Kronecker`, `BlockDiag` or `LowRankUpdate`.
- Call `symmetrize` after a covariance update to remove rounding asymmetry.
- Use `safe_cholesky` only when a matrix is near-singular by rounding; it adds jitter and returns a dense array.

## Goes wrong

- `discrete_lyapunov_solve` returns `NaN` or `inf` for a defective `G`, and has no unique solution when some `λ_i λ_j = 1`.
- `EigenDecomposition.from_matrix` needs concrete values and rejects matrices with complex eigenvalues.
- `selected_inverse` raises `TypeError` for anything but a `BlockTriDiag`, and `ValueError` if it is not symmetric.
- `schur_complement` returns a `LowRankUpdate`, not a dense matrix; call `.as_matrix()` only when you need one.
- The legacy three-argument form of `conditional_variance` is deprecated and does not subtract the Schur term.

## Next

- The general verbs these specialise: [solves](solves.md).
- Operators these identities apply to: [operators](operators.md), [gmrf](gmrf.md).
- Tags that route a solve to a cheaper path: [tags](tags.md).
