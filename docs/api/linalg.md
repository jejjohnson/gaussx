# Linear-Algebra Utilities

Numerically careful building blocks shared by the higher layers: robust
factorizations, classical matrix identities in operator form, and batched /
matrix-RHS solve helpers.

## Robust factorization & hygiene

`safe_cholesky` retries with geometrically growing diagonal jitter inside a
statically-bounded `jax.lax.fori_loop` (JIT-compatible and reverse-mode
differentiable) when a matrix is not numerically positive-definite;
`symmetrize` removes the floating-point asymmetry that accumulates in
covariance updates.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [safe_cholesky, symmetrize]

## Matrix identities

Woodbury, Schur complements, and conditional (Schur-complement) variances —
the identities behind every Gaussian conditioning step, exposed directly so
recipes never re-derive them.

!!! warning "Legacy `conditional_variance` signature"
    The pre-#152 three-positional form
    `conditional_variance(base_diag, A_X, S_u)` is still accepted until
    gaussx 0.7.0 — it is
    detected when the second positional argument is a
    `lineax.AbstractLinearOperator` — but it emits a `DeprecationWarning` and
    skips the $K_{XZ}$-based Schur subtraction, treating its first argument as
    an already-computed Schur diagonal. Call
    `conditional_variance(K_XX_diag, K_XZ, A_X, S_u=S_u)` instead.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [woodbury_solve, schur_complement, conditional_variance, diag_conditional_variance, cov_transform, sandwich, trace_product]

## Marginal variances & selected inverses

`diag_inv` returns $\operatorname{diag}(Q^{-1})$ — the marginal variances
of a Gaussian with precision $Q$ — and dispatches on structure before
falling back to dense Cholesky or Hutchinson:

| Structure | Path | Cost |
|---|---|---|
| `BlockTriDiag` (rw1, rw2, ar1, temporal SDE priors) | block Takahashi recursion, `selected_inverse` | $O(N d^3)$ |
| `Kronecker` $A \otimes B$ | $\operatorname{diag}(A^{-1}) \otimes \operatorname{diag}(B^{-1})$ | per factor |
| `KroneckerSum` $A \oplus B$ (grid Laplacians, SPDE on a raster) | $(U_A \circ U_A)\,M\,(U_B \circ U_B)^\top$, $M_{ij} = 1/(\lambda^A_i + \lambda^B_j)$ | $O(H^2 W + H W^2)$ |
| $A \otimes B + cI$ | same, with $M_{ij} = 1/(\lambda^A_i \lambda^B_j + c)$ | per factor |

`pinv=True` drops the zero eigenvalues of an intrinsic precision on a
grid. A factor that already carries its eigenbasis (`DiagonalizedOperator`,
`KroneckerSum`) keeps it, so `solve`, `logdet` and `diag_inv` of a
space-time $A \otimes B + cI$ never form the spatial factor $B$.

```python
# Posterior sd of an AR(1) trend under Gaussian noise: Q + σ⁻² I stays block-tridiagonal
H = gaussx.BlockTriDiag(Q.diagonal + jnp.eye(1) / sigma**2, Q.sub_diagonal)
sd = jnp.sqrt(gaussx.diag_inv(H))  # O(N), not O(N³)

# Prior sd of an intrinsic field on an H × W grid: two small matrix products
sd = jnp.sqrt(gaussx.diag_inv(gaussx.KroneckerSum(L_H, L_W), pinv=True))
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [diag_inv, selected_inverse]

## Matrix-RHS & batched solves

Solve $AX = B$ for matrix right-hand sides with one factorization
(`solve_matrix`), per-column or per-row structured dispatch
(`solve_columns` / `solve_rows`), or $O(n)$ Thomas-algorithm tridiagonal
solves.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [solve_matrix, solve_columns, solve_rows, tridiagonal_solve, tridiagonal_solve_batched]

## Matrix diagonalization & shifted Kronecker-sum solves

Factor each 1D operator once with `EigenDecomposition` (non-symmetric
diagonalizable factors with a real spectrum are supported), then solve
$(A_0 \oplus A_1 \oplus \dots - \sigma I)\,x = b$ in tensor form for any
shift $\sigma$ with `kronecker_sum_solve` — the matrix-diagonalization method
for separable operators such as tensor-product spectral Laplacians.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [EigenDecomposition, kronecker_sum_solve]

## Stable distances & Lyapunov

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [stable_squared_distances, discrete_lyapunov_solve]
