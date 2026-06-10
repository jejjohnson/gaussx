# Linear-Algebra Utilities

Numerically careful building blocks shared by the higher layers: robust
factorizations, classical matrix identities in operator form, and batched /
matrix-RHS solve helpers.

## Robust factorization & hygiene

`safe_cholesky` retries with geometrically growing diagonal jitter inside a
`jax.lax.while_loop` (JIT-compatible) when a matrix is not numerically
positive-definite; `symmetrize` removes the floating-point asymmetry that
accumulates in covariance updates.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [safe_cholesky, symmetrize]

## Matrix identities

Woodbury, Schur complements, and conditional (Schur-complement) variances —
the identities behind every Gaussian conditioning step, exposed directly so
recipes never re-derive them.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [woodbury_solve, schur_complement, conditional_variance, diag_conditional_variance, cov_transform, sandwich, trace_product, diag_inv]

## Matrix-RHS & batched solves

Solve $AX = B$ for matrix right-hand sides with one factorization
(`solve_matrix`), per-column or per-row structured dispatch
(`solve_columns` / `solve_rows`), or $O(n)$ Thomas-algorithm tridiagonal
solves.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [solve_matrix, solve_columns, solve_rows, solve_tridiagonal, solve_tridiagonal_batched, batched_kernel_matvec, batched_kernel_rmatvec]

## Stable kernel arithmetic & Lyapunov

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [stable_squared_distances, stable_rbf_kernel, discrete_lyapunov_solve]
