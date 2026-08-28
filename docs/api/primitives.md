# Primitives

Layer 0: pure functions over `lineax.AbstractLinearOperator` with **structural
dispatch** — each primitive inspects the operator (diagonal, Kronecker,
block-diagonal, low-rank, block-tridiagonal, …) and routes to the cheapest exact
algorithm, falling back to a dense computation only when no structured path
exists. Where a structured operator has to be materialized anyway — `cholesky`
of a `SumOfKroneckers` — a [`DenseFallbackWarning`](#gaussx.DenseFallbackWarning)
names the matrix-free alternative.

## Solve, logdet & Cholesky

The workhorses behind Gaussian densities: $A^{-1}b$, $\log|A|$, and $A = LL^\top$.
`cholesky` returns a *lazy* lower-triangular operator that preserves structure
(the Cholesky of a `Kronecker` is a `Kronecker` of Cholesky factors);
`cholesky_logdet` turns an existing factor into $\log|A| = 2\sum_i \log L_{ii}$
for free.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [solve, logdet, cholesky, cholesky_logdet]

## Trace & diagonal

Exact where structure allows; stochastic (Hutchinson / XTrace probing) for
matrix-free operators. `trace_and_diag` shares one probe pass between both
estimates.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [trace, diag, trace_and_diag]

## Inverse, square root & spectral decompositions

`inv` and `sqrt` return lazy operators that route their matvecs through
structured solves / Lanczos; `eig`, `eigvals`, and `svd` take an optional `rank`
for partial (Krylov) decompositions.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [inv, sqrt, eig, eigvals, svd, frobenius_norm, submatrix]

## Matrix-free square-root products

$A^{\pm 1/2}b$ for an operator too large to factorise, via the
Hale–Higham–Trefethen contour-integral quadrature: a weighted sum of ~15
shifted solves $(A + \sigma_j I)^{-1}b$, each routed back through `solve`.
Accuracy depends on the condition number only logarithmically, and
`estimate_spectral_bounds` supplies the contour parameters when they are not
known ahead of time.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [sqrt_matmul, sqrt_inv_matmul, estimate_spectral_bounds]

## Joint inverse-quadratic and log-determinant

`inv_quad_logdet` returns $\mathrm{tr}(R^\top A^{-1}R)$ and $\log|A|$ from a
single modified-batched-CG pass — the two halves of a Gaussian log-density at
roughly the matvec budget of one. Supplying a preconditioner $P \approx A$
switches on the Artemev et al. variance reduction, estimating only the
near-identity residual $\log(P^{-1}A)$ stochastically.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [inv_quad_logdet]

## Root decompositions

Tall-factor approximations $RR^\top \approx A$ (and $R^- (R^-)^\top \approx
A^{-1}$) via Cholesky, pivoted Cholesky, Lanczos, or truncated SVD — the
building block for low-rank posterior sampling and BBMM-style solvers.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [root_decomposition, root_inv_decomposition, RootDecomposition]

## Support types

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SumKroneckerSqrt, DenseFallbackWarning]
