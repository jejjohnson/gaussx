# Primitives

Layer 0: pure functions over `lineax.AbstractLinearOperator` with **structural
dispatch** — each primitive inspects the operator (diagonal, Kronecker,
block-diagonal, low-rank, block-tridiagonal, …) and routes to the cheapest exact
algorithm, falling back to a dense computation (with a
[`DenseFallbackWarning`](#gaussx.DenseFallbackWarning)) only when no structured
path exists.

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
