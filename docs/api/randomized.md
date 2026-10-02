# Randomized Linear Algebra

Randomized factorisations that touch a matrix only through a few columns or
matvecs. They take an explicit PRNG `key`; `key=None` means
`jax.random.PRNGKey(0)`.

## Randomly pivoted Cholesky

`rp_cholesky` builds a partial Cholesky factor from the diagonal and a
`column(j)` callable, picking each pivot with probability proportional to the
residual diagonal (Chen, Epperly, Tropp & Webber, 2023). It returns the
pivots too, so they serve as landmark indices for Nyström, Falkon or SVGP
inducing points without ever forming the kernel matrix. `pivoting="greedy"`
is the classic pivoted Cholesky behind
[`PartialCholeskyPreconditioner`](solvers.md#gaussx.PartialCholeskyPreconditioner).

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [rp_cholesky]
