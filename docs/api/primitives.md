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

`solve(A, b, solver=...)` takes either a *lineax* solver, such as
`lineax.CG(...)`, which replaces the dense fallback and is threaded into the
structural rules (per Kronecker factor, per block), or a *gaussx* strategy, such
as `CGSolver()`, which then owns the whole solve. See
[Solvers](solvers.md).

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [solve, logdet, cholesky, cholesky_logdet]

## Pseudo-determinant

`pseudo_logdet` is $\log|A|_+ = \sum_{\lambda_i>0}\log\lambda_i$ for a
symmetric PSD $A$: half of it is the normalising constant of an intrinsic GMRF
(Besag / ICAR, RW1, RW2), whose structure matrix is singular. Each structure
takes its cheapest exact path:

| Operator | Path |
|---|---|
| `structure="laplacian"` on a `SparseOperator` or a `1 × 1`-block `BlockTriDiag` (Besag, RW1) | Matrix-tree theorem: $\log\operatorname{pdet}(L) = \sum_c(\log n_c + \log\lvert L_c^{(-k_c)}\rvert)$ over connected components, one sparse (or banded) Cholesky in all, sparsity kept |
| `null_space=B` (any basis of $\ker A$) | $\log\lvert A + BB^\top\rvert - \log\lvert B^\top B\rvert$; the sum is matvec-only, so `strategy=SLQLogdet()` estimates it matrix-free |
| `KroneckerSum` (a grid Laplacian) | Factor eigenvalues, all pairwise sums $\lambda^H_i + \lambda^W_j$; no factorisation |
| anything else | dense `eigvalsh` |

The eigenvalue paths drop $\lambda \le$ `rcond` $\cdot\,\lambda_{\max}$. The
matrix-tree theorem says every principal minor of a connected weighted
Laplacian equals the weighted spanning-tree count $\tau_w(G)$ and
$\operatorname{pdet}(L) = N\,\tau_w(G)$. $\log|R|_+$ does not depend on the
precision scale $\tau$ ($\log|\tau R|_+ = \operatorname{rank}(R)\log\tau +
\log|R|_+$), so compute it once and cache it.

```python
import jax.numpy as jnp
import lineax as lx
import gaussx

# ICAR normalising constant for model comparison: computed once, cached
R = gaussx.besag_structure(laplacian)  # a SparseOperator graph Laplacian
half_log_pdet = 0.5 * gaussx.pseudo_logdet(R, structure="laplacian")

# On a grid: no factorisation, sum the logs of the non-zero λ^H_i + λ^W_j
L_H = lx.MatrixLinearOperator(gaussx.rw1_structure(512).as_matrix(), lx.symmetric_tag)
L_W = lx.MatrixLinearOperator(gaussx.rw1_structure(512).as_matrix(), lx.symmetric_tag)
half_log_pdet_grid = 0.5 * gaussx.pseudo_logdet(gaussx.KroneckerSum(L_H, L_W))
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [pseudo_logdet]

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

## Generalised eigenproblems

`eigh_generalized` solves $A v = \lambda B v$ for symmetric $A$ and PSD $B$,
returning $B$-orthonormal eigenvectors (the minimisers of
$\operatorname{tr}(Y^\top A Y)$ s.t. $Y^\top B Y = I$ behind Laplacian
eigenmaps, LPP and manifold alignment). It dispatches on $B$: a positive
diagonal stays matrix-free (Lanczos with `rank=`), a tagged positive-definite
$B$ is Cholesky-whitened, and a singular $B$ has its null space eliminated by a
Schur complement.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [eigh_generalized]

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
roughly the matvec budget of one. Supplying `logdet_preconditioner=` $P \approx A$
switches on the Artemev et al. variance reduction, estimating only the
near-identity residual $\log(P^{-1}A)$ stochastically. $P$ approximates $A$
itself, unlike the `preconditioner=` ($M^{-1} \approx A^{-1}$) of the CG
solvers. The old keyword `preconditioner=` is a deprecated alias.

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
