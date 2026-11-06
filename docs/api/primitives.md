# Primitives

Layer 0: pure functions over `lineax.AbstractLinearOperator` with **structural
dispatch** — each primitive inspects the operator (diagonal, Kronecker,
block-diagonal, low-rank, block-tridiagonal, …) and routes to the cheapest exact
algorithm, falling back to a dense computation only when no structured path
exists. Where a structured operator has to be materialized anyway — `cholesky`
of a `SumOfKroneckers` — a [`DenseFallbackWarning`](#gaussx.DenseFallbackWarning)
names the matrix-free alternative.

## Solve, logdet & Cholesky

The workhorses behind Gaussian densities: $\mathbf{A}^{-1}\mathbf{b}$, $\log \det \mathbf{A}$, and $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$.
`cholesky` returns a *lazy* lower-triangular operator that preserves structure
(the Cholesky of a `Kronecker` is a `Kronecker` of Cholesky factors);
`cholesky_logdet` turns an existing factor into $\log \det \mathbf{A} = 2 \sum_i \log L_{ii}$
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

`pseudo_logdet` is $\log \det_+ \mathbf{A} = \sum_{i: \gamma_i > 0} \log \gamma_i$ for a
symmetric PSD $\mathbf{A}$: half of it is the normalising constant of an intrinsic GMRF
(Besag / ICAR, RW1, RW2), whose structure matrix is singular. Each structure
takes its cheapest exact path:

| Operator | Path |
|---|---|
| `structure="laplacian"` on a `SparseOperator` or a `1 × 1`-block `BlockTriDiag` (Besag, RW1) | Matrix-tree theorem: $\log \det_+ \mathbf{L} = \sum_c (\log N_c + \log \det \mathbf{L}_{c,\text{minor}})$ over connected components, one sparse (or banded) Cholesky in all, sparsity kept |
| `null_space=B` (any basis $\mathbf{B}$ of $\ker \mathbf{A}$) | $\log \det(\mathbf{A} + \mathbf{B}\mathbf{B}^\top) - \log \det(\mathbf{B}^\top \mathbf{B})$; the sum is matvec-only, so `strategy=SLQLogdet()` estimates it matrix-free |
| `KroneckerSum` (a grid Laplacian) | Factor eigenvalues, all pairwise sums $\gamma^H_i + \gamma^W_j$; no factorisation |
| anything else | dense `eigvalsh` |

The eigenvalue paths drop $\gamma_i \le$ `rcond` $\cdot \max_i \gamma_i$. The matrix-tree theorem says every principal minor of a connected weighted Laplacian equals the weighted spanning-tree count, and $\det_+ \mathbf{L}$ is $N$ times that count. $\log \det_+ \mathbf{R}$ does not depend on the precision scale $\tau$ ($\log \det_+(\tau \mathbf{R}) = \mathrm{rank}(\mathbf{R}) \log \tau + \log \det_+ \mathbf{R}$), so compute it once and cache it.

```python
import jax.numpy as jnp
import lineax as lx
import gaussx

# ICAR normalising constant for model comparison: computed once, cached
R = gaussx.besag_structure(laplacian)  # a SparseOperator graph Laplacian
half_log_pdet = 0.5 * gaussx.pseudo_logdet(R, structure="laplacian")

# On a grid: no factorisation, sum the logs of the non-zero γ^H_i + γ^W_j
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

Exact where structure allows; stochastic (Hutchinson / Hutch++ / XTrace
probing) for matrix-free operators. `trace_and_diag` shares one probe pass
between both estimates.

Hutch++ (`algorithm="hutchpp"`, Meyer et al., 2021) and XTrace
(`algorithm="xtrace"`, Epperly et al., 2024) deflate a randomized low-rank
part, $\mathrm{tr}(\mathbf{A}) = \mathrm{tr}(\mathbf{Q}^\top \mathbf{A} \mathbf{Q}) + \mathrm{tr}((\mathbf{I} - \mathbf{Q}\mathbf{Q}^\top) \mathbf{A} (\mathbf{I} - \mathbf{Q}\mathbf{Q}^\top))$, and probe only the remainder. On a decaying spectrum they reach relative error $\varepsilon$ in $O(1/\varepsilon)$ matvecs, against Hutchinson's $O(1/\varepsilon^2)$. Examples are traces of kernel matrices, of $\mathbf{A}^{-1}\mathbf{B}$ in GP hyperparameter gradients, or of Hessians.

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

`eigh_generalized` solves $\mathbf{A} \mathbf{v} = \gamma \mathbf{B} \mathbf{v}$ for symmetric $\mathbf{A}$ and PSD $\mathbf{B}$, returning $\mathbf{B}$-orthonormal eigenvectors (the minimisers of $\mathrm{tr}(\mathbf{Y}^\top \mathbf{A} \mathbf{Y})$ s.t. $\mathbf{Y}^\top \mathbf{B} \mathbf{Y} = \mathbf{I}$ behind Laplacian eigenmaps, LPP and manifold alignment). It dispatches on $\mathbf{B}$: a positive diagonal stays matrix-free (Lanczos with `rank=`), a tagged positive-definite $\mathbf{B}$ is Cholesky-whitened, and a singular $\mathbf{B}$ has its null space eliminated by a Schur complement.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [eigh_generalized]

## Matrix-free square-root products

$\mathbf{A}^{\pm 1/2}\mathbf{b}$ for an operator too large to factorise, via the
Hale–Higham–Trefethen contour-integral quadrature: a weighted sum of ~15
shifted solves $(\mathbf{A} + \sigma_j \mathbf{I})^{-1}\mathbf{b}$, each routed back through `solve`.
Accuracy depends on the condition number only logarithmically, and
`estimate_spectral_bounds` supplies the contour parameters when they are not
known ahead of time.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [sqrt_matmul, sqrt_inv_matmul, estimate_spectral_bounds]

## Joint inverse-quadratic and log-determinant

`inv_quad_logdet` returns $\mathrm{tr}(\mathbf{B}^\top \mathbf{A}^{-1} \mathbf{B})$ and $\log \det \mathbf{A}$ from a
single modified-batched-CG pass — the two halves of a Gaussian log-density at
roughly the matvec budget of one. Supplying `logdet_preconditioner=` $\mathbf{P} \approx \mathbf{A}$
switches on the Artemev et al. variance reduction, estimating only the
near-identity residual $\mathrm{tr}(\log(\mathbf{P}^{-1}\mathbf{A}))$ stochastically. $\mathbf{P}$ approximates $\mathbf{A}$
itself, unlike the `preconditioner=` ($\mathbf{M}^{-1} \approx \mathbf{A}^{-1}$) of the CG
solvers. The old keyword `preconditioner=` is a deprecated alias.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [inv_quad_logdet]

## Root decompositions

Tall-factor approximations $\mathbf{R}\mathbf{R}^\top \approx \mathbf{A}$ (and $\mathbf{R}^- (\mathbf{R}^-)^\top \approx \mathbf{A}^{-1}$) via Cholesky, pivoted Cholesky, Lanczos, or truncated SVD — the
building block for low-rank posterior sampling and BBMM-style solvers.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [root_decomposition, root_inv_decomposition, RootDecomposition]

## Quantiles

`mixture_quantile` inverts a CDF $F$ at levels $\tau$ by solving
$F(y) - \tau = 0$ on a bracket $[y_\text{lo}, y_\text{hi}]$ — the standard remedy for a mixture
$F_\text{mix} = \frac{1}{E}\sum_e F_e$, which has no closed-form inverse.
It runs [optimistix](https://docs.kidger.site/optimistix/)'s `root_find`
with `Chandrupatla`, a bracketing solver that mixes inverse quadratic
interpolation with bisection (Chandrupatla, 1997), so it needs no
derivatives, and differentiates through the implicit adjoint
$\partial y_\tau / \partial \boldsymbol{\theta} = -\partial_{\boldsymbol{\theta}} F / p(y_\tau)$.
`mixture_quantile_gaussian_approx` is the cheap moment-matched Gaussian
alternative for Gaussian ensembles (Lakshminarayanan et al., 2017).

```python
import jax.numpy as jnp
from jax.scipy.stats import norm
import gaussx

means = jnp.array([-1.0, 1.0])  # two-component mixture
cdf = lambda x: 0.5 * (norm.cdf(x - means[0]) + norm.cdf(x - means[1]))
gaussx.mixture_quantile(cdf, jnp.array([0.025, 0.5, 0.975]), -10.0, 10.0)
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [mixture_quantile, mixture_quantile_gaussian_approx, Chandrupatla]

## Support types

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [DenseFallbackWarning]
