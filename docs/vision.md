# Vision

> **gaussx** is a JAX/Equinox library that provides structured linear operators, Gaussian distributions, and exponential family primitives under one roof.

## Why structured linear algebra matters

Linear algebra is the computational backbone of scientific computing and machine learning --- more foundational than most practitioners realize. Nearly every algorithm eventually bottlenecks on solving a linear system, computing a determinant, or factorizing a matrix:

- **Linear regression** solves $(X^\top X) \beta = X^\top y$
- **Gaussian processes** need $(K + \sigma^2 I)^{-1} y$ and $\log|K + \sigma^2 I|$ for every evaluation of the marginal likelihood
- **Kalman filters** compute the gain $K = P H^\top (H P H^\top + R)^{-1}$ at every time step
- **Variational inference** requires $\log|\Sigma|$ and samples from $\mathcal{N}(\mu, \Sigma)$ --- both need a matrix square root or Cholesky factor
- **Natural gradient methods** invert the Fisher information matrix $F^{-1} \nabla \mathcal{L}$ at each update
- **Ensemble methods** form empirical covariances $\frac{1}{J} \sum (x_j - \bar{x})(x_j - \bar{x})^\top$ that are inherently low-rank
- **Spatial statistics on grids** produce Kronecker-structured covariances $K_x \otimes K_y$ where naive $O(N^3)$ becomes $O(n_x^3 + n_y^3)$
- **PDE solvers** invert elliptic operators, precondition iterative methods, and compute spectral decompositions
- **Optimal transport** solves Sinkhorn iterations that reduce to repeated matrix-vector products

The same handful of operations --- `solve`, `logdet`, `cholesky`, `trace`, `sqrt` --- appear again and again across these fields. The default approach of materializing a dense matrix and calling LAPACK works for toy problems but collapses at scale. Real problems have **structure**: the matrix is Kronecker, block diagonal, low-rank plus diagonal, sparse, or symmetric positive definite. Exploiting that structure is often the difference between $O(n^3)$ and $O(n)$ --- between a computation that takes hours and one that takes milliseconds.

Yet in practice, every research codebase re-discovers and re-implements these structural tricks from scratch. The Woodbury identity gets hand-coded in the GP library, then again in the filtering library, then again in the Bayesian optimization library. Each implementation is correct but isolated. Bugs don't get shared fixes. Performance improvements don't propagate. And newcomers face a wall of bespoke linear algebra code before they can focus on their actual research problem.

gaussx exists to end this cycle: provide the structured operators and dispatch-based primitives once, correctly, and let every downstream library build on them.

## Why gaussx (specifically)?

Every project in the JAX scientific computing ecosystem reimplements the same linear algebra patterns:

- **GP libraries** hand-roll Cholesky solvers, Woodbury identity, Kronecker eigendecomposition, stochastic trace estimation.
- **Kernel methods** hand-roll kernel matrix operations, CG solvers, centering matrices, HSIC traces.
- **Data assimilation** hand-rolls ensemble covariance computation, Kalman gain, matrix square roots for sigma points.
- **Bayesian learning** hand-rolls Fisher information inversion, natural-to-expectation parameter conversions.

The implementations are correct but scattered. Each project carries its own operator types, dispatch tables, and solve routines. Bug fixes don't propagate. Optimizations aren't shared. New projects start from scratch.

Meanwhile, the existing tools each solve *part* of the problem:

| Library | Strengths | Gap |
|---------|-----------|-----|
| **lineax** | Excellent solvers (CG, Cholesky, LU, GMRES, ...), clean operator abstraction | No Kronecker, BlockDiag, or LowRank operators; no logdet, trace, sqrt |
| **CoLA** | Rich operator zoo, matrix functions (logdet, trace, sqrt, exp) | Multi-backend, not Equinox-native, weaker solver coverage |
| **TFP JAX** | Battle-tested LinearOperators with batching | Heavy dependency with TF baggage |

**gaussx fills the gap**: lineax's solvers + CoLA's operator breadth + Gaussian distribution layer, all native to JAX/Equinox.

## Who is gaussx for?

**GP researcher** --- "I have a spatiotemporal kernel that's Kronecker-structured. I want `logdet` and `solve` to automatically exploit that structure without me reimplementing the Kronecker eigendecomposition trick every time."

**Data assimilation researcher** --- "I need ensemble covariance as a low-rank operator and Kalman gain via Woodbury. I want to call `low_rank_plus_diag(noise, U)` and get a structured operator back."

**Bayesian ML researcher** --- "I'm implementing natural gradient methods. I need natural-to-expectation parameter conversions and Fisher information for Gaussians, working with structured precision operators."

**Library developer** --- "I'm building a GP/filtering/optimization library. I need a shared linear algebra layer so I stop reimplementing `solve`, `logdet`, and `cholesky` with per-type dispatch in every project."

**Student / newcomer** --- "I want `gaussx.solve(K, y)` to just work, whether my matrix is dense, Kronecker, or low-rank+diagonal."

## Design Principles

| # | Principle | What it means |
|---|-----------|---------------|
| 1 | **Extend, don't replace** | Build on lineax's `AbstractLinearOperator` and solver suite. Don't reinvent what already works. |
| 2 | **Structure drives dispatch** | Operators carry structural tags (PSD, symmetric, Kronecker, ...). Primitives inspect types + tags via isinstance to select the efficient code path. No magic. |
| 3 | **Math-first layers** | Layer 0 functions match the equations in papers: `solve(A, b)`, `logdet(A)`, `cholesky(A)`. A researcher should read the code and see the math. |
| 4 | **One distribution, many strategies** | `MultivariateNormal` accepts any covariance operator and any solver strategy. The distribution handles the math, the solver handles the numerics. |
| 5 | **einops for readability** | All tensor reshaping uses `rearrange` and `einsum`. Kronecker `mv` reads like Roth's column lemma. |

## What gaussx is NOT

| Not this | Use instead |
|----------|-------------|
| GP library (kernels, priors, inference) | pyrox_gp |
| Probabilistic programming (MCMC, SVI) | NumPyro |
| General-purpose optimization | optax / optimistix |
| PDE solvers | finitevolx / spectraldiffx |
| Ensemble Kalman filters | filterX (consumes gaussx) |
| Multi-backend (PyTorch, NumPy) | JAX only |

## Ecosystem

```
                    ┌──────────────┐
                    │   lineax     │  Solvers, base operators, tags
                    │   matfree    │  Iterative/stochastic LA
                    └──────┬───────┘
                           │ extends
                    ┌──────▼───────┐
                    │    gaussx    │  ← equinox, jaxtyping, einops
                    └──────┬───────┘
                           │ consumed by
           ┌───────────────┼───────────────┐
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │  pyrox_gp   │ │   filterX   │ │ optax_bayes │
    │  (GPs)      │ │  (ensemble  │ │  (natural   │
    │             │ │   Kalman)   │ │   gradient) │
    └─────────────┘ └─────────────┘ └─────────────┘
```
