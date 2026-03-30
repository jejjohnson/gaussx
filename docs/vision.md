# Vision

> **gaussx** is a JAX/Equinox library that provides structured linear operators, Gaussian distributions, and exponential family primitives under one roof.

## Why gaussx?

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
