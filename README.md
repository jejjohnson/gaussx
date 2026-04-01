# gaussx

[![Tests](https://github.com/jejjohnson/gaussx/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/gaussx/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/gaussx/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/typecheck.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Structured linear algebra, Gaussian distributions, and exponential family primitives for JAX.**

Built on top of [lineax](https://github.com/patrick-kidger/lineax), [equinox](https://github.com/patrick-kidger/equinox), and [matfree](https://github.com/pnkraemer/matfree).

## Installation

```bash
pip install gaussx
```

Or with `uv`:

```bash
uv add gaussx
```

## Quick Start

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

# Structured operators with structural dispatch
A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 3.0]))
B = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
K = gaussx.Kronecker(A, B)

v = jnp.ones(6)
x = gaussx.solve(K, v)       # Per-factor solve (efficient)
ld = gaussx.logdet(K)         # n_B * logdet(A) + n_A * logdet(B)
L = gaussx.cholesky(K)        # Kronecker(chol(A), chol(B))

# Distributions with pluggable solver strategies
mvn = gaussx.MultivariateNormal(
    loc=jnp.zeros(6),
    covariance_operator=K,
    strategy=gaussx.DenseSolver(),
)
log_p = mvn.log_prob(v)
```

## What's Inside

### Layer 0 -- Primitives

Pure functions with `isinstance`-based structural dispatch. Each primitive automatically exploits the structure of its input operator (Kronecker, block-diagonal, low-rank, etc.).

`solve` | `logdet` | `cholesky` | `diag` | `trace` | `sqrt` | `inv` | `eig` | `eigvals` | `svd`

### Layer 1 -- Operators

Extend `lineax.AbstractLinearOperator` with structured matrices:

| Operator | Description |
|----------|-------------|
| `Kronecker` | Kronecker product A_1 &otimes; ... &otimes; A_k |
| `KroneckerSum` | Kronecker sum A &oplus; B = A &otimes; I + I &otimes; B |
| `BlockDiag` | Block diagonal diag(A_1, ..., A_k) |
| `BlockTriDiag` | Block tridiagonal (lower/upper variants) |
| `LowRankUpdate` | A + UDV^T |
| `SVDLowRankUpdate` | SVD-factored low-rank update |
| `ImplicitKernelOperator` | Matrix-free kernel operator |

### Layer 1.5 -- Solver Strategies

Pluggable solve + logdet algorithms that decouple numerics from distributions:

`DenseSolver` | `AutoSolver` | `CGSolver` | `PreconditionedCGSolver` | `LSMRSolver` | `BBMMSolver`

### Layer 2 -- Distributions & Sugar

**Distributions**: `MultivariateNormal`, `MultivariateNormalPrecision` (NumPyro-compatible), `conditional`

**Sugar** (compound operations built from primitives):

- **Gaussian utilities** -- log prob, entropy, KL, quadratic form, whiten/unwhiten
- **Schur complement** -- conditional variance from block matrices
- **Variational inference** -- ELBO (analytic & MC), SVGP (whitened), SpinGP
- **Bayesian linear regression** -- full & diagonal updates, GGN diagonal, Hutchinson trace
- **Natural gradients** -- damped natural updates, Gauss-Newton precision, Riemannian PSD correction
- **Kernel approximation** -- Nystrom, RFF, kernel centering, HSIC, MMD
- **Quadrature** -- sigma points, cubature points, Gauss-Hermite
- **Numerics** -- Woodbury solve, Joseph-form covariance update, covariance transform

### Layer 3 -- Recipes

Cross-cutting patterns combining multiple layers:

| Recipe | Functions |
|--------|-----------|
| **Kalman filter** | `kalman_filter`, `kalman_gain`, `rts_smoother` |
| **Parallel Kalman** | `parallel_kalman_filter`, `parallel_rts_smoother` |
| **SSM natural params** | `ssm_to_naturals`, `naturals_to_ssm`, `ssm_to_expectations`, `expectations_to_ssm` |
| **Gaussian sites (CVI)** | `GaussianSites`, `cvi_update_sites`, `sites_to_precision` |
| **Kronecker GP** | `kronecker_mll`, `kronecker_posterior_predictive` |
| **SpinGP** | `spingp_log_likelihood`, `spingp_posterior` |
| **LOVE** | `love_cache`, `love_variance` |
| **Ensemble** | `ensemble_covariance`, `ensemble_cross_covariance` |
| **Interpolation** | `conditional_interpolate` |

### Exponential Family

Gaussian in natural parameter form: `GaussianExpFam` with conversions between natural and expectation parameters, sufficient statistics, log partition, Fisher information, and KL divergence.

### Uncertainty Propagation

Gaussian-in / Gaussian-out transforms for nonlinear functions:

- **Integrators**: `TaylorIntegrator`, `UnscentedIntegrator`, `MonteCarloIntegrator`
- **State estimation**: `AssumedDensityFilter`
- **GP predictions under input uncertainty**: `uncertain_gp_predict`, `uncertain_svgp_predict`, `uncertain_vgp_predict`, `uncertain_bgplvm_predict`

## API Notes

A few usage details that are easy to miss:

- `gaussx.kronecker_posterior_predictive(...)` requires `K_test_diag_factors=` so predictive variances use the exact prior diagonal at the test points instead of reconstructing it from cross-covariances.
- `gaussx.ssm_to_naturals(A, Q, mu_0, P_0)` expects `Q[0]` to equal `P_0`; the function raises a `ValueError` if the initial covariance is inconsistent.
- `gaussx.ImplicitKernelOperator(...)` only reports `lineax` structure such as symmetry or PSD when those tags are passed explicitly.

## Development

```bash
git clone https://github.com/jejjohnson/gaussx.git
cd gaussx
make install      # install all dependency groups
make test         # run tests
make lint         # ruff check .
make typecheck    # ty check src/gaussx
make docs-serve   # preview docs locally
```

## License

MIT
