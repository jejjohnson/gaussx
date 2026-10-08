# gaussx

[![Tests](https://github.com/jejjohnson/gaussx/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/gaussx/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/gaussx/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/typecheck.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

<!-- --8<-- [start:intro] -->
**Structured linear algebra, Gaussian distributions, and exponential family primitives for JAX.**

Built on top of [lineax](https://github.com/patrick-kidger/lineax), [equinox](https://github.com/patrick-kidger/equinox), and [matfree](https://github.com/pnkraemer/matfree).
<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:install] -->
## Installation

```bash
pip install gaussx
```

Or with `uv`:

```bash
uv add gaussx
```

<!-- --8<-- [end:install] -->

<!-- --8<-- [start:quickstart] -->
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
x = gaussx.solve(K, v)  # Per-factor solve (efficient)
ld = gaussx.logdet(K)  # n_B * logdet(A) + n_A * logdet(B)
L = gaussx.cholesky(K)  # Kronecker(chol(A), chol(B))

# Distributions with pluggable solver strategies
mvn = gaussx.MultivariateNormal(
    loc=jnp.zeros(6),
    cov_operator=K,
    solver=gaussx.DenseSolver(),
)
log_p = mvn.log_prob(v)
```
<!-- --8<-- [end:quickstart] -->

<!-- --8<-- [start:inside] -->
## What's Inside

gaussx is a layered stack: each layer builds on the ones beneath it, so
you can enter wherever your problem lives. The
[architecture page](https://jejjohnson.github.io/gaussx/architecture/) explains
the layers and the dispatch; each heading below links to its API reference.

### Layer 0 -- [Primitives](https://jejjohnson.github.io/gaussx/api/primitives/) and [linear-algebra utilities](https://jejjohnson.github.io/gaussx/api/linalg/)

Pure functions with `isinstance`-based structural dispatch. Each primitive automatically exploits the structure of its input operator (Kronecker, block-diagonal, low-rank, etc.).

`solve` | `logdet` | `cholesky` | `diag` | `trace` | `sqrt` | `inv` | `eig` | `eigvals` | `svd` | `root_decomposition` | `root_inv_decomposition`

**Utilities**: `woodbury_solve` | `schur_complement` | `safe_cholesky` | `symmetrize` | `tridiagonal_solve` | `discrete_lyapunov_solve` | `cov_transform`

### Layer 1 -- [Operators](https://jejjohnson.github.io/gaussx/api/operators/)

Extend `lineax.AbstractLinearOperator` with structured matrices. All are immutable `equinox.Module` pytrees, safe under `jit` / `grad` / `vmap`:

| Operator | Description |
|----------|-------------|
| `Kronecker` | Kronecker product A_1 &otimes; ... &otimes; A_k |
| `KroneckerSum` | Kronecker sum A &oplus; B = A &otimes; I + I &otimes; B |
| `SumOfKroneckers` | Sum of Kronecker products &Sigma;_k A_k &otimes; B_k (**not** the same as `KroneckerSum`) |
| `BlockDiag` | Block diagonal diag(A_1, ..., A_k) |
| `BlockTriDiag` | Block tridiagonal (lower/upper variants) |
| `LowRankUpdate` | A + UDV^T (pass `orthonormal=True` for SVD / Nystrom factors) |
| `DiagonalizedOperator` | V^-1 diag(&lambda;) V for a fast transform pair (FFT, DCT, ...); `circulant` builds the periodic case |
| `Toeplitz` | Symmetric Toeplitz, O(n log n) matvec via FFT |
| `InterpolatedOperator` | Grid-interpolated (KISS-GP style) |
| `MaskedOperator` | Row/column sub-selection of a base operator |
| `SparseOperator` | Sparse matrix on a static `SparsityPattern`, with sparse Cholesky ([sparse](https://jejjohnson.github.io/gaussx/api/sparse/), [GMRF precisions](https://jejjohnson.github.io/gaussx/api/gmrf/)) |
| `sum_operator`, `scaled_operator`, `product_operator` | Lazy algebra |

### Layer 1.5 -- [Solver Strategies & Preconditioners](https://jejjohnson.github.io/gaussx/api/solvers/)

Pluggable solve + logdet algorithms that decouple numerics from distributions. `linear_solve` is the high-level front door; `solver=None` anywhere means structural dispatch:

**Solvers**: `DenseSolver` | `AutoSolver` | `CGSolver` | `PreconditionedCGSolver` | `MINRESSolver` | `LSMRSolver` | `BBMMSolver` | `ComposedSolver`

**Logdets**: `DenseLogdet` | `SLQLogdet` | `IndefiniteSLQLogdet`

**Preconditioners**: `JacobiPreconditioner` | `NystromPreconditioner` | `PartialCholeskyPreconditioner` | `OperatorPreconditioner` (bring your own M^-1)

### Layer 2 -- [Distributions, Sugar & Exponential Family](https://jejjohnson.github.io/gaussx/api/distributions/)

**Distributions**: `MultivariateNormal`, `MultivariateNormalPrecision` (NumPyro-compatible), `MarkovGaussian`, `LGSSM`

**Sugar** (compound operations built from primitives): `gaussian_log_prob` | `gaussian_entropy` | `gaussian_kl` | `quadratic_form` | `conditional` | `joseph_update` | `project`

**Exponential family**: `GaussianExpFam` with conversions between natural and expectation parameters, sufficient statistics, log partition, Fisher information, and KL divergence.

### Layer 3 -- Recipes

Domain workflows that combine the layers below.

#### [Gaussian processes](https://jejjohnson.github.io/gaussx/api/gp/)

| Recipe | Functions |
|--------|-----------|
| GP conditioning | `sparse_conditional`, `predict_mean`, `predict_variance`, `build_prediction_cache` |
| Variational bounds | `variational_elbo_gaussian`, `variational_elbo_mc`, `collapsed_elbo`, `gauss_kl`, `whitened_svgp_predict` |
| Kronecker GP | `kronecker_mll`, `kronecker_posterior_predictive` |
| LOVE / LOO | `love_cache`, `love_variance`, `leave_one_out_cv` |
| Pathwise sampling | `matheron_update` |
| Multi-output (OILMM) | `oilmm_project`, `oilmm_back_project` |
| Interpolation | `conditional_interpolate` |

#### [State-space models](https://jejjohnson.github.io/gaussx/api/ssm/)

| Recipe | Functions |
|--------|-----------|
| Kalman filter | `kalman_filter`, `kalman_gain`, `rts_smoother` |
| Parallel Kalman | `parallel_kalman_filter`, `parallel_rts_smoother` |
| Steady-state Kalman | `infinite_horizon_filter`, `infinite_horizon_smoother`, `dare` |
| SSM natural params | `ssm_to_naturals`, `naturals_to_ssm`, `ssm_to_expectations`, `expectations_to_ssm` |
| Gaussian sites (CVI) | `GaussianSites`, `cvi_update_sites`, `sites_to_precision` |
| SpInGP | `spingp_log_likelihood`, `spingp_posterior` |
| SDE kernels | `MaternSDE`, `PeriodicSDE`, `QuasiPeriodicSDE`, `CosineSDE`, `ConstantSDE`, `IntegratedWienerSDE`, `SumSDE`, `ProductSDE` |

#### [Quadrature & uncertainty propagation](https://jejjohnson.github.io/gaussx/api/quadrature/)

| Recipe | Functions |
|--------|-----------|
| Integrators | `GaussHermiteIntegrator`, `TaylorIntegrator`, `UnscentedIntegrator`, `MonteCarloIntegrator`, `sigma_points`, `cubature_points`, `gauss_hermite_points` |
| Likelihoods | `GaussianLikelihood`, `HeteroscedasticGaussianLikelihood`, `BernoulliLikelihood`, `PoissonLikelihood`, `SoftmaxLikelihood`, `StudentTLikelihood` |
| State estimation and EP | `AssumedDensityFilter`, `ep_tilted_moments` |
| Uncertain-input GP prediction | `uncertain_gp_predict`, `uncertain_svgp_predict`, `uncertain_vgp_predict`, `uncertain_bgplvm_predict` |

#### [Inference & ensembles](https://jejjohnson.github.io/gaussx/api/inference/)

| Recipe | Functions |
|--------|-----------|
| Bayesian linear regression | `blr_full_update`, `blr_diag_update`, `ggn_diagonal`, `hutchinson_hessian_diag` |
| Natural gradients | `damped_natural_update`, `gauss_newton_precision`, `riemannian_psd_correction` |
| Ensemble (EnKF) | `ensemble_covariance`, `ensemble_cross_covariance`, `ensemble_kalman_gain`, `etkf_transform` |
| Localization / inflation | `gaspari_cohn`, `localized_kalman_gain`, `inflate_rtpp`, `inflate_rtps` |

### Outside the stack

[Sketching](https://jejjohnson.github.io/gaussx/api/sketching/) (`GaussianSketch`, `SRHTSketch`, `hadamard_transform`, ...) and [randomized linear algebra](https://jejjohnson.github.io/gaussx/api/randomized/) (`randomized_svd`, `randomized_nystrom`, `rp_cholesky`, ...) are standalone tools that produce operators and factors for the layers above.
<!-- --8<-- [end:inside] -->

## Documentation

- **[API Reference](https://jejjohnson.github.io/gaussx/api/)** — organised by layer; every public symbol is documented
- **[Architecture](https://jejjohnson.github.io/gaussx/architecture/)** — the layered stack, dispatch flow, and per-primitive fast-path coverage
- **[Vision](https://jejjohnson.github.io/gaussx/vision/)** — why gaussx exists and what it deliberately is not

<!-- --8<-- [start:api-notes] -->
## API Notes

A few usage details that are easy to miss:

- `gaussx.kronecker_posterior_predictive(...)` requires `K_test_diag_factors=` so predictive variances use the exact prior diagonal at the test points instead of reconstructing it from cross-covariances.
- `gaussx.ssm_to_naturals(A, Q, mu_0, P_0)` takes the transition noise `Q` of shape `(N-1, d, d)` and `P_0` separately, the same layout as `gaussx.MarkovGaussian`. The older stacked layout (`Q[0] == P_0`) is deprecated and will be removed in gaussx 0.7.0; with it, an inconsistent initial covariance raises an `EquinoxRuntimeError` (via `equinox.error_if`), eagerly and at run time under `jax.jit` / `jax.vmap`.
- `gaussx.SumKronecker` and `gaussx.SVDLowRankUpdate` are **deprecated** aliases (for `SumOfKroneckers` and `LowRankUpdate(..., orthonormal=True)`); both warn on construction and will be removed in gaussx 0.7.0.
- Kernel operators, Nyström / RFF operators, HSIC / MMD, Falkon and EigenPro moved to [kernellib](https://github.com/jejjohnson/kernellib) in 0.2.0. They are lineax operators and work with every gaussx primitive and strategy.
<!-- --8<-- [end:api-notes] -->

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
