# State-Space Models & Kalman

Layer 3 recipes for linear-Gaussian state-space models. Stationary 1-D GP
kernels with rational spectral densities admit exact SDE representations

$$
\dot{x}(t) = F\,x(t) + L\,w(t), \qquad f(t) = H\,x(t),
$$

turning $O(N^3)$ GP inference into $O(N d^3)$ Kalman filtering. This page
covers the SDE kernel zoo, the filters and smoothers (sequential, parallel
associative-scan, square-root, and steady-state), and the natural-parameter /
site machinery for non-conjugate likelihoods.

## SDE kernels

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SDEKernel, SDEParams, ConstantSDE, MaternSDE, PeriodicSDE, QuasiPeriodicSDE, CosineSDE, ProductSDE, SumSDE, sde_autocovariance]

## Kalman filtering & smoothing

The forward filter and RTS smoother, their $O(\log N)$ parallel
(associative-scan) counterparts, and the steady-state (infinite-horizon)
variants built on the discrete algebraic Riccati equation.

### Observation masks

`kalman_filter` and `parallel_kalman_filter` take an optional `mask`,
dispatched on its rank:

| shape | meaning |
|---|---|
| `(T,)` | **Per-step gate.** `False` runs the predict step only and contributes nothing to the log-likelihood — the usual way to predict on a merged train/test grid. |
| `(T, M)` | **Per-channel gate**, for partially observed multivariate series where different channels are measured at different times. |

The per-channel path marginalises unobserved channels *exactly*: row $i$ of
$H_t$ is zeroed, a unit block is substituted into $R_t$, and the residual
entry is set to zero. The innovation covariance is then block-diagonal in the
observed/masked split, so column $i$ of the gain vanishes and the masked
channel cannot move the state — the posterior reproduces the row-deleted
filter to machine precision, with no branching. A dummy block also
contributes $-\tfrac12 \log 2\pi$ per masked channel to the full-vector
density, which is stripped per step, so `log_likelihood` is the exact marginal
$\log p(y_{\mathrm{obs}})$ and is invariant to both the mask pattern and the
dummy variance.

An all-`False` row of a `(T, M)` mask is equivalent to a `False` entry in the
`(T,)` form, and `M == 1` is unambiguous either way. Masked entries of
`observations` are never read, so they may be `NaN`. Operator-typed
`obs_model` / `obs_noise` are materialised under a `(T, M)` mask, since
zeroing rows is inherently dense; `form="sqrt"` supports the `(T,)` mask only
and raises `NotImplementedError` otherwise. `rts_smoother` needs no mask of
its own — it consumes filtered/predicted moments, which are already
mask-aware.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [EmissionModel, FilterState, kalman_filter, rts_smoother, kalman_gain, parallel_kalman_filter, parallel_rts_smoother, infinite_horizon_filter, infinite_horizon_smoother, InfiniteHorizonState, dare, DAREResult, pairwise_marginals]

## SpInGP

State-space (sparse-in-time) GP inference: marginal likelihood and posterior
through the SSM representation.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [spingp_log_likelihood, spingp_posterior]

## Sites & natural parameters

Conjugate-computation VI (CVI) site updates and the conversions between SSM
moment, expectation, and natural parameterizations used by non-conjugate
temporal inference.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [GaussianSites, cvi_update_sites, sites_to_precision, cavity_from_marginal, site_natural_from_tilted, site_mean_var_from_natural, expectations_to_ssm, naturals_to_ssm, ssm_to_expectations, ssm_to_naturals]

## Process noise

The exact discretisation $Q = P_\infty - A P_\infty A^\top$, which is the
forward direction of the discrete Lyapunov equation that
[`discrete_lyapunov_solve`](linalg.md#gaussx.discrete_lyapunov_solve) inverts.
The congruence is delegated to
[`cov_transform`](linalg.md#gaussx.cov_transform), so passing *operators*
returns a lazy operator with structure intact — matched `Kronecker` factors stay
factorised, a diagonal $P_\infty$ skips its $(N, N)$ materialization. Passing
arrays returns an array.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [process_noise_covariance]
