# Kalman filtering and smoothing

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: state space.

Open this when you have a linear-Gaussian state-space model and need filtered or smoothed states and the log-likelihood.
It also covers the conversions between state-space parameters, natural parameters, expectation parameters and the UDL factorisation of a Markov chain's precision.

## Names

| Name | What it is |
|---|---|
| `kalman_filter(transition, obs_model, process_noise, obs_noise, observations, init_mean, init_cov, *, mask=None, solver=None, woodbury_innovation=False)` | The sequential Kalman filter by `lax.scan`; returns a `FilterState`. |
| `FilterState` | Holds `filtered_means`, `filtered_covs`, `predicted_means`, `predicted_covs` and the scalar `log_likelihood`. |
| `rts_smoother(filter_state, transition, *, solver=None)` | The Rauch-Tung-Striebel backward pass; returns `(smoothed_means, smoothed_covs)`. |
| `parallel_kalman_filter(...)`, `parallel_rts_smoother(...)` | The same filter and smoother by `lax.associative_scan`, with `O(log T)` depth; the filter also takes `psd_project=` and `square_root=`. |
| `meanfield_kalman_filter(..., *, block_size)`, `meanfield_rts_smoother(..., *, block_size)` | Independent filters and smoothers on diagonal state blocks, dropping cross-block covariance. |
| `dare(A, H, Q, R, *, P_init=None, max_iter=100, tol=1e-8)` | The steady-state covariance and gain by structure-preserving doubling, returned as a `DAREResult`. |
| `DAREResult(P_inf, K_inf, converged)` | The steady-state covariance, the gain and a convergence flag. |
| `infinite_horizon_filter(transition, obs_model, process_noise, obs_noise, observations, init_mean=None, *, dare_result=None)` | A filter with the fixed steady-state gain; returns a `FilterState`. |
| `infinite_horizon_smoother(filter_state, transition, process_noise, *, dare_result=None)` | The matching fixed-gain smoother; returns `(smoothed_means, smoothed_covs)`. |
| `kalman_gain(P, H, R, *, solver=None, woodbury_innovation=False)` | The single-step gain `P Hᵀ (H P Hᵀ + R)⁻¹` from operators. |
| `EmissionModel(H)` | A wrapper for a linear observation matrix with projection helpers such as `project_mean` and `project_covariance`. |
| `masked_moment_inputs(obs_cov, cross_cov, obs_noise, y, y_hat, mask)` | Makes masked channels inert in a moment-matched update, for custom filter loops. |
| `pairwise_marginals(means, covariances, cross_covariances)` | The joint `p(x_k, x_{k+1})` for each consecutive pair. |
| `ssm_to_naturals(A, Q, mu_0, P_0)`, `naturals_to_ssm(theta_linear, theta_precision, *, initial_in_q)` | Convert chain parameters to natural parameters with a `BlockTriDiag` precision, and back. |
| `ssm_to_expectations(means, covs, cross_covs)`, `expectations_to_ssm(eta1, eta2)` | Convert chain marginals to expectation parameters, and back. |
| `UDLDecomposition(U_sub, D_diag, chol_D)` | The banded factorisation `U D̃ Uᵀ` of a `BlockTriDiag` precision. |
| `udl_decomposition(precision)`, `udl_from_ssm_params(A, Q, P0)`, `udl_to_ssm_params(udl)` | Factorise a precision, build the factorisation from chain parameters, and read the chain back. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr

import gaussx

A = jnp.array([[1.0, 0.1], [0.0, 1.0]])  # (2, 2) constant-velocity transition
H, Q, R = jnp.array([[1.0, 0.0]]), 0.01 * jnp.eye(2), 0.1 * jnp.eye(1)
y = jnp.cumsum(0.1 * jr.normal(jr.key(0), (20, 1)), axis=0)  # (T, M) = (20, 1)
m0, P0 = jnp.zeros(2), jnp.eye(2)

fs = gaussx.kalman_filter(A, H, Q, R, y, m0, P0)  # FilterState
sm, sP = gaussx.rts_smoother(fs, A)  # (20, 2), (20, 2, 2)
assert fs.filtered_means.shape == (20, 2) and jnp.isfinite(fs.log_likelihood)
assert jnp.all(
    jnp.trace(sP, axis1=1, axis2=2)
    <= jnp.trace(fs.filtered_covs, axis1=1, axis2=2) + 1e-4
)

pf = gaussx.parallel_kalman_filter(A, H, Q, R, y, m0, P0)
assert jnp.allclose(pf.filtered_means, fs.filtered_means, atol=1e-3)
assert jnp.allclose(pf.log_likelihood, fs.log_likelihood, atol=1e-2)

d = gaussx.dare(A, H, Q, R)
assert d.converged and d.K_inf.shape == (2, 1)
ih = gaussx.infinite_horizon_filter(A, H, Q, R, y, m0, dare_result=d)
assert jnp.allclose(ih.filtered_covs[-1], d.P_inf, atol=1e-3)
```

## Rules

- Pass `transition`, `obs_model`, `process_noise` and `obs_noise` as one array for a time-invariant model or as `(T, …)` stacks for a time-varying one; lineax operators are accepted in the time-invariant form only.
- The filters predict before each update, so `observations[0]` is scored against `A₀ x₀` and observes `x₁`; to observe the prior at step 0, use a time-varying transition with `A₀ = I` and `Q₀ = 0`.
- `kalman_filter` takes `mask=` with shape `(T,)` to skip whole steps or `(T, M)` to marginalise single channels; masked entries of `observations` are never read, so they may be `NaN`.
- Pass `solver=` (a strategy from skill `gaussx:solvers-and-preconditioners`) to `kalman_filter`, `rts_smoother`, `kalman_gain`, `meanfield_kalman_filter` and the infinite-horizon functions; `parallel_kalman_filter` only uses it with `woodbury_innovation=True`, and warns otherwise.
- `woodbury_innovation=True` builds the innovation covariance as a `LowRankUpdate`, so a structured `R` uses Woodbury solves.
- Do not pass `process_noise` to `rts_smoother`, `parallel_rts_smoother` or `meanfield_rts_smoother`; it is deprecated and ignored.
- For covariances that must be PSD in float32, use `parallel_kalman_filter(..., square_root=True)`; `psd_project=True` only clips the outputs.
- `meanfield_*` needs `block_size` dividing the state size, and the observation size must divide by the number of blocks.
- The infinite-horizon filter has no `mask` and no `init_cov`; its log-likelihood matches `kalman_filter` only when that filter starts from `dare(...).P_inf`.
- Check `DAREResult.converged` when you call `dare` directly; the infinite-horizon functions raise on a non-converged result.
- `naturals_to_ssm` should get `initial_in_q=False` to return `Q` of shape `(N-1, d, d)`; the default is deprecated.

## Goes wrong

- Treating `observations[0]` as observing the prior state gives a one-step offset.
- Using the infinite-horizon filter on gappy data is not supported; use `kalman_filter` with a mask.
- A time-varying `(T, …)` stack mixed with a lineax operator raises `TypeError`.
- `parallel_kalman_filter` with `psd_project=True` does not support a `(T, M)` mask.

## Next

- Filtering with nonlinear dynamics or observations, and CVI sites: [nonlinear-filtering](nonlinear-filtering.md).
- State-space forms of kernels: [sde-kernels](sde-kernels.md).
- GP interpolation at unobserved times: [gp](gp.md).
- The `BlockTriDiag` operator: skill `gaussx:structured-gaussian-linalg`.
