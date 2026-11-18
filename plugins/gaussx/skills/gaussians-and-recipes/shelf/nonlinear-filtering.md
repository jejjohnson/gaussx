# Nonlinear filtering and sites

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: state space.

Open this when the dynamics or the observation map is nonlinear, when you need per-time Gaussian sites and CVI or EP site updates, or when you want the precision-form (SpInGP) posterior of a Markov prior.
For linear models use [kalman](kalman.md).

## Names

| Name | What it is |
|---|---|
| `nonlinear_kalman_filter(dynamics, obs_fn, process_noise, obs_noise, observations, init_mean, init_cov, *, integrator=None, mask=None, joseph=True, solver=None, validate=None)` | A moment-matched filter, returned as a `FilterState`; the integrator chooses EKF, UKF, CKF or Gauss-Hermite. |
| `nonlinear_rts_smoother(filter_state, dynamics, *, integrator=None)` | The matching moment-matched backward pass; returns `(smoothed_means, smoothed_covs)`. |
| `nonlinear_kalman_predict(dynamics, mean, cov, process_noise, *, integrator=None)` | One predict step, returning `(mean, cov)`; use it to drive your own loop. |
| `nonlinear_kalman_update(obs_fn, mean, cov, observation, obs_noise, *, integrator=None, mask=None, joseph=True)` | One update step, returning `(mean, cov, log_likelihood_increment)`. |
| `nonlinear_rts_step(dynamics, mean_filtered, cov_filtered, mean_predicted, cov_predicted, mean_smoothed, cov_smoothed, *, integrator=None)` | One backward smoothing step, returning `(mean, cov)`. |
| `GaussianSites(nat1, nat2)` | Per-time Gaussian likelihood sites in natural parameters, with `nat2 = −½ Λ`. |
| `cvi_update_sites(sites, grad_nat1, grad_nat2, rho)` | The damped update `(1 − ρ) θ + ρ ∇` of every site. |
| `sites_to_precision(sites)` | A block-diagonal `BlockTriDiag` of the site precisions, to add to a prior precision. |
| `site_natural_from_tilted(tilted_mean, tilted_var, cav_mean, cav_var)` | Scalar site natural parameters `(nat1, nat2)` from tilted and cavity moments. |
| `site_mean_var_from_natural(site_nat1, site_nat2)` | The mean and variance of a scalar site from its natural parameters. |
| `cavity_from_marginal(marg_mean, marg_var, site_nat1, site_nat2)` | The cavity mean and variance after removing a site from a marginal. |
| `spingp_posterior(prior_precision, emission_model, obs_noise, observations, *, prior_mean=None, solver=None)` | The posterior mean and `BlockTriDiag` precision of a Markov prior with linear observations. |
| `spingp_log_likelihood(prior_precision, emission_model, obs_noise, observations, *, solver=None)` | The log marginal likelihood from the same precision-form computation. |
| `sigma_point_kalman_update(points, evaluations, mean, cov, observation, obs_noise, *, mean_weights, cov_weights=None, joseph=True, solver=None, validate=False)` | One moment-matched update from sigma-point evaluations the caller already has; returns `(mean, covariance, log_likelihood)`. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
H, Q, R = jnp.array([[1.0, 0.0]]), 0.01 * jnp.eye(2), 0.1 * jnp.eye(1)
y = jnp.cumsum(0.1 * jr.normal(jr.key(0), (12, 1)), axis=0)  # (T, M)
m0, P0 = jnp.zeros(2), jnp.eye(2)

# With affine maps the default integrator reproduces the linear filter.
f = gaussx.nonlinear_kalman_filter(lambda x: A @ x, lambda x: H @ x, Q, R, y, m0, P0)
lin = gaussx.kalman_filter(A, H, Q, R, y, m0, P0)
assert jnp.allclose(f.filtered_means, lin.filtered_means, atol=1e-3)
assert jnp.allclose(f.log_likelihood, lin.log_likelihood, atol=1e-2)
sm, sP = gaussx.nonlinear_rts_smoother(f, lambda x: A @ x)  # (12, 2), (12, 2, 2)
assert sm.shape == (12, 2)

sites = gaussx.GaussianSites(
    jnp.zeros((12, 2)), -0.5 * jnp.ones((12, 2, 2)) * jnp.eye(2)
)
sites = gaussx.cvi_update_sites(sites, jnp.ones((12, 2)), sites.nat2, rho=0.5)
assert jnp.allclose(sites.nat1, 0.5)

prec = gaussx.BlockTriDiag(
    jnp.broadcast_to(2.0 * jnp.eye(2), (12, 2, 2)),
    jnp.broadcast_to(-0.5 * jnp.eye(2), (11, 2, 2)),
)
mean, post = gaussx.spingp_posterior(prec, H, lx.MatrixLinearOperator(R), y)  # (24,)
dense = prec.as_matrix() + jnp.kron(jnp.eye(12), H.T @ H / 0.1)
rhs = (y @ H / 0.1).reshape(-1)
assert jnp.allclose(mean, jnp.linalg.solve(dense, rhs), atol=1e-3)
```

## Rules

- The integrator is the choice of filter: `TaylorIntegrator` gives the EKF, `UnscentedIntegrator` the UKF, `CubatureIntegrator` the CKF, `GaussHermiteIntegrator` the Gauss-Hermite filter, and `MonteCarloIntegrator` a Monte Carlo filter; see [integrators](integrators.md).
- Pass the same `dynamics` and `integrator` to `nonlinear_rts_smoother` that you passed to the filter.
- The default is `UnscentedIntegrator(alpha=1.0)`, not the class's own `alpha=1e-3`, because the small value costs about seven digits in float32.
- `dynamics` and `obs_fn` are plain callables from `(N,)` to `(N,)` and `(M,)`, and process noise is additive.
- The filter predicts first, so `observations[0]` observes `x₁`; to score a first observation against the prior, call `nonlinear_kalman_update` on the prior and filter the rest.
- `log_likelihood` is a moment-matched surrogate, exact only for affine maps.
- `validate=None` checks positive semi-definiteness only when the integrator does not guarantee it; a negative-weight rule can make the moment triple inconsistent and raise.
- `mask=` takes shape `(T,)` or `(T, M)`; masked channels are marginalised out exactly and may be `NaN`.
- `process_noise` passed to `nonlinear_rts_smoother` is deprecated and ignored, and its `solver=` is unused.
- Site natural parameters use `nat2 = −½ Λ`, whereas `site_natural_from_tilted`, `site_mean_var_from_natural` and `cavity_from_marginal` work with the precision itself, `nat2 = Λ`; multiply an exponential-family `nat2` by `−2` before passing it to them.
- `spingp_*` take `solver=` (see skill `gaussx:solvers-and-preconditioners`) for the posterior precision solve; `obs_noise` must be a lineax operator.
- `emission_model` is `(d_obs, d)` when shared or `(N, d_obs, d)` per time step.

## Goes wrong

- Using the class default `alpha=1e-3` for `UnscentedIntegrator` in float32 misplaces the log-likelihood even on a linear problem.
- A `rho` outside `[0, 1]` in `cvi_update_sites` is not a damped update.
- `site_mean_var_from_natural` returns a negative variance when `site_nat2` is not positive.
- A singular predicted covariance makes the smoother gain a least-squares solve, so `solver=` has no effect.

## Next

- Linear filters, smoothers and parameter conversions: [kalman](kalman.md).
- Integrators and moment transforms: [integrators](integrators.md).
- Expected log-likelihood and EP moments for site updates: [expectations](expectations.md).
- Natural-gradient and Newton site updates: [natural-gradient](natural-gradient.md).
