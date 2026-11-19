# Natural-gradient and Newton updates

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: inference and ensembles.

Open this when you update Gaussian natural parameters by damped steps (Bayesian learning rule, Newton, Gauss-Newton, EP cavities) or when you want a Laplace approximation of a latent Gaussian model with a non-Gaussian likelihood.
The caller supplies gradients and Hessians of the log-likelihood; these functions turn them into natural parameters.

## Names

| Name | What it is |
|---|---|
| `damped_natural_update(nat1_old, nat2_old, nat1_target, nat2_target, lr=1.0)` | The linear blend `(1 − lr) old + lr target` of both natural parameters; `nat2` may be an array or an operator. |
| `blr_diag_update(nat1, nat2_diag, grad, hessian_diag, lr, *, convention="expfam")` | One diagonal Bayesian learning rule step, returning `(nat1_new, nat2_new)`. |
| `blr_full_update(nat1, nat2, grad, hessian, lr, *, solver=None, convention="expfam")` | One full-rank Bayesian learning rule step. |
| `newton_update(mean, jacobian, hessian, *, precision_floor=1e-6)` | Converts a Newton step into site natural parameters `(nat1, nat2) = (g − H m, −H)`; `jacobian` is the gradient. |
| `cavity_distribution(post_mean, post_cov, site_nat1, site_nat2, power=1.0, *, precision_floor=None)` | The EP cavity mean and covariance after removing a site, on operators or on `(N,)` arrays. |
| `riemannian_psd_correction(hessian, site_precision, site_covariance, lr=1.0)` | A corrected Hessian that keeps the damped precision update positive definite. |
| `gauss_newton_precision(jacobian, *, base=None)` | The Gauss-Newton precision `JᵀJ`, or `base + JᵀJ` as a `LowRankUpdate` when `base` is given. |
| `ggn_diagonal(jacobian)` | The diagonal of `JᵀJ`, shape `(d,)`. |
| `hutchinson_hessian_diag(hvp_fn, key, d, n_samples=1, dtype=None)` | A stochastic estimate of the Hessian diagonal from Hessian-vector products. |
| `laplace_model(prior, likelihood, *, projector=None, offset=None)`, `laplace_result(model, prior, mode, n_iter, converged)` | The two halves of a Laplace approximation around a Newton loop: the checked `LaplaceModel`, whose `newton(x)` is one step, and the `LaplaceResult` at a mode. |
| `gauss_newton_operator(residual_fn, point, *, noise_precision=None, prior_precision=None, jac='fwd')` | The matrix-free Gauss-Newton precision of a residual function, as a lazy PSD operator. |
| `LaplaceResult` | Holds `mode`, `hessian`, `factor`, `log_marginal`, `n_iter` and `converged`. |
| `vb_mean_correction(result, prior, likelihood, *, subspace, projector=None, offset=None, n_iter=5, integrator=...)` | A low-rank variational correction of the Laplace mean within the span of `subspace`. |
| `log_marginal_likelihood(loc, cov_operator, y, *, solver=None)` | The GP log marginal likelihood `log 𝒩(y ∣ loc, K)`, a wrapper around `gaussian_log_prob`. |
| `gaussian_expected_log_lik(y, q_mu, q_cov, noise, *, solver=None)` | The closed form `E_q[log 𝒩(y ∣ f, R)] = log 𝒩(y ∣ q_mu, R) − ½ tr(R⁻¹ q_cov)`. |
| `trace_correction(K_xx, K_xz, K_zz, *, solver=None)` | The Nyström trace term `tr(K_xx) − tr(K_xzᵀ K_zz⁻¹ K_xz)` of the Titsias collapsed bound. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

# Newton site for a Poisson-like likelihood, then a damped blend with the old site.
g, h, f = jnp.array([0.5, -0.2]), jnp.array([-1.0, -2.0]), jnp.array([0.1, 0.2])
nat1, nat2 = gaussx.newton_update(f, g, h)  # diagonal path: (2,), (2,)
assert jnp.allclose(nat2, -h) and jnp.allclose(nat1, g - h * f)
old1, old2 = gaussx.damped_natural_update(jnp.zeros(2), jnp.ones(2), nat1, nat2, lr=0.5)
assert jnp.allclose(old2, 0.5 * (1.0 + nat2))

# BLR in the precision convention matches the Newton step at lr = 1.
b1, b2 = gaussx.blr_diag_update(
    jnp.zeros(2), jnp.ones(2), g, h, 1.0, convention="precision"
)
assert jnp.allclose(b2, -h) and jnp.allclose(b1, g - h * jnp.zeros(2))

n = 6
prior = gaussx.GaussianMRF(jnp.zeros(n), gaussx.ar1_precision(n, rho=0.5, tau=1.0))
y = jnp.array([0.0, 1.0, 3.0, 2.0, 1.0, 0.0])
model = gaussx.laplace_model(prior, gaussx.PoissonLikelihood(y))
mode = jnp.zeros(n)
for _ in range(20):  # your own Newton loop; argminax.solve.laplace_mode runs it for you
    mode = model.newton(mode)
res = gaussx.laplace_result(model, prior, mode, 20, True)
Q = gaussx.ar1_precision(n, rho=0.5, tau=1.0).as_matrix()
assert res.converged and jnp.allclose(y - jnp.exp(res.mode), Q @ res.mode, atol=1e-3)
```

## Rules

- Every function here works on natural parameters, and two conventions meet: `newton_update` and `cavity_distribution` use `nat2 = +Λ`, whereas the BLR updates default to the exponential-family `η₂ = −Λ/2`.
- Pass `convention="precision"` to `blr_diag_update` or `blr_full_update` to read and return `+Λ`; `nat1` is `Λ μ` in both.
- `damped_natural_update` is convention-agnostic, but both arguments must share one convention or the sign flips and the precision halves.
- `grad` and `hessian` are derivatives of the log-likelihood, so a log-concave site has a negative Hessian; `newton_update`'s `jacobian` argument is a gradient despite its name.
- `newton_update` and `cavity_distribution` floor the precision only on the `(N,)` diagonal path, through `precision_floor`; the full-matrix path does not floor.
- `blr_full_update` accepts `solver=`; pass a strategy from skill `gaussx:solvers-and-preconditioners` to change how it solves for the mean.
- Use `gauss_newton_precision(J, base=prior_precision)` when you will solve against the result, so `solve` and `logdet` use the Woodbury identity.
- `laplace_model` needs a `GaussianMRF` or `IntrinsicGMRF` prior and a site-factorising likelihood that holds `y`.
- Check `LaplaceResult.converged`, because implicit gradients through the mode assume it.
- `vb_mean_correction` takes the same `prior`, `likelihood`, `projector` and `offset` that you gave `laplace_model`; for a Gaussian likelihood the correction is zero.

## Goes wrong

- Mixing `nat2` conventions between a BLR update and `damped_natural_update` gives wrong precisions without an error.
- `gauss_newton_precision(J)` without `base` is singular when `D_obs < D_latent`, so `solve` and `logdet` on it return `NaN`.
- Adding a prior with `gauss_newton_precision(J) + prior` gives a plain lineax sum that is solved densely; pass `base=prior` instead.
- A positive Hessian entry from a non-log-concave site gives a negative precision on the full-matrix `newton_update` path.
- `laplace_model` raises `TypeError` for a prior class other than `GaussianMRF` or `IntrinsicGMRF`.

## Next

- Moments and site updates from quadrature: [expectations](expectations.md).
- The collapsed bound that uses `trace_correction`: [variational-gp](variational-gp.md).
- Sites in time series: [nonlinear-filtering](nonlinear-filtering.md).
- Gaussian and exponential-family parameter conversions: [expfam](expfam.md).
- Likelihood objects: [likelihoods](likelihoods.md).
