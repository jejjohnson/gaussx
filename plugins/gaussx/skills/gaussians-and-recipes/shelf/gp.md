# GP conditioning and prediction

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: Gaussian processes.

Open this when you have a GP kernel matrix and need the posterior mean and variance at test points, a leave-one-out score, or a structured-grid fit.
Build the training covariance as an operator first (skill `gaussx:structured-gaussian-linalg`), then solve once with a cache and predict many times.

## Names

| Name | What it is |
|---|---|
| `build_prediction_cache(operator, y, *, solver=None)` | Solves `K_y α = y` once and returns a `PredictionCache`. |
| `PredictionCache(alpha, operator=None, factor=None)` | The cached weights, the operator, and a Cholesky factor where one is cheap to keep. |
| `predict_mean(cache, K_cross)` | The predictive mean `K_cross @ alpha`, shape `(Nt,)`. |
| `predict_variance(cache, K_cross, K_test_diag)` | The predictive variance `k_** − diag(K_*f K_y⁻¹ K_f*)`, shape `(Nt,)`. |
| `love_cache(K_op, lanczos_order=50, key=None, *, initial_vector=None)` | A rank-`k` Lanczos approximation of `K⁻¹`, returned as a `LOVECache`. |
| `LOVECache(Q, inv_eigvals)` | The Ritz vectors and inverse Ritz values stored by `love_cache`. |
| `love_variance(cache, K_star_row)` | An `O(Nk)` approximation of `k_*ᵀ K⁻¹ k_*` for one test point. |
| `love_residual(cache, K_op, K_star_row)`, `love_variance_error_bound(cache, K_op, K_star_row, lambda_min)` | Convergence checks for a LOVE cache at one cross-covariance. |
| `leave_one_out_cv(operator, y, *, solver=None, diag_inv_method="solve", ...)` | Leave-one-out means, variances and log-likelihood without refitting, returned as a `LOOResult`. |
| `LOOResult` | Holds `loo_log_likelihood`, `loo_means` and `loo_variances`. |
| `kronecker_mll(K_factors, y, noise_var, grid_shape)` | The exact marginal log-likelihood of a Kronecker GP by per-factor eigendecomposition. |
| `kronecker_posterior_predictive(K_factors, y, noise_var, grid_shape, K_cross_factors, *, K_test_diag_factors)` | The posterior mean and variance of a Kronecker GP at test points. |
| `sparse_conditional(K_zz, K_xz, K_xx, q_mu, *, q_sqrt=None, white=False)` | The sparse-GP conditional `q(f_x)`, returning `(mean, var)`. |
| `base_conditional(K_mm, K_mn, K_nn, f, ...)` | Deprecated; use `sparse_conditional` with `K_xz = K_mn.T`. |
| `matheron_update(prior_sample_target, prior_sample_conditioning, observed_value, cross_covariance, conditioning_covariance, *, solver=None)` | Posterior samples from joint prior draws by Matheron's rule. |
| `oilmm_project(Y, W, noise_var)`, `oilmm_back_project(f_means, f_vars, W)` | Project multi-output data onto independent latent GPs with orthonormal `W`, and map latent predictions back. |
| `conditional_interpolate(A_fwd, Q_fwd, A_bwd, Q_bwd, mu_prev, P_prev, mu_next, P_next)` | A two-filter fusion at an unobserved time from a filtered state and a backward message. |
| `rts_interpolate(A_fwd, Q_fwd, A_bwd, Q_bwd, filtered_mean_prev, filtered_cov_prev, smoothed_mean_next, smoothed_cov_next)` | The smoothed marginal at an unobserved time from a filtered and a smoothed state. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

x, xs = jnp.linspace(0.0, 1.0, 8), jnp.linspace(0.1, 0.9, 3)


def k(a, b):  # (n,), (m,) → (n, m)
    return jnp.exp(-0.5 * ((a[:, None] - b[None, :]) / 0.3) ** 2)


Ky = k(x, x) + 0.1 * jnp.eye(8)  # (8, 8)
op = lx.MatrixLinearOperator(Ky, lx.positive_semidefinite_tag)
y = jnp.sin(3.0 * x) + 0.1 * jr.normal(jr.key(0), (8,))
Ks = k(xs, x)  # (3, 8)
cache = gaussx.build_prediction_cache(op, y)
mean = gaussx.predict_mean(cache, Ks)  # (3,)
var = gaussx.predict_variance(cache, Ks, jnp.ones(3))  # (3,)
assert jnp.allclose(mean, Ks @ jnp.linalg.solve(Ky, y), atol=1e-3)
assert jnp.allclose(var, 1 - jnp.diag(Ks @ jnp.linalg.solve(Ky, Ks.T)), atol=1e-3)
love = gaussx.love_cache(op, lanczos_order=8)  # full order is exact
q = gaussx.love_variance(love, Ks[0])
assert jnp.allclose(q, Ks[0] @ jnp.linalg.solve(Ky, Ks[0]), atol=1e-2)
loo = gaussx.leave_one_out_cv(op, y)
assert loo.loo_means.shape == (8,) and jnp.isfinite(loo.loo_log_likelihood)
```

## Rules

- Rebuild the `PredictionCache` whenever the kernel hyperparameters change, because it is tied to the operator it came from.
- `predict_variance` takes the cache first, then `K_cross`, then `K_test_diag`; the old order with `operator=` is deprecated.
- `build_prediction_cache` and `leave_one_out_cv` accept `solver=`; pass a strategy from skill `gaussx:solvers-and-preconditioners` to change the numerics, for example an iterative one for a large operator.
- A dense, diagonal, `Kronecker` or `BlockDiag` PSD operator with the default or a dense solver stores a Cholesky factor, so the variance needs only triangular solves.
- `love_variance` is a lower bound on `k_*ᵀ K⁻¹ k_*`, so `k_** − love_variance(...)` is an upper bound on the predictive variance; it is exact at `lanczos_order = N`.
- Choose `lanczos_order` by the kernel's effective rank at the noise level, and check it with `love_residual` or `love_variance_error_bound` on representative test points.
- `oilmm_project` needs `WᵀW = I` and does not check it unless you pass `check_orthonormal=True`.
- Use `sparse_conditional(K_zz, K_xz, ...)` with `K_xz` of shape `(N, M)`; `white=True` means `q_mu` and `q_sqrt` are in whitened space.

## Goes wrong

- Reusing a cache after changing the kernel gives predictions from the old `K_y`.
- A `PredictionCache` built from `alpha` alone has no operator, so `predict_variance` raises `ValueError`.
- Passing two smoothed marginals to `conditional_interpolate` counts the data twice and makes the covariance too small; use `rts_interpolate` for a filtered and a smoothed state.
- A non-orthonormal `W` in `oilmm_project` gives wrong latent data and noise without an error.
- A small `lanczos_order` on a smooth kernel with small noise leaves the variance too high.

## Next

- ELBOs and whitening for sparse variational GPs: [variational-gp](variational-gp.md).
- Time-series GPs as state-space models: [kalman](kalman.md).
- Matérn and other SDE kernels: [sde-kernels](sde-kernels.md).
- The operators and the `solve` and `logdet` verbs: skill `gaussx:structured-gaussian-linalg`.
