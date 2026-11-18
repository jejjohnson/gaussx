# Expectations and moment matching

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: quadrature.

Open this when you need a Gaussian expectation of a function: an expected log-likelihood, EP tilted moments, a moment-matched linear surrogate, psi statistics, or a GP prediction at uncertain inputs.
The integrator you pass decides the numerics; the integrators themselves are in [integrators](integrators.md).

## Names

All of these take a `gaussx.GaussianState(mean, cov)`, whose `cov` is a lineax operator, except where the shape says otherwise.

| Name | What it is |
|---|---|
| `expected_log_likelihood(likelihood, state, integrator=None)` | `E_q[log p(y ∣ f)]`; uses the closed form when the likelihood has one, otherwise needs an `integrator`. |
| `elbo(likelihood, state, kl, integrator=None)` | `expected_log_likelihood(...) − kl`. |
| `log_likelihood_expectation(likelihood_fn, state, integrator)` | `E[log p]` of a plain function `(N,) -> ()`, by quadrature. |
| `mean_expectation(fn, state, integrator)` | `E[f(x)]` for `x ∼ 𝒩(μ, Σ)`, shape `(M,)`. |
| `gradient_expectation(fn, state, integrator)` | `E[∇f(x)]` by Stein's lemma. |
| `cost_expectation(prediction_fn, cost_fn, state, target, integrator)` | `E[Cost(f(x), target)]`. |
| `moment_transform(fn, mean, cov, *, integrator=None)` | The moment triple `(E[g], Cov[g], Cov[x, g])` on dense arrays. |
| `moment_match(log_lik_fn, state, integrator, *, power=1.0)` | The EP tilted log-normaliser and its first two derivatives in the cavity mean, as a `MomentMatchResult`. |
| `MomentMatchResult(log_Z, d_log_Z, d2_log_Z)` | The result of `moment_match`. |
| `ep_tilted_moments(log_lik_fn, cav_mean, cav_var, *, order=None, integrator=None, power=1.0)` | The scalar EP tilted mean and variance, elementwise over a batch of cavities. |
| `statistical_linear_regression(conditional_mean_fn, conditional_var_fn, state, integrator)` | A linear-Gaussian surrogate `𝒩(y ∣ A f + b, Ω)` of a nonlinear conditional, as an `SLRResult`. |
| `SLRResult(A, b, omega, mu)` | The gain, offset, residual covariance and expected conditional mean. |
| `AssumedDensityFilter(n_samples=5000, ...)` | A Monte Carlo integrator that projects onto a Gaussian by matching two moments. |
| `kernel_expectations(kernel_fn, state, X_train, integrator)` | The numerical `(Ψ₀, Ψ₁, Ψ₂)` of a kernel function. |
| `compute_psi_statistics(kernel, state, X_train, *, integrator=None)` | `(Ψ₀, Ψ₁, Ψ₂)`, using closed forms when the kernel implements `AnalyticalPsiStatistics`. |
| `AnalyticalPsiStatistics` | A protocol with `psi0(state)`, `psi1(state, X_train)` and `psi2(state, X_train)`. |
| `uncertain_gp_predict(kernel_fn, X_train, alpha, K_inv, state, integrator)` | A GP predictive mean and variance at a Gaussian input. |
| `uncertain_svgp_predict(kernel_fn, Z, alpha, Q, state, integrator)`, `uncertain_vgp_predict(kernel_fn, X_train, alpha, Q, state, integrator)` | The same for a sparse or dense variational GP, with the variance-adjustment operator `Q`. |
| `uncertain_bgplvm_predict(kernel_fn, X_train, alpha, K_inv, state, integrator)` | The multi-output version, with `alpha` of shape `(N_train, D_out)`. |
| `uncertain_gp_predict_mc(predict_fn, state, n_particles=100, key=None)` | A Monte Carlo prediction by the law of total variance. |
| `theta_design(log_post, mode, *, method=None, ...)` | Integration points and log-weights over hyperparameters, with `"eb"`, `"grid"` or `"ccd"` designs. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

mu, v, y, s2 = jnp.array([0.3, -0.2]), jnp.array([0.5, 0.2]), jnp.array([1.0, 0.0]), 0.4
state = gaussx.GaussianState(mu, lx.DiagonalLinearOperator(v))
lik = gaussx.GaussianLikelihood(y, s2)

ell = gaussx.expected_log_likelihood(lik, state)  # closed form, no integrator
gh = gaussx.GaussHermiteIntegrator(order=5)
ell_q = gaussx.log_likelihood_expectation(lik.log_prob, state, gh)
assert jnp.allclose(ell, ell_q, atol=1e-3)
assert jnp.allclose(gaussx.elbo(lik, state, jnp.array(0.2)), ell - 0.2, atol=1e-5)

# Tilting 𝒩(f ∣ m, v) by a Gaussian likelihood in f gives a Gaussian.
lik1 = lambda f: -0.5 * (f - 1.0) ** 2 / s2
tm, tv = gaussx.ep_tilted_moments(lik1, mu[0], v[0])
prec = 1 / v[0] + 1 / s2
assert jnp.allclose(tv, 1 / prec, atol=1e-4)
assert jnp.allclose(tm, (mu[0] / v[0] + 1.0 / s2) / prec, atol=1e-4)

state1 = gaussx.GaussianState(mu[:1], lx.DiagonalLinearOperator(v[:1]))
mm = gaussx.moment_match(lambda f: lik1(f[0]), state1, gaussx.GaussHermiteIntegrator())
assert jnp.allclose(mm.d_log_Z, (1.0 - mu[0]) / (v[0] + s2), atol=1e-3)
```

## Rules

- `expected_log_likelihood` ignores `integrator` when the likelihood has a closed form (for example `GaussianLikelihood`) and raises `ValueError` when it has none and no integrator is given.
- Most functions here take `integrator` as a required positional argument; `GaussHermiteIntegrator` costs `order**dim` points, so prefer a cubature rule above two dimensions.
- `moment_match`, `statistical_linear_regression` and the `ep_tilted_moments` integrator need a point-based rule and raise `NotImplementedError` otherwise.
- `moment_match` computes the EP quantity, not `E_q[log p(y ∣ f)]`; use `expected_log_likelihood` for variational inference.
- `moment_match` returns derivatives in the cavity mean, and the EP site is not `(g, −H)`; its docstring gives the conversion and says to divide by `power` to recover the site for `p(y ∣ f)`.
- `ep_tilted_moments` passes a cavity with `cav_var <= 0` through unchanged, so skip or damp such sites yourself.
- Give `ep_tilted_moments` either `order` or `integrator`, not both.
- `moment_transform` defaults to `UnscentedIntegrator(alpha=1.0)` because the class default `alpha=1e-3` is unsafe in float32.
- `compute_psi_statistics` raises `ValueError` if the kernel has no closed form and no `integrator` is given.
- In `uncertain_vgp_predict` and `uncertain_svgp_predict`, `Q = K⁻¹ S K⁻¹ − K⁻¹`; `Q = −K⁻¹` recovers the exact GP, and `svgp_variance_adjustment` builds `Q` (see [variational-gp](variational-gp.md)).

## Goes wrong

- Calling `expected_log_likelihood` on a non-conjugate likelihood without an integrator raises `ValueError`.
- A negative-weight rule can make the tilted normaliser `Z` non-positive in `moment_match`; prefer a rule whose points cover the likelihood's mass.
- Feeding `(d_log_Z, −d2_log_Z)` of `moment_match` in as the EP site drops the `−H m` term and the `(I + H C)⁻¹` factor.
- A full-order `GaussHermiteIntegrator` in more than three dimensions evaluates the integrand at a very large number of points.

## Next

- Integrator choices and their point counts: [integrators](integrators.md).
- Likelihood objects: [likelihoods](likelihoods.md).
- ELBOs, the KL term and whitening: [variational-gp](variational-gp.md).
- Site updates from these moments: [natural-gradient](natural-gradient.md), [nonlinear-filtering](nonlinear-filtering.md).
