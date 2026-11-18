# Exponential-family forms

Role: ③ model (`role-ladder`, `references/model.md`).
Unit: Gaussians.

Open this when you need a Gaussian in natural or expectation parameters, for natural-gradient updates, site updates or a KL in natural coordinates.
The conversions come in an operator form and a dense Cholesky form; the three parameterisations are mean and covariance, natural, and expectation.

## Names

Natural parameters are `eta1 = Λ μ` and `eta2 = -½ Λ` for the precision `Λ`; expectation parameters are `m1 = μ` and `m2 = μ μᵀ + Σ`.

| Name | What it is |
|---|---|
| `GaussianExpFam(eta1, eta2)` | A Gaussian in natural parameters, with `eta2` an operator. Build it with `GaussianExpFam.from_mean_cov(mu, Sigma)` or `from_mean_prec(mu, Lambda)`. |
| `mean_cov_to_natural(mu, Sigma, *, solver=None)` | Operator form: returns `(eta1, eta2)`. |
| `natural_to_mean_cov(eta1, eta2, *, solver=None)` | Operator form: returns `(mu, Sigma)` with `Sigma` an operator. |
| `mean_chol_to_natural(mu, S_sqrt)` | Dense form with a lower Cholesky factor `S_sqrt`; batch axes allowed. |
| `natural_to_mean_chol(eta1, eta2, *, solver=None)` | Dense form: returns `(mu, S_sqrt)`. |
| `mean_chol_to_expectation(mu, S_sqrt)` | Returns `(m1, m2)`. |
| `expectation_to_mean_chol(m1, m2)` | Returns `(mu, S_sqrt)`. |
| `natural_to_expectation(eta1, eta2, *, solver=None)` | Dense arrays in, `(m1, m2)` out. |
| `expectation_to_natural(m1, m2, *, solver=None)` | Dense arrays in, `(eta1, eta2)` out. |
| `to_mean_cov(expfam)` | A `GaussianExpFam` as `(mu, Sigma)`. Not the expectation parameters. |
| `log_partition(expfam)` | The log-partition `A(η)`, which includes the `N/2 log 2π` normaliser. |
| `fisher_info(expfam)` | The precision `-2 eta2`: the Fisher information of the mean, not in natural coordinates. |
| `sufficient_stats(x)` | `T(x) = [x, x xᵀ]`. |
| `kl_divergence(q, p)` | `KL(q ‖ p)` between two `GaussianExpFam`, by the Bregman form. |
| `to_natural`, `to_expectation` | Deprecated aliases of `mean_cov_to_natural` and `to_mean_cov`; `to_expectation` returns `(mu, Sigma)`, not expectation parameters. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag
mu = jnp.array([0.5, -1.0])
S = jnp.array([[1.5, 0.4], [0.4, 0.8]])  # (2, 2)
cov = lx.MatrixLinearOperator(S, PSD)

eta1, eta2 = gaussx.mean_cov_to_natural(mu, cov)
assert jnp.allclose(eta1, jnp.linalg.solve(S, mu), atol=1e-3)
mu2, S2 = gaussx.natural_to_mean_cov(eta1, eta2)
assert jnp.allclose(S2.as_matrix(), S, atol=1e-3)

L = jnp.linalg.cholesky(S)
m1, m2 = gaussx.mean_chol_to_expectation(mu, L)  # m2 = mu mu^T + S
assert jnp.allclose(m2, jnp.outer(mu, mu) + S, atol=1e-4)

q = gaussx.GaussianExpFam.from_mean_cov(mu, cov)
p = gaussx.GaussianExpFam.from_mean_cov(
    jnp.zeros(2), lx.MatrixLinearOperator(jnp.eye(2), PSD)
)
kl = gaussx.kl_divergence(q, p)
assert jnp.allclose(kl, gaussx.kl_standard_normal(mu, cov), atol=1e-3)
```

## Rules

- Use the operator functions when `Sigma` is structured, and the `*_chol_*` and `*_expectation` functions for dense, batched arrays.
- The dense forms take a Cholesky factor `S_sqrt`, not a covariance.
- A `GaussianExpFam` holds `eta2 = -½ Λ`, so `-2 * eta2` is the precision.
- `kl_divergence(q, p)` is `KL(q ‖ p)`: the first argument is the distribution the expectation is taken under.
- Use `gaussian_kl` from [gaussians](gaussians.md) for the mean and covariance form.

## Goes wrong

- `to_expectation` is a deprecated alias that returns the mean and covariance; use `to_mean_cov`, or `natural_to_expectation` for `(m1, m2)`.
- `fisher_info` is the Fisher information of the mean only; it is not `∇²A(η)`.
- Operator and dense forms are separate functions; `natural_to_mean_cov` returns an operator, `natural_to_mean_chol` returns a Cholesky factor array.

## Next

- Natural-gradient steps that use these forms: [natural-gradient](natural-gradient.md).
- KL and density as plain functions: [gaussians](gaussians.md).
- Operators for `Sigma` and `eta2`: the skill `gaussx:structured-gaussian-linalg`.
