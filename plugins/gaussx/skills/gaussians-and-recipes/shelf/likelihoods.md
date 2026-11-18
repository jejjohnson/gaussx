# Likelihoods

Role: ③ model (`role-ladder`, `references/model.md`).
Unit: quadrature.

Open this when you have observations `y` and a latent vector `f` and need `log p(y | f)`, its site derivatives, or the expected log-likelihood under a Gaussian `q(f)`.
A likelihood object holds `y` and its parameters; the expectation under `q` comes from `expected_log_likelihood`, which uses a closed form when there is one and an [integrator](integrators.md) otherwise.

## Names

Each class is an `equinox` module with `log_prob(f)` returning a scalar and `site_derivatives(f)` returning the gradient and Hessian diagonal in `f`.

| Name | What it is |
|---|---|
| `AbstractLikelihood` | The base class. Subclasses implement `log_prob`; `has_analytical_ell()` and `analytical_expected_log_likelihood(q_mu, q_cov)` are optional. |
| `CallableLikelihood(log_density, data)` | A factorised likelihood from a per-site log density `log_density(f_i, data_i)`. |
| `GaussianLikelihood(y, noise_var)` | `log N(y \| f, noise_var I)`. The only one here with a closed-form expected log-likelihood. |
| `HeteroscedasticGaussianLikelihood(y, latent_dim=2)` | Input-dependent noise: `f` has length `2N`, its first `N` entries the mean and the last `N` the noise, through a softplus. |
| `PoissonLikelihood(y)` | Counts with a log link. |
| `BernoulliLikelihood(y)` | Binary data with a logit link. |
| `BinomialLikelihood(y, n_trials)` | Success counts with a logit link; `n_trials` is a scalar or shape `(N,)`. |
| `NegativeBinomialLikelihood(y, concentration)` | Overdispersed counts with a log link (NB2). |
| `StudentTLikelihood(y, df, scale)` | Robust regression. |
| `SoftmaxLikelihood(y, num_classes)` | Multi-class labels; `f` has length `N * num_classes`. |

## Use it

```python
import jax.numpy as jnp
import jax.scipy.special as jss
import lineax as lx

import gaussx

m, v = jnp.array([0.2, -0.3]), jnp.array([0.3, 0.2])  # q(f) = N(m, diag v)
q = gaussx.GaussianState(m, lx.DiagonalLinearOperator(v))
y = jnp.array([1.0, 2.0])
gh = gaussx.GaussHermiteIntegrator(order=12)  # 144 points in 2-D

lik = gaussx.PoissonLikelihood(y)
ell = gaussx.expected_log_likelihood(lik, q, gh)  # no closed form, so GH is used
exact = jnp.sum(y * m - jnp.exp(m + v / 2) - jss.gammaln(y + 1))
assert jnp.allclose(ell, exact, atol=1e-3)

gauss = gaussx.GaussianLikelihood(y, noise_var=0.5)
assert gauss.has_analytical_ell()
closed = gaussx.expected_log_likelihood(gauss, q)  # no integrator needed
num = gaussx.log_likelihood_expectation(gauss.log_prob, q, gh)
assert jnp.allclose(closed, num, atol=1e-3)
```

## Rules

- Build the likelihood with the data inside it, then pass `lik` and a `GaussianState` to `expected_log_likelihood(lik, state, integrator=None)`.
- A likelihood with a closed form (`GaussianLikelihood`) ignores the integrator; every other class needs one.
- `elbo(lik, state, kl, integrator=None)` is the expected log-likelihood minus a precomputed `kl`.
- The integrator sees the whole vector `f` at once, so a tensor-product rule costs `order ** N` points; see [integrators](integrators.md).
- `site_derivatives` is exact for a likelihood that factorises over sites; the default differentiates `log_prob`.

## Goes wrong

- `expected_log_likelihood` with no integrator and no closed form raises `ValueError`.
- The default `site_derivatives` is not exact for `SoftmaxLikelihood` or `HeteroscedasticGaussianLikelihood`, whose Hessian is not diagonal.
- `SoftmaxLikelihood` and the heteroscedastic class take a flat vector `f` (length `N * num_classes` and `2N`), not a matrix.

## Next

- The rules that evaluate the expectation: [integrators](integrators.md).
- `expected_log_likelihood`, `elbo` and the other expectation functions: [expectations](expectations.md).
- A variational GP that uses a likelihood: [variational-gp](variational-gp.md).
