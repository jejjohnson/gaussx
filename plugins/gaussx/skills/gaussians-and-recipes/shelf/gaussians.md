# Gaussians as functions

Role: ③ model (`role-ladder`, `references/model.md`).
Unit: Gaussians.

Open this when you need the density, entropy, KL, conditioning or a sample of a Gaussian whose covariance is a `lineax` operator.
Each is a plain function of `(loc, cov_operator)`; no distribution object is needed (for one, see [distributions](distributions.md)).

## Names

| Name | What it is |
|---|---|
| `gaussian_log_prob(loc, cov_operator, value, *, solver=None, key=None)` | `log N(value \| loc, Σ)` as a scalar. Dispatches on the structure of `Σ`. |
| `gaussian_entropy(cov_operator, *, solver=None, key=None)` | The entropy `½(N(1 + log 2π) + log \|Σ\|)`, independent of the mean. |
| `gaussian_kl(p_loc, p_cov, q_loc, q_cov)` | `KL(p ‖ q)` between two Gaussians with operator covariances. The general form. |
| `kl_standard_normal(m, S, *, solver=None, key=None)` | `KL(N(m, S) ‖ N(0, I))`, the prior term of a variational bound, with no inversion. |
| `quadratic_form(operator, x, *, solver=None)` | `xᵀ A⁻¹ x` from one solve. |
| `conditional(loc, cov, obs_idx, obs_values, *, solver=None)` | `p(x_A \| x_B = b)` from a joint. Returns `(mean, cov_operator)` over the unobserved indices. |
| `linear_gaussian_update(prior_mean, prior_cov, obs_operator, obs_cov, obs, *, solver=None)` | The matrix-free posterior of a linear-Gaussian observation, as a `LinearGaussianPosterior(mean, cov)` whose covariance is a lazy operator. |
| `information_diagnostics(jacobian, noise_cov, prior_cov)` | The averaging kernel, degrees of freedom for signal, information content, posterior covariance and retrieval-noise covariance of a linear retrieval, as an `InformationDiagnostics` of dense arrays. |
| `sample_mvn(mean, covariance, *, key, num_samples=1)` | Draws from `𝒩(mean, K)` with a factor chosen from the structure of `K`. Shape `(num_samples, *batch, N)`. |
| `sample_joint_conditional(joint_mean, joint_covariance, *, key, observed_index=1, observed_value=None, num_samples=1, solver=None)` | Joint draws of a partitioned Gaussian, and conditional draws when `observed_value` is given. Returns a dict. |
| `add_jitter(operator, jitter=1e-6)` | `A + ε I` as an operator. |
| `project(K_XZ, L_Z)` | `K_XZ K_ZZ⁻¹` from the lower Cholesky factor `L_Z` of `K_ZZ`. |
| `joseph_update(P_pred, K, H, R)` | The Joseph-form covariance update `(I − KH) P (I − KH)ᵀ + K R Kᵀ`, symmetrised. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jst
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag
A = jr.normal(jr.key(0), (4, 4))
S = A @ A.T + jnp.eye(4)  # (4, 4)
cov = lx.MatrixLinearOperator(S, PSD)
loc, x = jnp.zeros(4), jnp.array([0.3, -0.2, 0.5, 0.1])

lp = gaussx.gaussian_log_prob(loc, cov, x)
assert jnp.allclose(lp, jst.multivariate_normal.logpdf(x, loc, S), atol=1e-3)

eye = lx.MatrixLinearOperator(jnp.eye(4), PSD)
kl = gaussx.gaussian_kl(loc + 0.1, cov, loc, eye)
assert jnp.allclose(kl, gaussx.kl_standard_normal(loc + 0.1, cov), atol=1e-3)

idx, b = jnp.array([1, 3]), jnp.array([0.5, -0.5])  # observe x[1], x[3]
mu_c, cov_c = gaussx.conditional(loc, cov, idx, b)  # over x[0], x[2]
gain = S[jnp.ix_(jnp.array([0, 2]), idx)] @ jnp.linalg.inv(S[jnp.ix_(idx, idx)])
assert jnp.allclose(mu_c, gain @ b, atol=1e-3)

draws = gaussx.sample_mvn(loc, cov, key=jr.key(1), num_samples=5)  # (5, 4)
assert draws.shape == (5, 4)
```

## Rules

- Pass the covariance as a `lineax` operator and tag it positive semidefinite, so the solve and log-determinant take the structured path.
- `conditional` takes integer indices and returns the mean and covariance of the remaining indices in ascending index order.
- `sample_mvn` needs `key=` as a keyword; the number of draws is `num_samples`, and a batch of means gets independent noise.
- Use `gaussian_kl` for a general pair and `kl_standard_normal` for a standard normal prior.
- For a stochastic log-determinant, pass a strategy as `solver` and a `key` for its probes; see the skill `gaussx:solvers-and-preconditioners`.

## Goes wrong

- The covariance argument is an operator, not an array; wrap a dense matrix in `lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)`.
- Duplicate or out-of-range `obs_idx` raises `ValueError` when the indices are concrete.
- `kl_divergence` on `GaussianExpFam` is a different function; see [expfam](expfam.md).

## Next

- A distribution object with `log_prob` and `sample`: [distributions](distributions.md).
- Natural parameters and the matching KL: [expfam](expfam.md).
- A GP posterior built from these pieces: [gp](gp.md).
- Operators and the `solve` and `logdet` verbs: the skill `gaussx:structured-gaussian-linalg`.
