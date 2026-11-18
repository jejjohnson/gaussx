# Variational GPs

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: Gaussian processes.

Open this when you train a sparse or variational GP: you need the collapsed bound, the uncollapsed ELBO with a KL term, or whitened inducing parameters.
The posterior-prediction and conditioning side is in [gp](gp.md).

## Names

| Name | What it is |
|---|---|
| `collapsed_elbo(y, K_xx_diag, K_xz, K_zz, noise_var, *, jitter=1e-6)` | The Titsias collapsed bound for sparse GP regression, in `O(NM² + M³)`. |
| `variational_elbo_gaussian(y, f_loc, f_var, noise_var, kl)` | The uncollapsed (SVGP) ELBO for a Gaussian likelihood, given the marginals of `q(f)` and the KL term. |
| `variational_elbo_mc(log_likelihood_fn, f_samples, kl)` | A Monte Carlo ELBO for any likelihood, from samples `f_samples` of shape `(S, N)`. |
| `gauss_kl(q_mu, q_sqrt, K=None)` | `KL[q(u) ‖ p(u)]` with `p(u) = 𝒩(0, K)`, or `𝒩(0, I)` when `K` is `None`; `q_sqrt` is full `(R, M, M)` or diagonal `(M, R)`. |
| `whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)` | The whitened SVGP predictive mean and variance, returned as `(f_loc, f_var)`. |
| `svgp_variance_adjustment(K_zz_op, S_u)` | The operator `K_zz⁻¹ S_u K_zz⁻¹ − K_zz⁻¹` that appears in every sparse-GP predictive variance. |
| `unwhiten(m_tilde, L)` | The unwhitened mean `L m̃`. |
| `unwhiten_covariance(L, S_tilde)`, `whiten_covariance(L, S_tilde)` | The unwhitened covariance `L S̃ Lᵀ`; `whiten_covariance` is an old alias of `unwhiten_covariance`, not an inverse. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

x, z = jnp.linspace(0.0, 1.0, 10), jnp.linspace(0.0, 1.0, 4)


def k(a, b):  # (n,), (m,) → (n, m)
    return jnp.exp(-0.5 * ((a[:, None] - b[None, :]) / 0.3) ** 2)


Kzz, Kxz = k(z, z) + 1e-4 * jnp.eye(4), k(x, z)  # (4, 4), (10, 4)
y, s2 = jnp.sin(3.0 * x), 0.1
bound = gaussx.collapsed_elbo(y, jnp.ones(10), Kxz, Kzz, s2, jitter=0.0)
Q = Kxz @ jnp.linalg.solve(Kzz, Kxz.T)
C = Q + s2 * jnp.eye(10)
ref = -0.5 * (y @ jnp.linalg.solve(C, y) + jnp.linalg.slogdet(C)[1])
ref += -5 * jnp.log(2 * jnp.pi) - 0.5 / s2 * jnp.trace(jnp.eye(10) - Q)
assert jnp.allclose(bound, ref, rtol=1e-2, atol=1e-2)
# Whitened q(u) with m = 0 and S = I: the prediction is the prior.
op = lx.MatrixLinearOperator(Kzz, lx.positive_semidefinite_tag)
f_loc, f_var = gaussx.whitened_svgp_predict(
    op, Kxz, jnp.zeros(4), jnp.eye(4), jnp.ones(10)
)
assert jnp.allclose(f_var, 1.0, atol=1e-3) and jnp.allclose(f_loc, 0.0, atol=1e-5)

q_mu = 0.1 * jnp.ones((4, 1))  # (M, R)
kl = gaussx.gauss_kl(q_mu, jnp.ones((4, 1)))  # white prior, diagonal q_sqrt
assert jnp.allclose(kl, 0.5 * jnp.sum(q_mu**2), atol=1e-4)
assert jnp.isfinite(gaussx.variational_elbo_gaussian(y, f_loc, f_var, s2, kl))
```

## Rules

- `collapsed_elbo` needs `K_xx_diag`, the diagonal of the full kernel, not the full matrix.
- `jitter` is added to `K_zz` only when it is an array, a `lineax.MatrixLinearOperator` or a diagonal operator; for other structured operators put the jitter in the operator.
- `solver=` on `collapsed_elbo` and `gauss_kl` is deprecated and ignored, because both use Cholesky factors; a structured `K_zz` (`Kronecker`, `BlockDiag`) is factorised per factor.
- `gauss_kl` takes `q_mu` of shape `(M, R)` and sums the KL over the `R` outputs; leave `K` out for a whitened model.
- In the whitened parameterisation `u_mean` and `u_chol` describe `q` in whitened space, and `whitened_svgp_predict` does the `L_zz` solves itself.
- Pass the KL from `gauss_kl` into `variational_elbo_gaussian` or `variational_elbo_mc`; neither computes it.
- Use `variational_elbo_mc` when the likelihood is not Gaussian; for a quadrature-based expected log-likelihood use [expectations](expectations.md).

## Goes wrong

- `jitter` is not added to a `Kronecker` or `BlockDiag` `K_zz`, so a near-singular factor is not regularised unless you add jitter to each factor.
- Reading `whiten_covariance` as the inverse of `unwhiten_covariance` is wrong, since both compute `L S̃ Lᵀ`.
- Mixing whitened and unwhitened `q_mu` between `gauss_kl` and `whitened_svgp_predict` gives a bound for one model and predictions for another.

## Next

- Posterior mean and variance, conditionals and the cache: [gp](gp.md).
- Expected log-likelihood for non-Gaussian likelihoods: [expectations](expectations.md).
- Likelihood objects: [likelihoods](likelihoods.md).
- Cholesky factors and operators: skill `gaussx:structured-gaussian-linalg`.
