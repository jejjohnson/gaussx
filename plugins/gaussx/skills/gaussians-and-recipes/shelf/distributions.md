# Distribution objects

Role: ③ model (`role-ladder`, `references/model.md`).
Unit: Gaussians.

The normals, `LGSSM` and `MarkovGaussian` exist only when gaussx is installed with its numpyro extra (`pip install "gaussx[numpyro]"`); without it they raise `AttributeError` that points to the extra. The four GMRF fields are equinox modules and need no extra.
Open this when you need a Gaussian as a numpyro distribution (`log_prob`, `sample`, a numpyro site), with a structured covariance, a state-space chain or a Markov random field.

## Names

| Name | What it is |
|---|---|
| `AbstractMultivariateNormal` | The shared base of the two normals below: `covariance_operator`, `precision_operator`, dense `covariance_matrix`, `log_prob`, `kl`. |
| `MultivariateNormal(loc, cov_operator, solver=None)` | A normal parameterised by a covariance operator. `solver` defaults to `AutoSolver()`. |
| `MultivariateNormalPrecision(loc, prec_operator, solver=None)` | The same normal parameterised by a precision operator. |
| `LGSSM(A, H, Q, R, m0, P0, n_steps)` | A linear-Gaussian state-space model as a density over observations of shape `(T, M)`; `log_prob` is the Kalman marginal likelihood. |
| `MaskedLGSSM(..., n_steps, obs_mask)` | An `LGSSM` whose `log_prob` marginalises the unobserved channels given a `(T, M)` boolean mask. |
| `LGSSMFactory(A, H, Q, R, m0, P0, n_steps)` | A callable `obs_mask -> MaskedLGSSM`. |
| `MarkovGaussian(A, Q, mu0, P0, *, b=None)` | A chain-Markov Gaussian over the states `x_{0:T-1}`, event shape `(T, d)`, with `A` and `Q` of shape `(T-1, d, d)`. |
| `GaussianMRF(loc, precision, precision_factors=None, solver=None, ...)` | A Gaussian Markov random field with a structured precision operator; has `condition_on_observations` and `condition_on_constraints`. |
| `ConstrainedGMRF(base, constraint_matrix, constraint_value)` | A `GaussianMRF` conditioned on hard linear constraints; built by `GaussianMRF.condition_on_constraints`. |
| `IntrinsicGMRF(loc, precision_scale, structure, null_space, ...)` | An improper field with precision `τR` for a semi-definite structure `R` (random walk, ICAR). |
| `BYM2GMRF(structure_scaled, tau, phi, null_space, ...)` | The BYM2 pair `(b, u*)` with its exact density; an `IntrinsicGMRF` subclass. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag
S = jnp.array([[2.0, 0.5, 0.0], [0.5, 1.5, 0.2], [0.0, 0.2, 1.0]])  # (3, 3)
x = jnp.array([0.3, -0.2, 0.5])
cov = lx.MatrixLinearOperator(S, PSD)
mvn = gaussx.MultivariateNormal(jnp.zeros(3), cov)
ref = gaussx.gaussian_log_prob(jnp.zeros(3), cov, x)
assert jnp.allclose(mvn.log_prob(x), ref, atol=1e-4)

P = lx.MatrixLinearOperator(jnp.linalg.inv(S), PSD)
prec = gaussx.MultivariateNormalPrecision(jnp.zeros(3), P)
assert jnp.allclose(prec.log_prob(x), mvn.log_prob(x), atol=1e-3)

T, d = 6, 2
A, Q = 0.9 * jnp.eye(d), 0.1 * jnp.eye(d)
At, Qt = jnp.broadcast_to(A, (T - 1, d, d)), jnp.broadcast_to(Q, (T - 1, d, d))
chain = gaussx.MarkovGaussian(At, Qt, jnp.zeros(d), jnp.eye(d))
assert chain.log_prob(chain.sample(jr.key(0), (4,))).shape == (4,)
ssm = gaussx.LGSSM(A, jnp.eye(d), Q, 2 * Q, jnp.zeros(d), jnp.eye(d), n_steps=T)
assert ssm.sample(jr.key(1)).shape == (T, d)

prior = gaussx.GaussianMRF(jnp.zeros(3), lx.MatrixLinearOperator(2.0 * jnp.eye(3), PSD))
field = prior.condition_on_constraints(jnp.ones(3), 1.0)  # sum x = 1
assert jnp.allclose(field.sample(jr.key(2), (5,)).sum(-1), 1.0, atol=1e-3)
```

## Rules

- `MultivariateNormal` takes any `lineax` operator, so a Kronecker or low-rank covariance keeps its structure in `log_prob` and `sample`.
- Choose `MultivariateNormalPrecision` when the natural object is a precision; the covariance is then its lazy inverse.
- `LGSSM` accepts dense arrays or operators for `A`, `H`, `Q` and `R`, while `m0` and `P0` are dense.
- `LGSSM` is a density over observations with the states marginalised; `MarkovGaussian` is the density over the states themselves.
- A `GaussianMRF` precision can be sparse, block-tridiagonal, Kronecker or dense, and it takes the cheapest exact path for each.
- For a graph with a null space (ICAR), use `IntrinsicGMRF`, or `BYM2GMRF` when the structure depends on the hyperparameters.

## Goes wrong

- Without the numpyro extra the normals and the state-space classes are missing; the function forms in [gaussians](gaussians.md) and the GMRF fields still work.
- A GMRF field is not a numpyro distribution: `numpyro.sample("x", field)` raises `TypeError`, so pass `pyrox_gauss.GMRFDistribution(field)`.
- `MaskedLGSSM` names its mask `obs_mask`, not `mask`, because `mask` is numpyro's own method.
- Dense `covariance_matrix` and `precision_matrix` materialise `N x N` arrays; keep them for small problems.

## Next

- The same Gaussian as plain functions: [gaussians](gaussians.md).
- Filtering and smoothing the chain that `LGSSM` describes: [kalman](kalman.md).
- Solver strategies for the `solver` argument: the skill `gaussx:solvers-and-preconditioners`.
