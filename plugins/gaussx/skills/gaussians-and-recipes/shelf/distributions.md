# GMRF fields

Role: ③ model (`role-ladder`, `references/model.md`).
Unit: Gaussians.

The four GMRF fields are equinox modules: a Gaussian with a structured precision, with `log_prob`, `sample` and marginal variances that take the cheapest exact path for that structure. gaussx imports no NumPyro; the NumPyro distribution classes (`MultivariateNormal`, `MultivariateNormalPrecision`, `MarkovGaussian`, the `LGSSM` family) and the `GMRFDistribution` wrapper live in `pyrox_gauss` (package `pyrox-gauss`).
Open this when you need a Gaussian Markov random field with a sparse, block-tridiagonal, Kronecker or dense precision, or an intrinsic (ICAR, random-walk) prior.

## Names

| Name | What it is |
|---|---|
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
P = lx.MatrixLinearOperator(jnp.linalg.inv(S), PSD)  # precision Λ = S⁻¹
prior = gaussx.GaussianMRF(jnp.zeros(3), P)
ref = gaussx.gaussian_log_prob(jnp.zeros(3), lx.MatrixLinearOperator(S, PSD), x)
assert jnp.allclose(prior.log_prob(x), ref, atol=1e-3)

field = prior.condition_on_constraints(jnp.ones(3), 1.0)  # sum x = 1
assert jnp.allclose(field.sample(jr.key(2), (5,)).sum(-1), 1.0, atol=1e-3)

n = 6
icar = gaussx.IntrinsicGMRF(
    jnp.zeros(n), 2.0, gaussx.rw1_structure(n), jnp.ones((n, 1)) / jnp.sqrt(n)
)
u = icar.sample(jr.key(3))
assert jnp.abs(u.sum()) < 1e-4  # the hard constraint 1ᵀ u = 0
```

## Rules

- A `GaussianMRF` precision can be sparse, block-tridiagonal, Kronecker or dense, and it takes the cheapest exact path for each.
- A precision that is singular along a constrained direction (the Laplace posterior of two intrinsic parts) needs `condition_on_constraints(A_c, e, null_space=V)`.
- For a graph with a null space (ICAR), use `IntrinsicGMRF`, or `BYM2GMRF` when the structure depends on the hyperparameters.
- For a Gaussian given by its covariance, use the functions in [gaussians](gaussians.md) (`gaussian_log_prob`, `sample_mvn`) rather than a field.

## Goes wrong

- A GMRF field is not a numpyro distribution: `numpyro.sample("x", field)` raises `TypeError`, so pass `pyrox_gauss.GMRFDistribution(field)`.
- `MultivariateNormal`, `LGSSM` and the other NumPyro classes are no longer gaussx attributes; the `AttributeError` names the `pyrox_gauss` import to use instead.

## Next

- The same Gaussian as plain functions: [gaussians](gaussians.md).
- Filtering and smoothing a state-space chain: [kalman](kalman.md).
- Solver strategies for the `solver` argument: the skill `gaussx:solvers-and-preconditioners`.
