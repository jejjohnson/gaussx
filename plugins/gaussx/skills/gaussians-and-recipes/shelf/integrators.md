# Integrators

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: quadrature.

Open this when you need `E[g(x)]`, the output covariance or the cross-covariance of a nonlinear `g` under a Gaussian `x`, and you want to choose the rule.
Every integrator has the same call, `integrator.integrate(fn, state)`, so a likelihood, a filter or a linearisation can take any of them.

## Names

| Name | What it is |
|---|---|
| `AbstractIntegrator` | The protocol: `integrate(fn, state)`, `guarantees_psd(dim)` and, for point rules, `points_and_weights(state)`. |
| `GaussianState(mean, cov)` | The input Gaussian: a mean of shape `(N,)` and a covariance `lineax` operator of shape `(N, N)`. |
| `PropagationResult` | What `integrate` returns: `.state` (a `GaussianState` for the output) and `.cross_cov` of shape `(N_in, N_out)`, or `None`. |
| `GaussHermiteIntegrator(order=20)` | A tensor-product rule with `order ** N` points, exact to degree `2 * order - 1`. For `N <= 2`. |
| `UnscentedIntegrator(alpha=0.001, beta=2.0, kappa=0.0)` | `2N + 1` sigma points. Prefer `alpha=1.0`: the default loses digits in float32. |
| `CubatureIntegrator()` | The third-order spherical-radial rule with `2N` points of equal weight. |
| `FifthOrderCubatureIntegrator()` | A degree-5 rule with `2 N² + 1` points; its weights are negative for `N > 4`. |
| `TaylorIntegrator(order=1, correct_variance=True)` | A first or second order expansion about the mean; not point-based. |
| `MonteCarloIntegrator(n_samples=1000, regularization=1e-6, key=None)` | Sample moments; `key=None` uses `jax.random.key(0)`. |
| `gauss_hermite_points(order, dim, dtype=None)` | Standard-normal points and weights, shapes `(order ** dim, dim)` and `(order ** dim,)`. |
| `sigma_points(mean, cov, alpha=0.001, beta=2.0, kappa=0.0)` | Returns `(points, w_m, w_c)`. |
| `cubature_points(mean, cov)` | Returns `(points, weights)` with `2N` points. |
| `fifth_order_cubature_points(mean, cov)` | Returns `(points, weights)` with `2 N² + 1` points. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

m, v = jnp.array([0.5, -0.2]), jnp.array([0.4, 0.3])
state = gaussx.GaussianState(m, lx.DiagonalLinearOperator(v))  # N(m, diag v)


def g(x):  # (2,) -> (1,)
    return jnp.array([x[0] ** 2 + x[1]])


exact_mean = m[0] ** 2 + v[0] + m[1]
exact_cross = jnp.array([[2 * m[0] * v[0]], [v[1]]])  # Cov[x, g(x)], (2, 1)
rules = [
    gaussx.GaussHermiteIntegrator(order=5),
    gaussx.UnscentedIntegrator(alpha=1.0),
    gaussx.CubatureIntegrator(),
    gaussx.FifthOrderCubatureIntegrator(),
]
for rule in rules:
    out = rule.integrate(g, state)  # PropagationResult
    assert jnp.allclose(out.state.mean[0], exact_mean, atol=1e-3)
    assert jnp.allclose(out.cross_cov, exact_cross, atol=1e-3)

pts, w = gaussx.cubature_points(m, state.cov)  # (4, 2), (4,)
assert pts.shape == (4, 2) and jnp.allclose(w.sum(), 1.0)
```

## Rules

- Wrap the input as `GaussianState(mean, cov_operator)`; the function maps `(N,)` to `(M,)`, so return a 1-element array for a scalar.
- Read the output as `result.state.mean`, `result.state.cov.as_matrix()` and `result.cross_cov`.
- Use `GaussHermiteIntegrator` for one or two dimensions, `CubatureIntegrator` or `UnscentedIntegrator(alpha=1.0)` for more, and `FifthOrderCubatureIntegrator` when degree 5 is needed.
- `TaylorIntegrator` needs `fn` to be differentiable and has no `points_and_weights`.
- `guarantees_psd(dim)` says whether the output covariance is positive semidefinite by construction at that dimension.

## Goes wrong

- `UnscentedIntegrator()` with its default `alpha=1e-3` puts a weight near `-1e6` on the centre point; in float32 the second digit is wrong.
- `GaussHermiteIntegrator(order=20)` at `N = 4` is 160 000 function evaluations.
- `FifthOrderCubatureIntegrator` in more than four dimensions is not guaranteed to give a PSD covariance.
- `MonteCarloIntegrator` with `key=None` draws the same samples on every call.

## Next

- A likelihood expectation that takes an integrator: [likelihoods](likelihoods.md).
- `mean_expectation`, `moment_transform` and the other expectation functions: [expectations](expectations.md).
- Filters that call an integrator at each step: [nonlinear-filtering](nonlinear-filtering.md).
