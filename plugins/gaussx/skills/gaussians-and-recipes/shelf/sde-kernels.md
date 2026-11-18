# SDE kernels

Role: ③ model (`role-ladder`, `references/model.md`).
Unit: state space.

Open this when you want a one-dimensional GP kernel as a linear SDE, so that a Kalman filter or smoother costs `O(T)` instead of `O(T³)`.
A kernel object gives the continuous-time matrices `(F, L, H, Q_c, P_inf)`, and `discretise(dt)` gives the transition `A` and process noise `Q` for a step.

## Names

| Name | What it is |
|---|---|
| `SDEKernel` | The abstract base. Subclasses give `state_dim`, `sde_params()`, `discretise(dt)`, `discretise_sequence(dt)`, `initial_covariance()` and `stationary`. |
| `SDEParams` | A named tuple `(F, L, H, Q_c, P_inf)` for `dx = F x dt + L dW`; `P_inf` may be `None`. |
| `MaternSDE(variance, lengthscale, order)` | Matern with `order` 0, 1 or 2 for smoothness 1/2, 3/2 or 5/2; `state_dim = order + 1`. |
| `ConstantSDE(variance)` | A constant kernel, with `A = I` and `Q = 0`. |
| `CosineSDE(variance, frequency)` | `σ² cos(ω τ)` as a two-dimensional rotation. |
| `PeriodicSDE(variance, lengthscale, period, n_harmonics=6)` | The periodic kernel truncated to `n_harmonics` Fourier terms; `state_dim = 2 * (n_harmonics + 1)`. |
| `IntegratedWienerSDE(diffusion, order=1, P_0=None)` | The local linear trend; non-stationary, so `P_inf` is `None` and it has no autocovariance. |
| `SumSDE(kernels)` | A sum of kernels, block-diagonal in state. |
| `ProductSDE(kernel1, kernel2)` | A product of two kernels, by Kronecker composition. |
| `QuasiPeriodicSDE(kernel1, kernel2)` | A `ProductSDE` of a modulating kernel (Matern) and a periodic one. |
| `sde_autocovariance(kernel, tau)` | The stationary `K(τ) = H exp(F \|τ\|) P_inf Hᵀ`. |
| `discretize_mfd(F, Q_c, dt)` | Returns `(A, Q)` by matrix-fraction decomposition; needs no `P_inf`. Its `Q_c` is the full `d x d` diffusion `L Q_c Lᵀ`. |
| `discretize_mfd_sequence(F, Q_c, dt)` | `discretize_mfd` over a vector of steps, giving `(N, d, d)` arrays. |
| `process_noise_covariance(A, Pinf)` | `Q = Pinf - A Pinf Aᵀ`, from the stationary covariance. |
| `LinearizedSDE(A, b, Q)` | A linear surrogate `dx = (A_t x + b_t) dt + Q_t^{1/2} dW`, with arrays of leading length `T`. |
| `linearize_sde(drift_fn, diffusion, path_means, path_covs, integrator=None)` | The best linear drift of a nonlinear SDE along a Gaussian path; returns a `LinearizedSDE`. |
| `sde_kl_divergence(drift_fn, linear_drift, path_means, path_covs, dt, integrator=None)` | The drift path-KL between the linear and the nonlinear SDE, as a scalar. |

## Use it

```python
import jax.numpy as jnp

import gaussx

k = gaussx.MaternSDE(
    variance=jnp.array(1.0), lengthscale=jnp.array(0.5), order=1
)  # d = 2
tau = jnp.array([0.0, 0.2, 1.0])
r = jnp.sqrt(3.0) * tau / 0.5
assert jnp.allclose(gaussx.sde_autocovariance(k, tau), (1 + r) * jnp.exp(-r), atol=1e-4)

p = k.sde_params()
A, Q = k.discretise(jnp.array(0.1))  # (2, 2), (2, 2)
assert jnp.allclose(Q, gaussx.process_noise_covariance(A, p.P_inf), atol=1e-4)
A2, Q2 = gaussx.discretize_mfd(p.F, p.L @ p.Q_c @ p.L.T, jnp.array(0.1))
assert jnp.allclose(A, A2, atol=1e-4) and jnp.allclose(Q, Q2, atol=1e-4)

trend = gaussx.IntegratedWienerSDE(diffusion=jnp.array(0.1))
both = gaussx.SumSDE((k, trend))  # block-diagonal, d = 2 + 2
assert both.state_dim == 4 and not both.stationary
```

## Rules

- Hyperparameters are scalar arrays (`jnp.array(1.0)`); `order` and `n_harmonics` are static integers.
- Use `kernel.discretise(dt)` rather than computing `expm` yourself; each kernel keeps its own exact route.
- Pass `discretize_mfd` the full diffusion `L @ Q_c @ L.T`, a `d x d` matrix.
- A non-stationary kernel starts from `kernel.initial_covariance()`, not from `P_inf`.
- Branch on `kernel.stationary`, not on `sde_params().P_inf is None`.
- `linearize_sde` defaults to `UnscentedIntegrator(alpha=1.0)`; see [integrators](integrators.md).

## Goes wrong

- `sde_autocovariance` raises `ValueError` for a non-stationary kernel such as `IntegratedWienerSDE`, and when `P_inf` is `None`.
- `MaternSDE` supports only `order` 0, 1 and 2; any other value raises `ValueError`.
- `sde_kl_divergence` rejects `TaylorIntegrator` at every order.
- A short `lengthscale` in `PeriodicSDE` needs more harmonics than the default 6.

## Next

- Filtering and smoothing with these matrices: [kalman](kalman.md).
- A nonlinear drift handled by linearisation: [nonlinear-filtering](nonlinear-filtering.md).
- The distribution over the states of the chain: [distributions](distributions.md).
