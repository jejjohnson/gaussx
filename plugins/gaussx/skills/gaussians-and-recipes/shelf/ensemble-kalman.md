# Ensemble Kalman

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: inference and ensembles.

Open this when you assimilate an observation into an ensemble of states or parameters: a stochastic EnKF analysis, an ETKF transform, an EKI step, or the localisation and inflation that go with them.
The observation operator never appears; you pass its image of the ensemble as `obs_particles`.

## Names

Ensembles have shape `(J, N)` with members on the first axis, and `obs_noise` is a lineax operator of shape `(M, M)`.

| Name | What it is |
|---|---|
| `enkf_analysis(particles, obs_particles, observation, obs_noise, *, key=None, perturbed_obs=None, localization=None, obs_localization=None, solver=None, dense_innovation=None, bessel=True)` | The perturbed-observation analysis step; returns the analysis ensemble `(J, N)`. |
| `ensemble_kalman_gain(particles, obs_particles, obs_noise, *, solver=None, bessel=True, dense_innovation=None)` | The gain `C^{xH} (C^{HH} + R)⁻¹` of shape `(N, M)`. |
| `localized_kalman_gain(particles, obs_particles, obs_noise, rho_xy, rho_yy, *, solver=None, bessel=True)` | The gain with Schur-product tapers on the state-observation and observation-observation covariances. |
| `etkf_transform(obs_particles, y, obs_noise, *, inflation=1.0, solver=None)` | The deterministic ETKF weights `(w_mean, transform)` of shapes `(J,)` and `(J, J)`. |
| `etkf_analysis(particles, obs_particles, y, obs_noise, *, inflation=1.0, solver=None, low_rank=True, obs_weights=None)` | The ETKF analysis ensemble, without forming the `(J, J)` transform. |
| `ensemble_rts_step(smoothed_next, analysis, forecast_next, *, rcond=None)` | One backward step of the ensemble Rauch-Tung-Striebel smoother; returns the smoothed ensemble. |
| `ledoit_wolf_shrinkage(particles, *, bessel=True)` | The Ledoit-Wolf shrunk ensemble covariance as a `LowRankUpdate` over a diagonal, and the shrinkage `λ`. |
| `gaussian_taper(r, length_scale)`, `soar_taper(r, length_scale)` | The Gaussian and second-order auto-regressive localization tapers of a distance. |
| `eki_step(particles, obs_particles, observation, obs_noise, *, dt=1.0, step=None, key=None, perturbed_obs=None, deterministic=False, ...)` | One ensemble Kalman inversion update with tempering `dt`; equals `enkf_analysis` at `dt=1`. |
| `discrepancy_step_size(obs_particles, observation, obs_noise, *, remaining, bessel=True)` | The adaptive EKI tempering step from the data-misfit rule. |
| `tikhonov_augment(particles, obs_particles, observation, obs_noise, prior_mean, prior_cov)` | The augmented `(obs_particles, observation, obs_noise)` that puts a Gaussian prior into an EKI step. |
| `ensemble_covariance(particles, *, bessel=False)` | The empirical covariance as a low-rank `LowRankUpdate`. |
| `ensemble_cross_covariance(particles_theta, particles_G, *, bessel=False)` | The empirical cross-covariance, shape `(N, M)`. |
| `localization_matrix(coords_a, coords_b, c, metric=euclidean_distance)` | A pairwise Gaspari-Cohn taper with entries in `[0, 1]`. |
| `gaspari_cohn(r, c)` | The fifth-order compactly supported taper function. |
| `euclidean_distance(coords_a, coords_b)`, `haversine_distance(coords_a, coords_b, radius=6371000.0)` | Pairwise distances in a plane and on a sphere. |
| `inflate_multiplicative(ensemble, factor)` | Scales the perturbations about the mean by `factor`. |
| `inflate_rtpp(posterior, prior, alpha)`, `inflate_rtps(posterior, prior, beta, eps=1e-12)` | Relaxation of the analysis perturbations or spread toward the prior; both keep the posterior mean. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

J, N = 40, 3
X = jr.normal(jr.key(0), (J, N)) @ jnp.diag(jnp.array([1.0, 0.5, 2.0]))  # (J, N)
H = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # (M, N) = (2, 3)
G, y = X @ H.T, jnp.array([1.0, -1.0])  # (J, M), (M,)
R = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))

Xa = gaussx.enkf_analysis(X, G, y, R, key=jr.key(1))  # (J, N)
assert Xa.shape == (J, N)

# ETKF is deterministic and matches the Kalman update with the sample covariance.
w, T = gaussx.etkf_transform(G, y, R)
Xp = X - X.mean(0)
mean_a = X.mean(0) + w @ Xp
K = gaussx.ensemble_kalman_gain(X, G, R)  # (N, M)
assert jnp.allclose(mean_a, X.mean(0) + K @ (y - G.mean(0)), atol=1e-3)
assert jnp.allclose((T @ Xp).mean(0), 0.0, atol=1e-4)

coords = jnp.arange(N, dtype=float)[:, None]
rho_xy = gaussx.localization_matrix(coords, jnp.array([[0.0], [2.0]]), 1.5)  # (N, M)
Kloc = gaussx.localized_kalman_gain(X, G, R, rho_xy, jnp.ones((2, 2)))
assert Kloc.shape == (N, 2)
assert jnp.allclose(
    gaussx.inflate_multiplicative(Xa, 1.1).mean(0), Xa.mean(0), atol=1e-4
)
```

## Rules

- Give `enkf_analysis` exactly one of `key` (it draws exact perturbations from `obs_noise`) or `perturbed_obs` of shape `(J, M)`; there is no deterministic flag, and the deterministic alternative is `etkf_transform`.
- Apply the ETKF output as `x_bar + w_mean @ Xp` for the mean and `transform @ Xp` for the anomalies, where `Xp` is the forecast state perturbations of shape `(J, N)`.
- Pass `localization=` (and `obs_localization=`) to `enkf_analysis` or `eki_step` to switch to the localised gain, building the tapers with `localization_matrix`.
- Use `haversine_distance` as the `metric` of `localization_matrix` for latitude-longitude coordinates.
- The gain and analysis functions accept `solver=` from skill `gaussx:solvers-and-preconditioners`; a matrix-free strategy such as `CGSolver` wants `dense_innovation=False`.
- `dense_innovation=None` picks the dense `(M, M)` innovation when `J >= M` and the Woodbury route otherwise; the Woodbury route needs a positive definite `obs_noise`, so pass `dense_innovation=True` for a singular one.
- `bessel` defaults to `True` in the gains, `enkf_analysis` and `eki_step`, but to `False` in `ensemble_covariance` and `ensemble_cross_covariance`; a gain built by hand from those two differs unless you pass `bessel=True`.
- An EKI schedule is exact in the linear-Gaussian limit only when the `dt` values sum to one; `discrepancy_step_size` returns a step capped by `remaining`.
- Use `eki_step(..., deterministic=True)` for the deterministic ETKF-based variant, and `step=` for a state-side operator that sets a per-block rate.
- `etkf_transform` needs `J >= 2` and a positive `inflation`.

## Goes wrong

- The Gaussian update is biased for a non-Gaussian prior, and the bias does not shrink with ensemble size; transform the state into coordinates where the prior is near Gaussian first.
- A taper of the wrong shape can broadcast silently in hand-built code; the library checks `localization` shapes `(N, M)` and `(M, M)` and raises `ValueError`.
- Forgetting `bessel=True` when mixing `ensemble_covariance` with the gains gives a covariance that is too small by `(J − 1) / J`.
- `inflate_rtps` divides by the posterior spread, so a collapsed coordinate relies on `eps` to stay finite.

## Next

- Gaussian conditioning for a single known covariance: [gaussians](gaussians.md).
- Sequential Kalman filters for linear models: [kalman](kalman.md).
- Nonlinear moment-matched filters: [nonlinear-filtering](nonlinear-filtering.md).
- Operators for `obs_noise` and the `solve` verb: skill `gaussx:structured-gaussian-linalg`.
