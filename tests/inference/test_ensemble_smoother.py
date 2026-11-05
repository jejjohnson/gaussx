"""The ensemble RTS backward step against dense forms and the moment smoother."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import gaussx
from gaussx._einx import einsum
from gaussx._testing import tree_allclose


# Float64 round-off, or the larger of float32 round-off through a pseudo-inverse.
TOL = 1e-9 if jax.config.read("jax_enable_x64") else 2e-3


def _anomalies(members):
    return members - einx.mean("[member] state", members)


def _ensembles(ensemble_size, state_size, dtype=None, seed=0):
    """An analysis, its forecast, and a smoothed ensemble observing the forecast."""
    keys = jr.split(jr.key(seed), 4)
    analysis = jr.normal(keys[0], (ensemble_size, state_size))
    forecast = 0.9 * analysis + 0.3 * jr.normal(keys[1], (ensemble_size, state_size))
    gain = jr.normal(keys[2], (state_size, state_size)) / state_size
    residual = jr.normal(keys[3], (ensemble_size, state_size)) - forecast
    smoothed = forecast + einsum(residual, gain, "member a, a b -> member b")
    if dtype is not None:
        return smoothed.astype(dtype), analysis.astype(dtype), forecast.astype(dtype)
    return smoothed, analysis, forecast


@pytest.mark.parametrize(
    ("ensemble_size", "state_size"), [(6, 2), (20, 5), (6, 40), (40, 40)]
)
def test_matches_the_dense_ensemble_space_form(ensemble_size, state_size):
    smoothed, analysis, forecast = _ensembles(ensemble_size, state_size)
    anomalies = _anomalies(forecast)
    expected = analysis + (smoothed - forecast) @ anomalies.T @ jnp.linalg.pinv(
        anomalies @ anomalies.T, hermitian=True
    ) @ _anomalies(analysis)
    result = gaussx.ensemble_rts_step(smoothed, analysis, forecast)
    assert tree_allclose(result, expected, rtol=TOL, atol=TOL)


@pytest.mark.parametrize(("ensemble_size", "state_size"), [(50, 3), (200, 10)])
def test_matches_the_state_space_gain_when_the_covariance_is_invertible(
    ensemble_size, state_size
):
    smoothed, analysis, forecast = _ensembles(ensemble_size, state_size)
    forecast_anomalies = _anomalies(forecast)
    cross = _anomalies(analysis).T @ forecast_anomalies
    # G = Σ_af Σ_ff⁻¹, with the same divisor on both sides
    gain = jnp.linalg.solve(forecast_anomalies.T @ forecast_anomalies, cross.T).T
    expected = analysis + (smoothed - forecast) @ gain.T
    result = gaussx.ensemble_rts_step(smoothed, analysis, forecast)
    assert tree_allclose(result, expected, rtol=TOL, atol=TOL)


def test_a_smoothed_ensemble_equal_to_the_forecast_returns_the_analysis():
    _, analysis, forecast = _ensembles(10, 4)
    result = gaussx.ensemble_rts_step(forecast, analysis, forecast)
    assert tree_allclose(result, analysis, rtol=TOL, atol=TOL)


def test_a_forecast_without_spread_returns_the_analysis():
    smoothed, analysis, _ = _ensembles(10, 4)
    forecast = jnp.ones((10, 4))
    result = gaussx.ensemble_rts_step(smoothed, analysis, forecast)
    assert tree_allclose(result, analysis)


def test_a_cutoff_above_every_singular_value_returns_the_analysis():
    smoothed, analysis, forecast = _ensembles(10, 4)
    result = gaussx.ensemble_rts_step(smoothed, analysis, forecast, rcond=1.0)
    assert tree_allclose(result, analysis)


def test_matches_the_moment_smoother_on_a_linear_gaussian_model():
    # Moment oracle: one Kalman step, then the RTS gain back to the first time.
    transition = jnp.array([[0.9, 0.2], [-0.1, 0.8]])
    process_noise = jnp.array([[0.3, 0.05], [0.05, 0.2]])
    obs_model = jnp.array([[1.0, 0.0]])
    obs_noise = jnp.array([[0.1]])
    init_mean = jnp.array([1.0, -1.0])
    init_cov = jnp.array([[1.0, 0.2], [0.2, 0.5]])
    observations = jnp.array([[0.5], [0.9]])
    state = gaussx.kalman_filter(
        transition,
        obs_model,
        process_noise,
        obs_noise,
        observations,
        init_mean,
        init_cov,
    )
    moment_means, moment_covariances = gaussx.rts_smoother(state, transition)
    ensemble_size = 2000
    keys = jr.split(jr.key(0), 3)
    analysis = jr.multivariate_normal(
        keys[0],
        state.filtered_means[0],
        state.filtered_covs[0],
        (ensemble_size,),
    )
    forecast = einsum(
        analysis, transition, "member a, b a -> member b"
    ) + jr.multivariate_normal(keys[1], jnp.zeros(2), process_noise, (ensemble_size,))
    # X_1^s = X_1^f + K (y − H x − ε), with the Kalman gain of the oracle
    predicted = state.predicted_covs[1]
    gain = (
        predicted
        @ obs_model.T
        @ jnp.linalg.inv(obs_model @ predicted @ obs_model.T + obs_noise)
    )
    noise = jnp.sqrt(obs_noise[0, 0]) * jr.normal(keys[2], (ensemble_size, 1))
    residual = (
        observations[1]
        - einsum(forecast, obs_model, "member a, b a -> member b")
        - noise
    )
    smoothed = forecast + einsum(residual, gain, "member o, a o -> member a")

    result = gaussx.ensemble_rts_step(smoothed, analysis, forecast)

    sample_mean = einx.mean("[member] state", result)
    anomalies = _anomalies(result)
    sample_covariance = anomalies.T @ anomalies / (ensemble_size - 1)
    # Monte Carlo tolerance: a few standard errors of J = 2000 members
    assert jnp.allclose(sample_mean, moment_means[0], atol=0.08)
    assert jnp.allclose(sample_covariance, moment_covariances[0], atol=0.08)


@pytest.mark.parametrize(
    ("ensemble_size", "state_size"), [(6, 2), (6, 40), (6, 5), (10, 1)]
)
def test_gradients_are_finite_with_repeated_zero_singular_values(
    ensemble_size, state_size
):
    smoothed, analysis, forecast = _ensembles(ensemble_size, state_size)

    def loss(smoothed_next, analysis_now, forecast_next):
        return jnp.sum(
            gaussx.ensemble_rts_step(smoothed_next, analysis_now, forecast_next) ** 2
        )

    gradients = jax.grad(loss, argnums=(0, 1, 2))(smoothed, analysis, forecast)
    assert all(jnp.isfinite(gradient).all() for gradient in gradients)


def test_gradients_are_finite_through_a_rank_deficient_forecast():
    _, analysis, _ = _ensembles(8, 12)
    keys = jr.split(jr.key(7))
    forecast = jr.normal(keys[0], (8, 2)) @ jr.normal(keys[1], (2, 12))
    smoothed = 1.1 * forecast

    def loss(forecast_next):
        return jnp.sum(gaussx.ensemble_rts_step(smoothed, analysis, forecast_next) ** 2)

    assert jnp.isfinite(jax.grad(loss)(forecast)).all()


@pytest.mark.x64_only(reason="finite differences need float64")
@pytest.mark.parametrize(("ensemble_size", "state_size"), [(6, 2), (6, 12)])
def test_gradient_matches_central_differences(ensemble_size, state_size):
    smoothed, analysis, forecast = _ensembles(ensemble_size, state_size)

    def loss(forecast_next):
        return jnp.sum(
            jnp.sin(gaussx.ensemble_rts_step(smoothed, analysis, forecast_next))
        )

    tangent = jr.normal(jr.key(3), forecast.shape)
    step = 1e-6
    finite_difference = (
        loss(forecast + step * tangent) - loss(forecast - step * tangent)
    ) / (2.0 * step)
    assert jnp.allclose(
        jnp.vdot(jax.grad(loss)(forecast), tangent), finite_difference, rtol=1e-6
    )


def test_jit_and_vmap_over_time_match_a_loop():
    steps = [_ensembles(8, 3, seed=seed) for seed in range(3)]
    stacked = [jnp.stack(members) for members in zip(*steps, strict=True)]
    batched = jax.vmap(gaussx.ensemble_rts_step)(*stacked)
    for index, members in enumerate(steps):
        assert tree_allclose(batched[index], gaussx.ensemble_rts_step(*members))
    jitted = jax.jit(gaussx.ensemble_rts_step)(*steps[0])
    assert tree_allclose(jitted, gaussx.ensemble_rts_step(*steps[0]))


@pytest.mark.x64_only(reason="compares float32 runs with a float64 reference")
@pytest.mark.parametrize(
    ("ensemble_size", "state_size"), [(6, 2), (20, 5), (20, 200), (100, 20)]
)
def test_float32_matches_float64(ensemble_size, state_size):
    reference = gaussx.ensemble_rts_step(*_ensembles(ensemble_size, state_size))
    single = gaussx.ensemble_rts_step(
        *_ensembles(ensemble_size, state_size, dtype=jnp.float32)
    )
    assert single.dtype == jnp.float32
    assert jnp.allclose(single, reference, atol=2e-4, rtol=0.0)


@pytest.mark.x64_only(reason="compares float32 runs with a float64 reference")
def test_a_weak_direction_stays_resolved_in_float32():
    # σ₂ / σ₁ ≈ 10⁻⁴, so σ₂² / σ₁² is below the rounding of F Fᵀ in float32.
    scales = jnp.array([1.0, 1e-4])
    analysis = jr.normal(jr.key(0), (50, 2)) * scales
    forecast = 0.9 * analysis + 0.1 * jr.normal(jr.key(1), (50, 2)) * scales
    # The smoothed forecast moves the weak variable by a tenth of its spread.
    smoothed = forecast + 0.1 * scales * jr.normal(jr.key(2), (50, 2))
    reference = gaussx.ensemble_rts_step(smoothed, analysis, forecast)
    single = gaussx.ensemble_rts_step(
        smoothed.astype(jnp.float32),
        analysis.astype(jnp.float32),
        forecast.astype(jnp.float32),
    )
    weak_spread = jnp.sqrt(einx.mean("[member] state", _anomalies(analysis) ** 2))[1]
    assert jnp.abs(single - reference)[:, 1].max() / weak_spread < 1e-2


@pytest.mark.x64_only(reason="compares float32 runs with a float64 reference")
def test_a_large_mean_does_not_reintroduce_the_null_direction():
    smoothed, analysis, forecast = _ensembles(20, 200)
    # Rounding in the mean leaves F e a little off zero in float32.
    shift, spread = 300.0, 0.1
    moved = [shift + spread * members for members in (smoothed, analysis, forecast)]
    reference = gaussx.ensemble_rts_step(*moved)
    single = gaussx.ensemble_rts_step(*[m.astype(jnp.float32) for m in moved])
    assert jnp.abs(single - reference).max() / spread < 3e-3


def test_rejects_mismatched_shapes_and_a_single_member():
    members = jnp.zeros((4, 2))
    with pytest.raises(ValueError, match="same shape"):
        gaussx.ensemble_rts_step(members, jnp.zeros((4, 3)), members)
    with pytest.raises(ValueError, match="J >= 2"):
        gaussx.ensemble_rts_step(
            jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.zeros((1, 2))
        )
