"""The update from precomputed evaluations against its moment-transform twin."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._testing import default_tolerances, tree_allclose


def _prior(state_size=3, seed=0):
    keys = jr.split(jr.key(seed))
    factor = jr.normal(keys[0], (state_size, state_size))
    cov = factor @ factor.T + jnp.eye(state_size)
    return jr.normal(keys[1], (state_size,)), cov


def _nonlinear(state):
    return jnp.array(
        [jnp.sin(state[0]) + state[1] ** 2, jnp.tanh(state[1] * state[-1])]
    )


def _sigma_inputs(mean, cov, obs_fn=_nonlinear, **rule):
    points, mean_weights, cov_weights = gaussx.sigma_points(
        mean, lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag), **rule
    )
    return points, jax.vmap(obs_fn)(points), mean_weights, cov_weights


@pytest.mark.parametrize("joseph", [True, False])
@pytest.mark.parametrize(
    "rule", [{"alpha": 1.0}, {"alpha": 1.0, "kappa": 1.0}, {"alpha": 0.5, "beta": 2.0}]
)
def test_matches_the_update_that_calls_the_observation_function(joseph, rule):
    mean, cov = _prior()
    observation = jnp.array([0.4, -0.2])
    noise = 0.3 * jnp.eye(2) + 0.05
    points, evaluations, mean_weights, cov_weights = _sigma_inputs(mean, cov, **rule)
    result = gaussx.sigma_point_kalman_update(
        points,
        evaluations,
        mean,
        cov,
        observation,
        noise,
        mean_weights=mean_weights,
        cov_weights=cov_weights,
        joseph=joseph,
    )
    expected = gaussx.nonlinear_kalman_update(
        _nonlinear,
        mean,
        cov,
        observation,
        noise,
        integrator=gaussx.UnscentedIntegrator(**rule),
        joseph=joseph,
    )
    rtol, atol = default_tolerances(result)
    assert tree_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("joseph", [True, False])
def test_a_linear_map_is_the_kalman_update(joseph):
    mean, cov = _prior(4)
    obs_model = jr.normal(jr.key(1), (2, 4))
    noise = jnp.array([[0.3, 0.1], [0.1, 0.5]])
    observation = jnp.array([1.0, -0.5])

    def linear(state):
        return obs_model @ state

    points, evaluations, mean_weights, cov_weights = _sigma_inputs(
        mean, cov, linear, alpha=1.0
    )
    updated_mean, updated_cov, log_likelihood = gaussx.sigma_point_kalman_update(
        points,
        evaluations,
        mean,
        cov,
        observation,
        noise,
        mean_weights=mean_weights,
        cov_weights=cov_weights,
        joseph=joseph,
    )
    innovation_cov = obs_model @ cov @ obs_model.T + noise
    gain = cov @ obs_model.T @ jnp.linalg.inv(innovation_cov)
    residual = observation - obs_model @ mean
    expected_mean = mean + gain @ residual
    expected_cov = cov - gain @ innovation_cov @ gain.T
    expected_log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(
        observation, obs_model @ mean, innovation_cov
    )
    rtol, atol = default_tolerances(updated_mean)
    tight = {"rtol": 10 * rtol, "atol": 10 * atol}
    assert tree_allclose(updated_mean, expected_mean, **tight)
    assert tree_allclose(updated_cov, expected_cov, **tight)
    assert tree_allclose(log_likelihood, expected_log_likelihood, **tight)


def test_covariance_weights_default_to_the_mean_weights():
    mean, cov = _prior()
    points, evaluations, mean_weights, _ = _sigma_inputs(mean, cov, alpha=1.0)
    inputs = (points, evaluations, mean, cov, jnp.zeros(2), jnp.eye(2))
    implicit = gaussx.sigma_point_kalman_update(*inputs, mean_weights=mean_weights)
    explicit = gaussx.sigma_point_kalman_update(
        *inputs, mean_weights=mean_weights, cov_weights=mean_weights
    )
    assert tree_allclose(implicit, explicit)


def test_operators_match_dense_inputs():
    mean, cov = _prior()
    points, evaluations, mean_weights, cov_weights = _sigma_inputs(mean, cov, alpha=1.0)
    weights = {"mean_weights": mean_weights, "cov_weights": cov_weights}
    observation = jnp.array([0.1, 0.2])
    noise = 0.4 * jnp.ones(2)
    dense = gaussx.sigma_point_kalman_update(
        points, evaluations, mean, cov, observation, jnp.diag(noise), **weights
    )
    operators = gaussx.sigma_point_kalman_update(
        points,
        evaluations,
        mean,
        lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag),
        observation,
        lx.DiagonalLinearOperator(noise),
        **weights,
    )
    assert tree_allclose(dense, operators)


def test_tempering_the_noise_matches_a_scaled_covariance():
    # The UKI use: Δt enters once, through Δt⁻¹ Γ.
    mean, cov = _prior()
    points, evaluations, mean_weights, cov_weights = _sigma_inputs(mean, cov, alpha=1.0)
    weights = {"mean_weights": mean_weights, "cov_weights": cov_weights}
    gamma = lx.DiagonalLinearOperator(0.4 * jnp.ones(2))
    observation = jnp.array([0.1, 0.2])
    tempered = gaussx.sigma_point_kalman_update(
        points,
        evaluations,
        mean,
        cov,
        observation,
        gaussx.scaled_operator(gamma, 4.0),
        **weights,
    )
    dense = gaussx.sigma_point_kalman_update(
        points, evaluations, mean, cov, observation, 1.6 * jnp.eye(2), **weights
    )
    assert tree_allclose(tempered, dense)


def test_jit_and_vmap_match_a_loop():
    mean, cov = _prior()
    points, evaluations, mean_weights, cov_weights = _sigma_inputs(mean, cov, alpha=1.0)
    weights = {"mean_weights": mean_weights, "cov_weights": cov_weights}
    observations = jr.normal(jr.key(4), (3, 2))

    def update(observation):
        return gaussx.sigma_point_kalman_update(
            points, evaluations, mean, cov, observation, 0.5 * jnp.eye(2), **weights
        )

    batched = eqx.filter_vmap(update)(observations)
    for index in range(3):
        looped = update(observations[index])
        rtol, atol = default_tolerances(looped)
        for part, whole in zip(looped, batched, strict=True):
            assert tree_allclose(part, whole[index], rtol=rtol, atol=atol)
    jitted = eqx.filter_jit(update)(observations[0])
    rtol, atol = default_tolerances(jitted)
    assert tree_allclose(jitted, update(observations[0]), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "prior", ["full", "isotropic"], ids=["full", "repeated-eigenvalues"]
)
def test_gradient_with_respect_to_evaluations_is_finite(prior):
    mean, cov = _prior()
    if prior == "isotropic":
        cov = jnp.eye(3)
    points, evaluations, mean_weights, cov_weights = _sigma_inputs(mean, cov, alpha=1.0)

    def loss(values):
        updated_mean, updated_cov, log_likelihood = gaussx.sigma_point_kalman_update(
            points,
            values,
            mean,
            cov,
            jnp.array([0.2, 0.3]),
            0.5 * jnp.eye(2),
            mean_weights=mean_weights,
            cov_weights=cov_weights,
        )
        return jnp.sum(updated_mean) + jnp.trace(updated_cov) + log_likelihood

    gradient = jax.grad(loss)(evaluations)
    assert gradient.shape == evaluations.shape
    assert jnp.isfinite(gradient).all()
    assert jnp.abs(gradient).max() > 0.0


@pytest.mark.x64_only(reason="finite differences need float64")
def test_gradient_with_respect_to_evaluations_matches_central_differences():
    mean, cov = _prior()
    points, evaluations, mean_weights, cov_weights = _sigma_inputs(mean, cov, alpha=1.0)

    def loss(values):
        updated_mean, updated_cov, log_likelihood = gaussx.sigma_point_kalman_update(
            points,
            values,
            mean,
            cov,
            jnp.array([0.2, 0.3]),
            0.5 * jnp.eye(2),
            mean_weights=mean_weights,
            cov_weights=cov_weights,
        )
        return jnp.sum(jnp.sin(updated_mean)) + jnp.trace(updated_cov) + log_likelihood

    tangent = jr.normal(jr.key(5), evaluations.shape)
    step = 1e-6
    finite_difference = (
        loss(evaluations + step * tangent) - loss(evaluations - step * tangent)
    ) / (2.0 * step)
    assert jnp.allclose(
        jnp.vdot(jax.grad(loss)(evaluations), tangent), finite_difference, rtol=1e-7
    )


@pytest.mark.x64_only(reason="compares float32 runs with a float64 reference")
@pytest.mark.parametrize("joseph", [True, False])
def test_float32_matches_float64(joseph):
    mean, cov = _prior(5)
    points, evaluations, mean_weights, cov_weights = _sigma_inputs(mean, cov, alpha=1.0)
    arguments = (
        points,
        evaluations,
        mean,
        cov,
        jnp.array([0.2, 0.3]),
        0.5 * jnp.eye(2),
    )
    weights = {"mean_weights": mean_weights, "cov_weights": cov_weights}
    reference = gaussx.sigma_point_kalman_update(*arguments, **weights, joseph=joseph)
    single = gaussx.sigma_point_kalman_update(
        *[value.astype(jnp.float32) for value in arguments],
        mean_weights=mean_weights.astype(jnp.float32),
        cov_weights=cov_weights.astype(jnp.float32),
        joseph=joseph,
    )
    assert all(part.dtype == jnp.float32 for part in single)
    for part, target in zip(single, reference, strict=True):
        assert jnp.allclose(part, target, rtol=1e-4, atol=1e-4)


def test_an_inconsistent_covariance_is_rejected_by_default():
    # Points of a broad prior, with a covariance that cannot support their Σ_xy.
    mean = jnp.zeros(2)
    broad = 100.0 * jnp.eye(2)
    points, evaluations, mean_weights, cov_weights = _sigma_inputs(
        mean, broad, lambda state: state[:2], alpha=1.0
    )
    arguments = (points, evaluations, mean, 0.01 * jnp.eye(2), jnp.zeros(2), jnp.eye(2))
    weights = {"mean_weights": mean_weights, "cov_weights": cov_weights}
    with pytest.raises(Exception, match="not positive"):
        gaussx.sigma_point_kalman_update(*arguments, **weights, joseph=False)
    unchecked = gaussx.sigma_point_kalman_update(
        *arguments, **weights, joseph=False, validate=False
    )
    assert jnp.isfinite(unchecked[1]).all()


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"evaluations": jnp.zeros((4, 2))}, "evaluations must have shape"),
        ({"mean": jnp.zeros(2)}, "mean must have shape"),
        ({"cov": jnp.eye(2)}, "cov must have shape"),
        ({"mean_weights": jnp.ones(5)}, "mean_weights must have shape"),
        ({"cov_weights": jnp.ones(5)}, "cov_weights must have shape"),
        ({"obs_noise": jnp.eye(1)}, "obs_noise must have shape"),
        ({"points": jnp.zeros(7)}, "points and evaluations must have shapes"),
    ],
)
def test_rejects_shapes_that_would_broadcast(changed, message):
    inputs = {
        "points": jnp.zeros((7, 3)),
        "evaluations": jnp.zeros((7, 2)),
        "mean": jnp.zeros(3),
        "cov": jnp.eye(3),
        "observation": jnp.zeros(2),
        "obs_noise": jnp.eye(2),
        "mean_weights": jnp.ones(7) / 7.0,
        "cov_weights": jnp.ones(7) / 7.0,
    }
    with pytest.raises(ValueError, match=message):
        gaussx.sigma_point_kalman_update(**(inputs | changed))
