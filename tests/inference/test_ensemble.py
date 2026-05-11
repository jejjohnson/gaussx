"""Tests for ensemble covariance and cross-covariance recipes."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np

from gaussx import (
    ensemble_covariance,
    ensemble_cross_covariance,
    ensemble_kalman_gain,
)
from gaussx._operators import LowRankUpdate
from gaussx._testing import tree_allclose


def test_ensemble_covariance_shape(getkey):
    J, N = 10, 5
    particles = jr.normal(getkey(), (J, N))
    C = ensemble_covariance(particles)
    assert isinstance(C, LowRankUpdate)
    assert C.as_matrix().shape == (N, N)


def test_ensemble_covariance_matches_dense(getkey):
    J, N = 20, 4
    particles = jr.normal(getkey(), (J, N))
    C = ensemble_covariance(particles)

    # Dense reference
    mean = jnp.mean(particles, axis=0)
    dev = particles - mean
    expected = (dev.T @ dev) / J

    assert tree_allclose(C.as_matrix(), expected, rtol=1e-4)


def test_ensemble_covariance_matches_numpy_divisors(getkey):
    J, N = 20, 4
    particles = jr.normal(getkey(), (J, N))

    mle = ensemble_covariance(particles)
    bessel = ensemble_covariance(particles, bessel=True)

    expected_mle = np.cov(np.asarray(particles).T, ddof=0)
    expected_bessel = np.cov(np.asarray(particles).T, ddof=1)

    assert tree_allclose(mle.as_matrix(), jnp.asarray(expected_mle), atol=1e-10)
    assert tree_allclose(bessel.as_matrix(), jnp.asarray(expected_bessel), atol=1e-10)


def test_ensemble_covariance_symmetric(getkey):
    particles = jr.normal(getkey(), (15, 6))
    C = ensemble_covariance(particles)
    mat = C.as_matrix()
    assert tree_allclose(mat, mat.T, atol=1e-10)


def test_ensemble_cross_covariance(getkey):
    J, N, M = 20, 5, 3
    theta = jr.normal(getkey(), (J, N))
    G = jr.normal(getkey(), (J, M))

    C = ensemble_cross_covariance(theta, G)
    assert C.shape == (N, M)

    # Dense reference
    dev_theta = theta - jnp.mean(theta, axis=0, keepdims=True)
    dev_G = G - jnp.mean(G, axis=0, keepdims=True)
    expected = (dev_theta.T @ dev_G) / J

    assert tree_allclose(C, expected, rtol=1e-5)


def test_ensemble_cross_covariance_matches_bessel_reference(getkey):
    J, N, M = 20, 5, 3
    theta = jr.normal(getkey(), (J, N))
    G = jr.normal(getkey(), (J, M))

    C = ensemble_cross_covariance(theta, G, bessel=True)

    dev_theta = theta - jnp.mean(theta, axis=0, keepdims=True)
    dev_G = G - jnp.mean(G, axis=0, keepdims=True)
    expected = (dev_theta.T @ dev_G) / (J - 1)

    assert tree_allclose(C, expected, atol=1e-10)


def test_ensemble_kalman_gain_matches_dense_linear_gaussian(getkey):
    J, N, M = 16, 5, 3
    particles = jr.normal(getkey(), (J, N))
    H = jr.normal(getkey(), (M, N))
    obs_particles = particles @ H.T
    obs_var = jnp.linspace(0.2, 0.6, M)
    obs_noise = lx.DiagonalLinearOperator(obs_var)

    K = ensemble_kalman_gain(particles, obs_particles, obs_noise)

    dev_x = particles - jnp.mean(particles, axis=0, keepdims=True)
    dev_y = obs_particles - jnp.mean(obs_particles, axis=0, keepdims=True)
    cross_cov = (dev_x.T @ dev_y) / (J - 1)
    obs_cov = (dev_y.T @ dev_y) / (J - 1)
    S = obs_cov + jnp.diag(obs_var)
    expected = jnp.linalg.solve(S, cross_cov.T).T

    assert tree_allclose(K, expected, atol=1e-7)


def test_ensemble_kalman_gain_mle_divisor(getkey):
    J, N, M = 16, 5, 3
    particles = jr.normal(getkey(), (J, N))
    H = jr.normal(getkey(), (M, N))
    obs_particles = particles @ H.T
    obs_var = jnp.linspace(0.2, 0.6, M)
    obs_noise = lx.DiagonalLinearOperator(obs_var)

    K = ensemble_kalman_gain(particles, obs_particles, obs_noise, bessel=False)

    dev_x = particles - jnp.mean(particles, axis=0, keepdims=True)
    dev_y = obs_particles - jnp.mean(obs_particles, axis=0, keepdims=True)
    cross_cov = (dev_x.T @ dev_y) / J
    obs_cov = (dev_y.T @ dev_y) / J
    S = obs_cov + jnp.diag(obs_var)
    expected = jnp.linalg.solve(S, cross_cov.T).T

    assert tree_allclose(K, expected, atol=1e-7)


def test_ensemble_recipes_jit_grad_and_vmap(getkey):
    J, N, M = 10, 4, 2
    particles = jr.normal(getkey(), (J, N))
    H = jr.normal(getkey(), (M, N))
    obs_var = jnp.linspace(0.2, 0.4, M)
    obs_noise = lx.DiagonalLinearOperator(obs_var)

    def gain_loss(particles):
        obs_particles = particles @ H.T
        K = ensemble_kalman_gain(particles, obs_particles, obs_noise)
        return jnp.sum(K)

    jitted = jax.jit(gain_loss)(particles)
    grad = jax.grad(gain_loss)(particles)
    vmapped = eqx.filter_vmap(
        lambda particles: ensemble_kalman_gain(
            particles,
            particles @ H.T,
            obs_noise,
        )
    )(jnp.stack([particles, particles + 0.1]))

    assert jnp.isfinite(jitted)
    assert grad.shape == particles.shape
    assert vmapped.shape == (2, N, M)
