"""Tests for ensemble covariance and cross-covariance recipes."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from gaussx._operators import LowRankUpdate
from gaussx._recipes import ensemble_covariance, ensemble_cross_covariance
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
