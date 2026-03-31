"""Tests for Joseph-form covariance update."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from gaussx._sugar._joseph import joseph_update
from gaussx._testing import random_pd_matrix, tree_allclose


def test_joseph_matches_standard_update(getkey):
    """Joseph form should match P - K S K^T in the exact-gain case."""
    N, M = 4, 2
    P_pred = random_pd_matrix(getkey(), N)
    H = jr.normal(getkey(), (M, N))
    R = random_pd_matrix(getkey(), M)

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ jnp.linalg.inv(S)

    P_joseph = joseph_update(P_pred, K, H, R)
    P_standard = P_pred - K @ S @ K.T

    assert tree_allclose(P_joseph, P_standard, rtol=1e-5)


def test_joseph_guarantees_symmetry(getkey):
    """Output should be symmetric even with perturbed gain."""
    N, M = 5, 3
    P_pred = random_pd_matrix(getkey(), N)
    H = jr.normal(getkey(), (M, N))
    R = random_pd_matrix(getkey(), M)

    # Perturb the Kalman gain to simulate an approximate gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ jnp.linalg.inv(S)
    K_noisy = K + 0.01 * jr.normal(getkey(), K.shape)

    P_updated = joseph_update(P_pred, K_noisy, H, R)

    assert tree_allclose(P_updated, P_updated.T, atol=1e-12)


def test_joseph_shape(getkey):
    """Output shape should match P_pred."""
    N, M = 6, 2
    P_pred = random_pd_matrix(getkey(), N)
    H = jr.normal(getkey(), (M, N))
    R = random_pd_matrix(getkey(), M)
    K = jr.normal(getkey(), (N, M))

    P_updated = joseph_update(P_pred, K, H, R)
    assert P_updated.shape == (N, N)


def test_joseph_zero_gain(getkey):
    """With K=0, Joseph update should return P_pred."""
    N, M = 3, 2
    P_pred = random_pd_matrix(getkey(), N)
    H = jr.normal(getkey(), (M, N))
    R = random_pd_matrix(getkey(), M)
    K = jnp.zeros((N, M))

    P_updated = joseph_update(P_pred, K, H, R)
    assert tree_allclose(P_updated, P_pred, atol=1e-12)
