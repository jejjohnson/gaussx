"""Tests for Kalman filter, RTS smoother, and Kalman gain recipes."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import (
    FilterState,
    kalman_filter,
    kalman_gain,
    rts_smoother,
)
from gaussx._testing import random_pd_matrix, tree_allclose


def test_kalman_filter_constant_state(getkey):
    """With zero process noise and identity dynamics, filter should converge."""
    N, M, T = 2, 2, 5
    A = jnp.eye(N)
    H = jnp.eye(M)
    Q = 1e-6 * jnp.eye(N)
    R = 0.1 * jnp.eye(M)

    true_state = jnp.array([1.0, 2.0])
    observations = true_state[None, :] + 0.1 * jr.normal(getkey(), (T, M))

    x0 = jnp.zeros(N)
    P0 = jnp.eye(N)

    state = kalman_filter(A, H, Q, R, observations, x0, P0)

    assert isinstance(state, FilterState)
    assert state.filtered_means.shape == (T, N)
    assert state.filtered_covs.shape == (T, N, N)
    assert state.log_likelihood.shape == ()

    # Last filtered mean should be close to true state
    assert tree_allclose(state.filtered_means[-1], true_state, atol=0.5)


def test_kalman_filter_log_likelihood_finite(getkey):
    """Log-likelihood should be finite."""
    N, M, T = 3, 2, 10
    A = 0.9 * jnp.eye(N)
    H = jr.normal(getkey(), (M, N))
    Q = 0.1 * jnp.eye(N)
    R = 0.5 * jnp.eye(M)
    observations = jr.normal(getkey(), (T, M))

    state = kalman_filter(A, H, Q, R, observations, jnp.zeros(N), jnp.eye(N))
    assert jnp.isfinite(state.log_likelihood)


def test_rts_smoother_basic(getkey):
    """Smoother should produce smoother estimates than filter."""
    N, M, T = 2, 2, 8
    A = 0.95 * jnp.eye(N)
    H = jnp.eye(M)
    Q = 0.1 * jnp.eye(N)
    R = 0.5 * jnp.eye(M)
    observations = jr.normal(getkey(), (T, M))

    state = kalman_filter(A, H, Q, R, observations, jnp.zeros(N), jnp.eye(N))
    s_means, s_covs = rts_smoother(state, A, Q)

    assert s_means.shape == (T, N)
    assert s_covs.shape == (T, N, N)


def test_kalman_gain_basic(getkey):
    """Kalman gain should match manual computation."""
    N, M = 4, 2
    P_mat = random_pd_matrix(getkey(), N)
    H_mat = jr.normal(getkey(), (M, N))
    R_mat = random_pd_matrix(getkey(), M)

    P = lx.MatrixLinearOperator(P_mat)
    H = lx.MatrixLinearOperator(H_mat)
    R = lx.MatrixLinearOperator(R_mat)

    K = kalman_gain(P, H, R)

    # Manual: K = P H^T (H P H^T + R)^{-1}
    S = H_mat @ P_mat @ H_mat.T + R_mat
    expected = P_mat @ H_mat.T @ jnp.linalg.inv(S)

    assert tree_allclose(K, expected, rtol=1e-4)


def test_kalman_gain_shape(getkey):
    N, M = 5, 3
    P = lx.MatrixLinearOperator(random_pd_matrix(getkey(), N))
    H = lx.MatrixLinearOperator(jr.normal(getkey(), (M, N)))
    R = lx.MatrixLinearOperator(random_pd_matrix(getkey(), M))

    K = kalman_gain(P, H, R)
    assert K.shape == (N, M)
