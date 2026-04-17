"""Tests for parallel Kalman filter and RTS smoother."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from gaussx import FilterState, kalman_filter, rts_smoother
from gaussx._ssm._parallel_kalman import (
    parallel_kalman_filter,
    parallel_rts_smoother,
)
from gaussx._testing import tree_allclose


def _make_model(getkey, N=2, M=2):
    A = 0.95 * jnp.eye(N)
    H = jnp.eye(M, N)
    Q = 0.1 * jnp.eye(N)
    R = 0.5 * jnp.eye(M)
    x0 = jnp.zeros(N)
    P0 = jnp.eye(N)
    return A, H, Q, R, x0, P0


def test_parallel_kf_matches_sequential(getkey):
    """Parallel KF should produce same results as sequential KF."""
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 10
    obs = jr.normal(getkey(), (T, 2))

    seq_state = kalman_filter(A, H, Q, R, obs, x0, P0)
    par_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)

    assert tree_allclose(par_state.filtered_means, seq_state.filtered_means, rtol=1e-4)
    assert tree_allclose(par_state.filtered_covs, seq_state.filtered_covs, rtol=1e-4)
    assert tree_allclose(par_state.log_likelihood, seq_state.log_likelihood, rtol=1e-3)


def test_parallel_kf_returns_filter_state(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    assert isinstance(state, FilterState)
    assert state.filtered_means.shape == (5, 2)
    assert state.filtered_covs.shape == (5, 2, 2)


def test_parallel_rts_last_matches_filter(getkey):
    """Last smoothed state should equal last filtered state."""
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 8
    obs = jr.normal(getkey(), (T, 2))

    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    par_means, par_covs = parallel_rts_smoother(state, A, Q)

    # Last smoothed == last filtered (by definition)
    assert tree_allclose(par_means[-1], state.filtered_means[-1], rtol=1e-6)
    assert tree_allclose(par_covs[-1], state.filtered_covs[-1], rtol=1e-6)


def test_parallel_rts_shape(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (6, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    s_means, s_covs = parallel_rts_smoother(state, A, Q)
    assert s_means.shape == (6, 2)
    assert s_covs.shape == (6, 2, 2)


def test_parallel_rts_matches_sequential(getkey):
    """Dense RTS helper should match the validated sequential smoother."""
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (7, 2))

    seq_state = kalman_filter(A, H, Q, R, obs, x0, P0)
    par_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)

    seq_means, seq_covs = rts_smoother(seq_state, A, Q)
    par_means, par_covs = parallel_rts_smoother(par_state, A, Q)

    assert tree_allclose(par_means, seq_means, rtol=1e-4)
    assert tree_allclose(par_covs, seq_covs, rtol=1e-4)
