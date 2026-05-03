"""Tests for parallel Kalman filter and RTS smoother."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import DenseSolver, FilterState, kalman_filter, rts_smoother
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


def test_parallel_kf_with_dense_solver_matches_default(getkey):
    """Passing solver=DenseSolver() must match the default dispatch path."""
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (6, 2))

    default_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    dense_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, solver=DenseSolver())

    assert tree_allclose(
        dense_state.filtered_means, default_state.filtered_means, rtol=1e-5
    )
    assert tree_allclose(
        dense_state.filtered_covs, default_state.filtered_covs, rtol=1e-5
    )
    assert tree_allclose(
        dense_state.log_likelihood, default_state.log_likelihood, rtol=1e-4
    )


def test_parallel_rts_with_dense_solver_matches_default(getkey):
    """Smoother solver=DenseSolver() must match the default dispatch path."""
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))

    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    default_means, default_covs = parallel_rts_smoother(state, A, Q)
    dense_means, dense_covs = parallel_rts_smoother(state, A, Q, solver=DenseSolver())

    assert tree_allclose(dense_means, default_means, rtol=1e-5)
    assert tree_allclose(dense_covs, default_covs, rtol=1e-5)


# ----------------------------------------------------------------
# Operator-typed inputs (time-invariant)
# ----------------------------------------------------------------


def test_parallel_kf_obs_noise_diagonal_operator(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))
    R_diag = jnp.diag(R)

    ref = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    op = parallel_kalman_filter(A, H, Q, lx.DiagonalLinearOperator(R_diag), obs, x0, P0)
    assert tree_allclose(ref.filtered_means, op.filtered_means, rtol=1e-5)
    assert tree_allclose(ref.log_likelihood, op.log_likelihood, rtol=1e-5)


def test_parallel_kf_transition_block_diag_operator(getkey):
    from gaussx import BlockDiag

    A1 = 0.9 * jnp.eye(2)
    A2 = 0.7 * jnp.eye(2)
    A_dense = jnp.block([[A1, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), A2]])
    N, M, T = 4, 2, 6
    H = jr.normal(getkey(), (M, N))
    Q = 0.1 * jnp.eye(N)
    R = 0.5 * jnp.eye(M)
    obs = jr.normal(getkey(), (T, M))
    x0, P0 = jnp.zeros(N), jnp.eye(N)

    ref = parallel_kalman_filter(A_dense, H, Q, R, obs, x0, P0)
    A_op = BlockDiag(lx.MatrixLinearOperator(A1), lx.MatrixLinearOperator(A2))
    op = parallel_kalman_filter(A_op, H, Q, R, obs, x0, P0)
    assert tree_allclose(ref.filtered_means, op.filtered_means, rtol=1e-5)


# ----------------------------------------------------------------
# Time-varying inputs and mask
# ----------------------------------------------------------------


def test_parallel_kf_ti_broadcast_matches(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 6
    obs = jr.normal(getkey(), (T, 2))
    ref = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    A_seq = jnp.broadcast_to(A, (T, *A.shape))
    H_seq = jnp.broadcast_to(H, (T, *H.shape))
    Q_seq = jnp.broadcast_to(Q, (T, *Q.shape))
    R_seq = jnp.broadcast_to(R, (T, *R.shape))
    tv = parallel_kalman_filter(A_seq, H_seq, Q_seq, R_seq, obs, x0, P0)
    assert tree_allclose(ref.filtered_means, tv.filtered_means, rtol=1e-6)
    assert tree_allclose(ref.log_likelihood, tv.log_likelihood, rtol=1e-6)


def test_parallel_kf_mask_predict_only(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 6
    obs = jr.normal(getkey(), (T, 2))
    mask = jnp.array([True, False, True, False, True, True])
    out = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, mask=mask)
    idx = jnp.where(~mask)[0]
    assert tree_allclose(out.filtered_means[idx], out.predicted_means[idx], atol=1e-7)


def test_parallel_rts_smoother_tv(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 6
    A_seq = jnp.broadcast_to(A, (T, *A.shape))
    obs = jr.normal(getkey(), (T, 2))
    state = parallel_kalman_filter(A_seq, H, Q, R, obs, x0, P0)
    s_means, _s_covs = parallel_rts_smoother(state, A_seq, Q)
    # Last smoothed == last filtered.
    assert tree_allclose(s_means[-1], state.filtered_means[-1], rtol=1e-6)
