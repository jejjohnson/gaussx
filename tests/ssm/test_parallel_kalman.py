"""Tests for the parallel Kalman filter and RTS smoother.

The parallel implementation is the Särkkä-García-Fernández covariance
form via :func:`jax.lax.associative_scan`. These tests pin numerical
parity against the validated sequential filter / smoother in
``_kalman.py`` and exercise the API surface (TI / TV broadcast, mask,
operator inputs, JIT, vmap, grad).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

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


# ----------------------------------------------------------------
# Parity with the sequential filter / smoother
# ----------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 2, 8, 64])
def test_parallel_kf_matches_sequential(getkey, T):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (T, 2))

    seq_state = kalman_filter(A, H, Q, R, obs, x0, P0)
    par_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)

    assert tree_allclose(par_state.filtered_means, seq_state.filtered_means, rtol=1e-4)
    assert tree_allclose(par_state.filtered_covs, seq_state.filtered_covs, rtol=1e-4)
    assert tree_allclose(
        par_state.predicted_means, seq_state.predicted_means, rtol=1e-4
    )
    assert tree_allclose(par_state.predicted_covs, seq_state.predicted_covs, rtol=1e-4)
    assert tree_allclose(par_state.log_likelihood, seq_state.log_likelihood, rtol=1e-3)


@pytest.mark.parametrize("T", [1, 2, 8, 64])
def test_parallel_rts_matches_sequential(getkey, T):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (T, 2))

    seq_state = kalman_filter(A, H, Q, R, obs, x0, P0)
    par_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)

    seq_means, seq_covs = rts_smoother(seq_state, A, Q)
    par_means, par_covs = parallel_rts_smoother(par_state, A, Q)

    assert tree_allclose(par_means, seq_means, rtol=1e-4)
    assert tree_allclose(par_covs, seq_covs, rtol=1e-4)


def test_parallel_kf_returns_filter_state(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    assert isinstance(state, FilterState)
    assert state.filtered_means.shape == (5, 2)
    assert state.filtered_covs.shape == (5, 2, 2)
    assert state.predicted_means.shape == (5, 2)
    assert state.predicted_covs.shape == (5, 2, 2)


def test_parallel_kf_sqrt_matches_covariance_form(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (100, 2))

    cov_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    sqrt_state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, form="sqrt")

    assert tree_allclose(sqrt_state.filtered_means, cov_state.filtered_means, rtol=1e-5)
    assert tree_allclose(sqrt_state.filtered_covs, cov_state.filtered_covs, rtol=1e-5)
    assert tree_allclose(
        sqrt_state.predicted_means, cov_state.predicted_means, rtol=1e-5
    )
    assert tree_allclose(sqrt_state.predicted_covs, cov_state.predicted_covs, rtol=1e-5)
    assert tree_allclose(sqrt_state.log_likelihood, cov_state.log_likelihood, rtol=1e-5)


def test_parallel_rts_sqrt_matches_covariance_form(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (64, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, form="sqrt")

    cov_means, cov_covs = parallel_rts_smoother(state, A, Q)
    sqrt_means, sqrt_covs = parallel_rts_smoother(state, A, Q, form="sqrt")

    assert tree_allclose(sqrt_means, cov_means, rtol=1e-5)
    assert tree_allclose(sqrt_covs, cov_covs, rtol=1e-5)


def test_parallel_kf_sqrt_covariances_are_psd(getkey):
    dtype = jnp.float32
    A = jnp.array([[0.999, 0.01], [0.0, 0.98]], dtype=dtype)
    H = jnp.array([[1.0, 0.0]], dtype=dtype)
    Q = jnp.diag(jnp.array([1e-8, 1e-10], dtype=dtype))
    R = jnp.array([[1e-6]], dtype=dtype)
    x0 = jnp.zeros(2, dtype=dtype)
    P0 = jnp.eye(2, dtype=dtype)
    obs = jr.normal(getkey(), (128, 1), dtype=dtype)

    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, form="sqrt")
    covs = jnp.concatenate([state.filtered_covs, state.predicted_covs], axis=0)
    min_eig = jnp.min(jnp.linalg.eigvalsh(covs))
    atol = jnp.array(jnp.finfo(dtype).eps * 100, dtype=dtype)

    assert jnp.isfinite(min_eig)
    assert min_eig >= -atol


def test_parallel_kf_rejects_unknown_form(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))

    with pytest.raises(ValueError, match="form"):
        parallel_kalman_filter(A, H, Q, R, obs, x0, P0, form="information")


def test_parallel_rts_rejects_unknown_form(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (5, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)

    with pytest.raises(ValueError, match="form"):
        parallel_rts_smoother(state, A, Q, form="information")


def test_parallel_rts_last_matches_filter(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 8
    obs = jr.normal(getkey(), (T, 2))
    state = parallel_kalman_filter(A, H, Q, R, obs, x0, P0)
    par_means, par_covs = parallel_rts_smoother(state, A, Q)
    assert tree_allclose(par_means[-1], state.filtered_means[-1], rtol=1e-6)
    assert tree_allclose(par_covs[-1], state.filtered_covs[-1], rtol=1e-6)


# ----------------------------------------------------------------
# solver= kwarg passthrough (currently a no-op; pinned for API stability)
# ----------------------------------------------------------------


def test_parallel_kf_with_dense_solver_matches_default(getkey):
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


def test_parallel_kf_woodbury_innovation_matches_sequential(getkey):
    N, M, T = 4, 32, 5
    A = 0.95 * jnp.eye(N)
    H = jr.normal(getkey(), (M, N)) / jnp.sqrt(N)
    Q = 0.05 * jnp.eye(N)
    R_diag = 0.3 + 0.1 * jnp.linspace(0.0, 1.0, M)
    obs = jr.normal(getkey(), (T, M))
    x0, P0 = jnp.zeros(N), jnp.eye(N)

    ref = kalman_filter(
        A,
        H,
        Q,
        lx.DiagonalLinearOperator(R_diag),
        obs,
        x0,
        P0,
        woodbury_innovation=True,
    )
    got = parallel_kalman_filter(
        A,
        H,
        Q,
        lx.DiagonalLinearOperator(R_diag),
        obs,
        x0,
        P0,
        woodbury_innovation=True,
    )

    assert tree_allclose(got.filtered_means, ref.filtered_means, rtol=1e-6)
    assert tree_allclose(got.filtered_covs, ref.filtered_covs, rtol=1e-6)
    assert tree_allclose(got.log_likelihood, ref.log_likelihood, rtol=1e-6)


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


def test_parallel_kf_rejects_3d_with_operator(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 4
    obs = jr.normal(getkey(), (T, 2))
    H_seq = jnp.broadcast_to(H, (T, *H.shape))
    A_op = lx.MatrixLinearOperator(A)
    with pytest.raises(TypeError):
        parallel_kalman_filter(A_op, H_seq, Q, R, obs, x0, P0)


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
    par = parallel_kalman_filter(A, H, Q, R, obs, x0, P0, mask=mask)
    seq = kalman_filter(A, H, Q, R, obs, x0, P0, mask=mask)
    assert tree_allclose(par.filtered_means, seq.filtered_means, rtol=1e-4)
    assert tree_allclose(par.filtered_covs, seq.filtered_covs, rtol=1e-4)
    assert tree_allclose(par.log_likelihood, seq.log_likelihood, rtol=1e-3)
    idx = jnp.where(~mask)[0]
    assert tree_allclose(par.filtered_means[idx], par.predicted_means[idx], atol=1e-7)


def test_parallel_rts_smoother_tv(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    T = 6
    A_seq = jnp.broadcast_to(A, (T, *A.shape))
    obs = jr.normal(getkey(), (T, 2))
    state = parallel_kalman_filter(A_seq, H, Q, R, obs, x0, P0)
    s_means, _s_covs = parallel_rts_smoother(state, A_seq, Q)
    assert tree_allclose(s_means[-1], state.filtered_means[-1], rtol=1e-6)


# ----------------------------------------------------------------
# JIT / vmap / grad smoke tests
# ----------------------------------------------------------------


def test_parallel_kf_jit(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (8, 2))

    def fn(A_, H_, Q_, R_, obs_, x0_, P0_):
        return parallel_kalman_filter(A_, H_, Q_, R_, obs_, x0_, P0_).log_likelihood

    eager = fn(A, H, Q, R, obs, x0, P0)
    jitted = jax.jit(fn)(A, H, Q, R, obs, x0, P0)
    assert tree_allclose(eager, jitted, rtol=1e-6)


def test_parallel_kf_vmap(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    B, T = 4, 6
    obs_batch = jr.normal(getkey(), (B, T, 2))

    def fn(obs_):
        return parallel_kalman_filter(A, H, Q, R, obs_, x0, P0).log_likelihood

    batched = jax.vmap(fn)(obs_batch)
    sequential = jnp.stack([fn(obs_batch[b]) for b in range(B)])
    assert tree_allclose(batched, sequential, rtol=1e-5)


def test_parallel_kf_grad(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    obs = jr.normal(getkey(), (8, 2))

    def loss(log_q_diag):
        Q_ = jnp.diag(jnp.exp(log_q_diag))
        return -parallel_kalman_filter(A, H, Q_, R, obs, x0, P0).log_likelihood

    log_q = jnp.log(jnp.diag(Q))
    g_par = jax.grad(loss)(log_q)

    def loss_seq(log_q_diag):
        Q_ = jnp.diag(jnp.exp(log_q_diag))
        return -kalman_filter(A, H, Q_, R, obs, x0, P0).log_likelihood

    g_seq = jax.grad(loss_seq)(log_q)
    assert tree_allclose(g_par, g_seq, rtol=1e-3, atol=1e-5)


def test_parallel_kf_sqrt_jit_vmap_grad(getkey):
    A, H, Q, R, x0, P0 = _make_model(getkey)
    B, T = 3, 8
    obs_batch = jr.normal(getkey(), (B, T, 2))

    def fn(obs_, log_q_diag):
        Q_ = jnp.diag(jnp.exp(log_q_diag))
        return parallel_kalman_filter(
            A, H, Q_, R, obs_, x0, P0, form="sqrt"
        ).log_likelihood

    log_q = jnp.log(jnp.diag(Q))
    batched = jax.jit(jax.vmap(lambda obs_: fn(obs_, log_q)))(obs_batch)
    grad = jax.grad(lambda log_q_: -fn(obs_batch[0], log_q_))(log_q)

    assert batched.shape == (B,)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.all(jnp.isfinite(grad))
