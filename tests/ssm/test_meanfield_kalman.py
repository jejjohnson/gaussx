"""Tests for the mean-field (block-diagonal) Kalman filter and smoother."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import lineax as lx
import pytest

from gaussx import (
    BlockDiag,
    FilterState,
    kalman_filter,
    meanfield_kalman_filter,
    meanfield_rts_smoother,
    rts_smoother,
)
from gaussx._testing import random_pd_matrix, tree_allclose


def _block_system(key, L, d, m, T):
    """A block-independent LGSSM: ``L`` decoupled ``d``-state blocks.

    Returns per-block stacks ``(L, ...)`` plus the assembled full-state
    dense system. Keys are split deterministically; callers pin the key
    (the randomness is incidental — any stable system would do).
    """
    keys = jr.split(key, 6)
    A_b = 0.9 * jnp.stack([jnp.eye(d) for _ in range(L)])
    A_b = A_b + 0.05 * jr.normal(keys[0], (L, d, d))
    H_b = jr.normal(keys[1], (L, m, d))
    Q_b = jnp.stack([random_pd_matrix(k, d) for k in jr.split(keys[2], L)])
    R_b = jnp.stack([random_pd_matrix(k, m) for k in jr.split(keys[3], L)])
    P0_b = jnp.stack([random_pd_matrix(k, d) for k in jr.split(keys[4], L)])
    m0 = jr.normal(keys[5], (L * d,))

    A = jax.scipy.linalg.block_diag(*A_b)
    H = jax.scipy.linalg.block_diag(*H_b)
    Q = jax.scipy.linalg.block_diag(*Q_b)
    R = jax.scipy.linalg.block_diag(*R_b)
    P0 = jax.scipy.linalg.block_diag(*P0_b)
    y = jr.normal(jr.fold_in(key, 7), (T, L * m))
    return (A_b, H_b, Q_b, R_b, P0_b), (A, H, Q, R, P0), m0, y


def test_meanfield_matches_full_filter_block_independent():
    """Spike check: mean-field == full filter on a decoupled system (1e-6)."""
    L, d, m, T = 3, 2, 2, 12
    _, (A, H, Q, R, P0), m0, y = _block_system(jr.key(0), L, d, m, T)

    full = kalman_filter(A, H, Q, R, y, m0, P0)
    mf = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d)

    assert isinstance(mf, FilterState)
    # The mean-field factorisation is exact for a truly block-diagonal
    # system, so agreement is limited only by float64 roundoff (issue
    # gh-29 asks for 1e-6).
    assert tree_allclose(mf.filtered_means, full.filtered_means, atol=1e-6)
    assert tree_allclose(mf.filtered_covs, full.filtered_covs, atol=1e-6)
    assert tree_allclose(mf.predicted_means, full.predicted_means, atol=1e-6)
    assert tree_allclose(mf.predicted_covs, full.predicted_covs, atol=1e-6)
    assert tree_allclose(mf.log_likelihood, full.log_likelihood, atol=1e-6)


def test_single_block_reduces_to_full_filter():
    """``block_size == D`` runs one block: identical to the full filter."""
    D, M, T = 4, 3, 10
    key = jr.key(1)
    A = 0.8 * jnp.eye(D) + 0.05 * jr.normal(jr.fold_in(key, 0), (D, D))
    H = jr.normal(jr.fold_in(key, 1), (M, D))
    Q = random_pd_matrix(jr.fold_in(key, 2), D)
    R = random_pd_matrix(jr.fold_in(key, 3), M)
    P0 = random_pd_matrix(jr.fold_in(key, 4), D)
    m0 = jr.normal(jr.fold_in(key, 5), (D,))
    y = jr.normal(jr.fold_in(key, 6), (T, M))

    full = kalman_filter(A, H, Q, R, y, m0, P0)
    mf = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=D)

    assert tree_allclose(mf.filtered_means, full.filtered_means, atol=1e-10)
    assert tree_allclose(mf.filtered_covs, full.filtered_covs, atol=1e-10)
    assert tree_allclose(mf.log_likelihood, full.log_likelihood, atol=1e-10)


def test_loglik_is_sum_of_per_block_logliks():
    """Total log-likelihood == sum of independently filtered block logliks."""
    L, d, m, T = 4, 2, 1, 8
    blocks, (A, H, Q, R, P0), m0, y = _block_system(jr.key(2), L, d, m, T)
    A_b, H_b, Q_b, R_b, P0_b = blocks

    mf = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d)

    per_block = sum(
        kalman_filter(
            A_b[i],
            H_b[i],
            Q_b[i],
            R_b[i],
            y[:, i * m : (i + 1) * m],
            m0[i * d : (i + 1) * d],
            P0_b[i],
        ).log_likelihood
        for i in range(L)
    )
    assert tree_allclose(mf.log_likelihood, per_block, atol=1e-8)


def test_coupled_system_covs_are_block_diagonal():
    """Off-block posterior covariance entries are exact zeros (dropped)."""
    D, d, T = 6, 2, 6
    key = jr.key(3)
    # Dense coupled system: the mean-field projection keeps only the
    # diagonal blocks, so the output covariances must stay block-diagonal.
    A = 0.5 * jr.normal(jr.fold_in(key, 0), (D, D))
    H = jr.normal(jr.fold_in(key, 1), (D, D))
    Q = random_pd_matrix(jr.fold_in(key, 2), D)
    R = random_pd_matrix(jr.fold_in(key, 3), D)
    P0 = random_pd_matrix(jr.fold_in(key, 4), D)
    y = jr.normal(jr.fold_in(key, 5), (T, D))

    mf = meanfield_kalman_filter(A, H, Q, R, y, jnp.zeros(D), P0, block_size=d)

    mask = jnp.kron(jnp.eye(D // d), jnp.ones((d, d)))
    assert jnp.all(mf.filtered_covs * (1 - mask) == 0.0)
    assert jnp.isfinite(mf.log_likelihood)


def test_parallel_matches_sequential():
    """``parallel=True`` (associative scan per block) agrees with the scan."""
    L, d, m, T = 3, 2, 2, 12
    _, (A, H, Q, R, P0), m0, y = _block_system(jr.key(4), L, d, m, T)

    seq = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d)
    par = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d, parallel=True)

    assert tree_allclose(par.filtered_means, seq.filtered_means, rtol=1e-4)
    assert tree_allclose(par.filtered_covs, seq.filtered_covs, rtol=1e-4)
    assert tree_allclose(par.log_likelihood, seq.log_likelihood, rtol=1e-3)


def test_blockdiag_operator_inputs_match_dense():
    """BlockDiag operator inputs split structurally and match dense arrays."""
    L, d, m, T = 3, 2, 2, 10
    blocks, (A, H, Q, R, P0), m0, y = _block_system(jr.key(5), L, d, m, T)
    A_b, H_b, Q_b, R_b, _ = blocks

    A_op = BlockDiag(*[lx.MatrixLinearOperator(A_b[i]) for i in range(L)])
    H_op = BlockDiag(*[lx.MatrixLinearOperator(H_b[i]) for i in range(L)])
    Q_op = BlockDiag(*[lx.MatrixLinearOperator(Q_b[i]) for i in range(L)])
    R_op = BlockDiag(*[lx.MatrixLinearOperator(R_b[i]) for i in range(L)])

    dense = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d)
    op = meanfield_kalman_filter(A_op, H_op, Q_op, R_op, y, m0, P0, block_size=d)

    assert tree_allclose(op.filtered_means, dense.filtered_means, atol=1e-10)
    assert tree_allclose(op.filtered_covs, dense.filtered_covs, atol=1e-10)
    assert tree_allclose(op.log_likelihood, dense.log_likelihood, atol=1e-10)


def test_time_varying_inputs():
    """(T, ...) stacks are split per block and agree with the full filter."""
    L, d, m, T = 2, 2, 1, 8
    key = jr.key(6)
    _, (A, H, Q, R, P0), m0, y = _block_system(key, L, d, m, T)

    # Time-varying block-diagonal stacks built by scaling the TI system.
    scales = 1.0 + 0.1 * jnp.arange(T)
    A_tv = scales[:, None, None] * 0.9 * jnp.broadcast_to(A, (T, L * d, L * d))
    Q_tv = jnp.broadcast_to(Q, (T, L * d, L * d))
    H_tv = jnp.broadcast_to(H, (T, L * m, L * d))
    R_tv = jnp.broadcast_to(R, (T, L * m, L * m))

    full = kalman_filter(A_tv, H_tv, Q_tv, R_tv, y, m0, P0)
    mf = meanfield_kalman_filter(A_tv, H_tv, Q_tv, R_tv, y, m0, P0, block_size=d)

    assert tree_allclose(mf.filtered_means, full.filtered_means, atol=1e-6)
    assert tree_allclose(mf.filtered_covs, full.filtered_covs, atol=1e-6)
    assert tree_allclose(mf.log_likelihood, full.log_likelihood, atol=1e-6)


def test_step_mask_matches_full_filter():
    """A ``(T,)`` step mask gates every block and matches the full filter."""
    L, d, m, T = 2, 2, 2, 10
    _, (A, H, Q, R, P0), m0, y = _block_system(jr.key(7), L, d, m, T)
    mask = jnp.arange(T) % 3 != 0

    full = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
    mf = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d, mask=mask)

    assert tree_allclose(mf.filtered_means, full.filtered_means, atol=1e-6)
    assert tree_allclose(mf.log_likelihood, full.log_likelihood, atol=1e-6)


def test_channel_mask_matches_full_filter():
    """A ``(T, M)`` channel mask is split per block; marginal loglik matches."""
    L, d, m, T = 2, 2, 2, 10
    _, (A, H, Q, R, P0), m0, y = _block_system(jr.key(8), L, d, m, T)
    mask = jr.bernoulli(jr.key(80), 0.7, (T, L * m))

    full = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
    mf = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d, mask=mask)

    assert tree_allclose(mf.filtered_means, full.filtered_means, atol=1e-6)
    assert tree_allclose(mf.log_likelihood, full.log_likelihood, atol=1e-6)


def test_meanfield_smoother_matches_full_rts():
    """Mean-field smoother == full RTS smoother on a decoupled system."""
    L, d, m, T = 3, 2, 2, 12
    _, (A, H, Q, R, P0), m0, y = _block_system(jr.key(9), L, d, m, T)

    full_state = kalman_filter(A, H, Q, R, y, m0, P0)
    full_means, full_covs = rts_smoother(full_state, A, Q)

    mf_state = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d)
    mf_means, mf_covs = meanfield_rts_smoother(mf_state, A, Q, block_size=d)

    assert mf_means.shape == (T, L * d)
    assert mf_covs.shape == (T, L * d, L * d)
    assert tree_allclose(mf_means, full_means, atol=1e-6)
    assert tree_allclose(mf_covs, full_covs, atol=1e-6)


def test_smoother_parallel_matches_sequential():
    """Parallel smoother mode agrees with the sequential scan."""
    L, d, m, T = 2, 2, 2, 10
    _, (A, H, Q, R, P0), m0, y = _block_system(jr.key(10), L, d, m, T)

    state = meanfield_kalman_filter(A, H, Q, R, y, m0, P0, block_size=d)
    seq_means, seq_covs = meanfield_rts_smoother(state, A, Q, block_size=d)
    par_means, par_covs = meanfield_rts_smoother(
        state, A, Q, block_size=d, parallel=True
    )

    assert tree_allclose(par_means, seq_means, rtol=1e-4)
    assert tree_allclose(par_covs, seq_covs, rtol=1e-4)


def test_jit_and_grad():
    """The filter jits and differentiates through the log-likelihood."""
    L, d, m, T = 2, 2, 1, 6
    _, (A, H, Q, R, P0), m0, y = _block_system(jr.key(11), L, d, m, T)

    @jax.jit
    def loglik(A_):
        return meanfield_kalman_filter(
            A_, H, Q, R, y, m0, P0, block_size=d
        ).log_likelihood

    g = jax.grad(loglik)(A)
    assert jnp.all(jnp.isfinite(g))

    # Consume the covariances as a jit *output* so the block-diagonal
    # embedding cannot be dead-code-eliminated from the compiled path.
    @jax.jit
    def covs(A_):
        return meanfield_kalman_filter(A_, H, Q, R, y, m0, P0, block_size=d)

    state = covs(A)
    assert state.filtered_covs.shape == (T, L * d, L * d)
    assert jnp.all(jnp.isfinite(state.filtered_covs))
    assert jnp.all(jnp.isfinite(state.predicted_covs))


def test_invalid_block_size_raises():
    """Indivisible state or observation dimensions are rejected."""
    D, T = 4, 5
    A = jnp.eye(D)
    Q = jnp.eye(D)
    y3 = jnp.zeros((T, 3))
    with pytest.raises(ValueError, match="not divisible by block_size"):
        meanfield_kalman_filter(
            A,
            jnp.eye(D),
            Q,
            jnp.eye(D),
            jnp.zeros((T, D)),
            jnp.zeros(D),
            jnp.eye(D),
            block_size=3,
        )
    with pytest.raises(ValueError, match="Observation dimension"):
        meanfield_kalman_filter(
            A,
            jnp.zeros((3, D)),
            Q,
            jnp.eye(3),
            y3,
            jnp.zeros(D),
            jnp.eye(D),
            block_size=2,
        )
    state = kalman_filter(
        A, jnp.eye(D), Q, jnp.eye(D), jnp.zeros((T, D)), jnp.zeros(D), jnp.eye(D)
    )
    with pytest.raises(ValueError, match="not divisible by block_size"):
        meanfield_rts_smoother(state, A, Q, block_size=3)
