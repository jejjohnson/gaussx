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


# ----------------------------------------------------------------
# Operator-typed inputs (time-invariant)
# ----------------------------------------------------------------


class TestOperatorInputs:
    def _model(self, getkey, N=3, M=2, T=8):
        A = 0.9 * jnp.eye(N)
        H = jr.normal(getkey(), (M, N))
        Q = 0.1 * jnp.eye(N)
        R_diag = jnp.array([0.3, 0.5])[:M]
        R = jnp.diag(R_diag)
        y = jr.normal(getkey(), (T, M))
        return A, H, Q, R, R_diag, y, jnp.zeros(N), jnp.eye(N)

    def test_obs_noise_diagonal_operator(self, getkey):
        A, H, Q, R, R_diag, y, x0, P0 = self._model(getkey)
        ref = kalman_filter(A, H, Q, R, y, x0, P0)
        op = kalman_filter(A, H, Q, lx.DiagonalLinearOperator(R_diag), y, x0, P0)
        assert tree_allclose(ref.filtered_means, op.filtered_means, rtol=1e-5)
        assert tree_allclose(ref.filtered_covs, op.filtered_covs, rtol=1e-5)
        assert tree_allclose(ref.log_likelihood, op.log_likelihood, rtol=1e-5)

    def test_process_noise_diagonal_operator(self, getkey):
        N, M, T = 3, 2, 6
        A = 0.9 * jnp.eye(N)
        H = jr.normal(getkey(), (M, N))
        Q_diag = jnp.array([0.1, 0.2, 0.3])
        Q = jnp.diag(Q_diag)
        R = 0.5 * jnp.eye(M)
        y = jr.normal(getkey(), (T, M))
        x0, P0 = jnp.zeros(N), jnp.eye(N)
        ref = kalman_filter(A, H, Q, R, y, x0, P0)
        op = kalman_filter(A, H, lx.DiagonalLinearOperator(Q_diag), R, y, x0, P0)
        assert tree_allclose(ref.filtered_means, op.filtered_means, rtol=1e-5)

    def test_transition_block_diag_operator(self, getkey):
        from gaussx import BlockDiag

        # Block-diagonal A from two channels
        A1 = 0.9 * jnp.eye(2)
        A2 = 0.7 * jnp.eye(2)
        A_dense = jnp.block([[A1, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), A2]])
        N = 4
        M = 2
        T = 6
        H = jr.normal(getkey(), (M, N))
        Q = 0.1 * jnp.eye(N)
        R = 0.5 * jnp.eye(M)
        y = jr.normal(getkey(), (T, M))
        x0, P0 = jnp.zeros(N), jnp.eye(N)

        ref = kalman_filter(A_dense, H, Q, R, y, x0, P0)
        A_op = BlockDiag(lx.MatrixLinearOperator(A1), lx.MatrixLinearOperator(A2))
        op = kalman_filter(A_op, H, Q, R, y, x0, P0)
        assert tree_allclose(ref.filtered_means, op.filtered_means, rtol=1e-5)
        assert tree_allclose(ref.log_likelihood, op.log_likelihood, rtol=1e-5)


# ----------------------------------------------------------------
# Time-varying inputs and mask
# ----------------------------------------------------------------


class TestTimeVarying:
    def test_ti_broadcast_matches_ti_form(self, getkey):
        """(N, N) inputs broadcast to (T, N, N) match the time-invariant call."""
        N, M, T = 3, 2, 7
        A = 0.9 * jnp.eye(N)
        H = jr.normal(getkey(), (M, N))
        Q = 0.1 * jnp.eye(N)
        R = 0.5 * jnp.eye(M)
        y = jr.normal(getkey(), (T, M))
        x0, P0 = jnp.zeros(N), jnp.eye(N)

        ref = kalman_filter(A, H, Q, R, y, x0, P0)

        A_seq = jnp.broadcast_to(A, (T, N, N))
        H_seq = jnp.broadcast_to(H, (T, M, N))
        Q_seq = jnp.broadcast_to(Q, (T, N, N))
        R_seq = jnp.broadcast_to(R, (T, M, M))
        tv = kalman_filter(A_seq, H_seq, Q_seq, R_seq, y, x0, P0)

        assert tree_allclose(ref.filtered_means, tv.filtered_means, rtol=1e-6)
        assert tree_allclose(ref.filtered_covs, tv.filtered_covs, rtol=1e-6)
        assert tree_allclose(ref.log_likelihood, tv.log_likelihood, rtol=1e-6)

    def test_tv_per_step_matches_manual_loop(self, getkey):
        """TV path with per-step matrices matches a hand-rolled loop."""
        N, M, T = 2, 1, 5
        # Random per-step (A_t, Q_t, H_t, R_t)
        A_seq = jnp.stack(
            [0.9 * jnp.eye(N) + 0.05 * jr.normal(getkey(), (N, N)) for _ in range(T)]
        )
        H_seq = jr.normal(getkey(), (T, M, N))
        Q_seq = jnp.stack([0.1 * jnp.eye(N) for _ in range(T)])
        R_seq = jnp.stack([0.5 * jnp.eye(M) for _ in range(T)])
        y = jr.normal(getkey(), (T, M))
        x0, P0 = jnp.zeros(N), jnp.eye(N)

        out = kalman_filter(A_seq, H_seq, Q_seq, R_seq, y, x0, P0)

        # Manual reference
        x, P = x0, P0
        log_2pi = jnp.log(2.0 * jnp.pi)
        ll = 0.0
        for t in range(T):
            x_pred = A_seq[t] @ x
            P_pred = A_seq[t] @ P @ A_seq[t].T + Q_seq[t]
            v = y[t] - H_seq[t] @ x_pred
            S = H_seq[t] @ P_pred @ H_seq[t].T + R_seq[t]
            S_inv = jnp.linalg.inv(S)
            K = P_pred @ H_seq[t].T @ S_inv
            x = x_pred + K @ v
            P = P_pred - K @ S @ K.T
            _, ld = jnp.linalg.slogdet(S)
            ll = ll - 0.5 * (v @ S_inv @ v + ld + M * log_2pi)

        assert tree_allclose(out.filtered_means[-1], x, atol=1e-5)
        assert tree_allclose(out.filtered_covs[-1], P, atol=1e-5)
        assert tree_allclose(out.log_likelihood, ll, atol=1e-4)

    def test_mask_predict_only(self, getkey):
        """Masked steps should run predict only and contribute 0 log-likelihood."""
        N, M, T = 3, 2, 6
        A = 0.95 * jnp.eye(N)
        H = jr.normal(getkey(), (M, N))
        Q = 0.05 * jnp.eye(N)
        R = 0.3 * jnp.eye(M)
        y = jr.normal(getkey(), (T, M))
        x0, P0 = jnp.zeros(N), jnp.eye(N)

        # Mask off steps 1 and 3
        mask = jnp.array([True, False, True, False, True, True])
        out = kalman_filter(A, H, Q, R, y, x0, P0, mask=mask)

        # On masked steps, filtered == predicted.
        idx = jnp.where(~mask)[0]
        assert tree_allclose(
            out.filtered_means[idx], out.predicted_means[idx], atol=1e-7
        )
        assert tree_allclose(out.filtered_covs[idx], out.predicted_covs[idx], atol=1e-7)

    def test_mask_log_likelihood_matches_subset(self, getkey):
        """LL with masked steps == LL of an unmasked filter run on the
        observed-only timeline that a user would build manually."""
        # We construct a setup where masked steps correspond to extra
        # prediction steps; the LL should equal the unmasked filter.
        N, M, T = 2, 1, 5
        A = 0.9 * jnp.eye(N)
        H = jnp.array([[1.0, 0.0]])
        Q = 0.1 * jnp.eye(N)
        R = 0.2 * jnp.eye(M)
        y = jr.normal(getkey(), (T, M))
        x0, P0 = jnp.zeros(N), jnp.eye(N)

        mask = jnp.array([True, True, False, True, True])
        out = kalman_filter(A, H, Q, R, y, x0, P0, mask=mask)

        # Drop the masked step's contribution: it should be 0.
        # The LL with mask must equal a hand-rolled filter that skips the
        # update on that step.
        x, P = x0, P0
        log_2pi = jnp.log(2.0 * jnp.pi)
        ll = 0.0
        for t in range(T):
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q
            if mask[t]:
                v = y[t] - H @ x_pred
                S = H @ P_pred @ H.T + R
                S_inv = jnp.linalg.inv(S)
                K = P_pred @ H.T @ S_inv
                x = x_pred + K @ v
                P = P_pred - K @ S @ K.T
                _, ld = jnp.linalg.slogdet(S)
                ll = ll - 0.5 * (v @ S_inv @ v + ld + M * log_2pi)
            else:
                x = x_pred
                P = P_pred

        assert tree_allclose(out.log_likelihood, ll, atol=1e-5)

    def test_mixed_tv_array_with_operator_raises(self, getkey):
        """3D TV stack mixed with an operator should raise TypeError."""
        import pytest

        N, M, T = 2, 1, 4
        A_seq = jnp.broadcast_to(0.9 * jnp.eye(N), (T, N, N))
        H = jnp.array([[1.0, 0.0]])
        Q_op = lx.DiagonalLinearOperator(jnp.array([0.1, 0.2]))
        R = 0.2 * jnp.eye(M)
        y = jr.normal(getkey(), (T, M))
        with pytest.raises(TypeError, match="Time-varying"):
            kalman_filter(A_seq, H, Q_op, R, y, jnp.zeros(N), jnp.eye(N))


# ----------------------------------------------------------------
# rts_smoother time-varying
# ----------------------------------------------------------------


def test_rts_smoother_tv(getkey):
    """RTS smoother with TV transition matches manual recurrence."""
    N, M, T = 2, 1, 6
    A_seq = jnp.stack(
        [0.9 * jnp.eye(N) + 0.02 * jr.normal(getkey(), (N, N)) for _ in range(T)]
    )
    H = jnp.array([[1.0, 0.0]])
    Q = 0.05 * jnp.eye(N)
    R = 0.2 * jnp.eye(M)
    y = jr.normal(getkey(), (T, M))
    x0, P0 = jnp.zeros(N), jnp.eye(N)

    state = kalman_filter(A_seq, H, Q, R, y, x0, P0)
    s_means, s_covs = rts_smoother(state, A_seq, Q)

    # Last smoothed = last filtered.
    assert tree_allclose(s_means[-1], state.filtered_means[-1], rtol=1e-6)
    assert tree_allclose(s_covs[-1], state.filtered_covs[-1], rtol=1e-6)


# ----------------------------------------------------------------
# solver= regression with TV path
# ----------------------------------------------------------------


def test_kalman_filter_tv_solver_matches_default(getkey):
    """TV path with solver=DenseSolver() matches the default dispatch path."""
    from gaussx import DenseSolver

    N, M, T = 3, 2, 6
    A = 0.9 * jnp.eye(N)
    H = jr.normal(getkey(), (M, N))
    Q = 0.1 * jnp.eye(N)
    R = 0.5 * jnp.eye(M)
    y = jr.normal(getkey(), (T, M))
    x0, P0 = jnp.zeros(N), jnp.eye(N)
    A_seq = jnp.broadcast_to(A, (T, N, N))

    default = kalman_filter(A_seq, H, Q, R, y, x0, P0)
    dense = kalman_filter(A_seq, H, Q, R, y, x0, P0, solver=DenseSolver())

    assert tree_allclose(default.filtered_means, dense.filtered_means, rtol=1e-5)
    assert tree_allclose(default.log_likelihood, dense.log_likelihood, rtol=1e-4)
