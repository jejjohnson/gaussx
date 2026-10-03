"""Tests for Kalman filter, RTS smoother, and Kalman gain recipes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest
from jax.scipy.stats import multivariate_normal as mvn
from jax.test_util import check_grads

from gaussx import (
    BlockDiag,
    FilterState,
    LowRankUpdate,
    MaternSDE,
    kalman_filter,
    kalman_gain,
    nonlinear_kalman_filter,
    parallel_kalman_filter,
    rts_smoother,
)
from gaussx._einx import rearrange
from gaussx._ssm._utils import _innovation_covariance
from gaussx._testing import random_pd_matrix, tree_allclose


class LazyDiagonal(lx.DiagonalLinearOperator):
    """Test helper that asserts as_matrix is not called in the tested path."""

    def as_matrix(self):
        raise AssertionError("as_matrix should not be called")


def _dense_joint_filtered_mean(A, H, Q, R, y, m0, P0):
    """E[x_T | y_1..y_T] from the dense joint Gaussian (predict first)."""
    T, N = y.shape[0], m0.shape[0]
    means, covs, P = [], [], P0
    m = m0
    for _ in range(T):
        m, P = A @ m, A @ P @ A.T + Q
        means.append(m)
        covs.append(P)
    # Cov(x_j, x_i) = A^{j-i} P_i for j >= i.
    cross = [[None] * T for _ in range(T)]
    for i in range(T):
        block = covs[i]
        for j in range(i, T):
            if j > i:
                block = A @ block
            cross[j][i] = block
            cross[i][j] = block.T
    Sigma = jnp.block(cross)
    H_full = jnp.kron(jnp.eye(T), H)
    S = H_full @ Sigma @ H_full.T + jnp.kron(jnp.eye(T), R)
    mean = jnp.concatenate(means)
    gain_rows = Sigma[-N:] @ H_full.T
    innovation = rearrange(y, "T M -> (T M)") - H_full @ mean
    return means[-1] + gain_rows @ jnp.linalg.solve(S, innovation)


@pytest.mark.slow
def test_kalman_filter_constant_state():
    """The last filtered mean is the exact conditional mean E[x_T | y]."""
    N, M, T = 2, 2, 5
    A = jnp.eye(N)
    H = jnp.eye(M)
    Q = 1e-6 * jnp.eye(N)
    R = 0.1 * jnp.eye(M)

    # Pinned: the check is exactness of the filter, not convergence to the
    # truth, so the noise draw is incidental (gh-412).
    true_state = jnp.array([1.0, 2.0])
    observations = true_state[None, :] + 0.1 * jr.normal(jr.key(0), (T, M))

    x0 = jnp.zeros(N)
    P0 = jnp.eye(N)

    state = kalman_filter(A, H, Q, R, observations, x0, P0)

    assert isinstance(state, FilterState)
    assert state.filtered_means.shape == (T, N)
    assert state.filtered_covs.shape == (T, N, N)
    assert state.log_likelihood.shape == ()

    expected = _dense_joint_filtered_mean(A, H, Q, R, observations, x0, P0)
    assert tree_allclose(state.filtered_means[-1], expected, rtol=1e-10, atol=1e-10)


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
    s_means, s_covs = rts_smoother(state, A)

    assert s_means.shape == (T, N)
    assert s_covs.shape == (T, N, N)


@pytest.mark.slow
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


@pytest.mark.slow
def test_kalman_gain_shape(getkey):
    N, M = 5, 3
    P = lx.MatrixLinearOperator(random_pd_matrix(getkey(), N))
    H = lx.MatrixLinearOperator(jr.normal(getkey(), (M, N)))
    R = lx.MatrixLinearOperator(random_pd_matrix(getkey(), M))

    K = kalman_gain(P, H, R)
    assert K.shape == (N, M)


def _make_woodbury_test_model(getkey, M=512, k=8, T=2):
    A = 0.95 * jnp.eye(k)
    H = jr.normal(getkey(), (M, k)) / jnp.sqrt(k)
    Q = 0.01 * jnp.eye(k)
    R_diag = 0.3 + 0.1 * jnp.linspace(0.0, 1.0, M)
    y = jr.normal(getkey(), (T, M))
    x0 = jnp.zeros(k)
    P0 = jnp.eye(k)
    return A, H, Q, R_diag, y, x0, P0


def test_innovation_covariance_woodbury_returns_low_rank_update(getkey):
    _, H, _, R_diag, _, _, P0 = _make_woodbury_test_model(getkey, M=32, k=4, T=1)
    R = lx.DiagonalLinearOperator(R_diag)

    S = _innovation_covariance(H, P0, R, woodbury=True)

    assert isinstance(S, LowRankUpdate)
    assert S.base is R


@pytest.mark.slow
def test_kalman_filter_woodbury_innovation_diagonal_matches_dense(getkey):
    A, H, Q, R_diag, y, x0, P0 = _make_woodbury_test_model(getkey)
    R = jnp.diag(R_diag)

    ref = kalman_filter(A, H, Q, R, y, x0, P0)
    got = kalman_filter(
        A,
        H,
        Q,
        lx.DiagonalLinearOperator(R_diag),
        y,
        x0,
        P0,
        woodbury_innovation=True,
    )

    assert tree_allclose(got.filtered_means, ref.filtered_means, atol=1e-6, rtol=1e-6)
    assert tree_allclose(got.filtered_covs, ref.filtered_covs, atol=1e-6, rtol=1e-6)
    assert tree_allclose(got.log_likelihood, ref.log_likelihood, atol=1e-6, rtol=1e-6)


@pytest.mark.slow
def test_kalman_filter_woodbury_innovation_blockdiag_matches_dense(getkey):
    A, H, Q, R_diag, y, x0, P0 = _make_woodbury_test_model(getkey)
    R = jnp.diag(R_diag)
    blocks = [
        lx.DiagonalLinearOperator(block)
        for block in jnp.split(R_diag, indices_or_sections=4)
    ]

    ref = kalman_filter(A, H, Q, R, y, x0, P0)
    got = kalman_filter(
        A,
        H,
        Q,
        BlockDiag(*blocks),
        y,
        x0,
        P0,
        woodbury_innovation=True,
    )

    assert tree_allclose(got.filtered_means, ref.filtered_means, atol=1e-6, rtol=1e-6)
    assert tree_allclose(got.filtered_covs, ref.filtered_covs, atol=1e-6, rtol=1e-6)
    assert tree_allclose(got.log_likelihood, ref.log_likelihood, atol=1e-6, rtol=1e-6)


@pytest.mark.slow
def test_kalman_filter_woodbury_innovation_jit_and_grad(getkey):
    A, H, Q, R_diag, y, x0, P0 = _make_woodbury_test_model(getkey, M=16, k=4, T=3)

    def log_likelihood(noise_scale, observations):
        return kalman_filter(
            A,
            H,
            Q,
            lx.DiagonalLinearOperator(noise_scale * R_diag),
            observations,
            x0,
            P0,
            woodbury_innovation=True,
        ).log_likelihood

    jitted_log_likelihood = jax.jit(log_likelihood)(jnp.array(1.0), y)
    noise_scale_gradient = jax.grad(lambda noise_scale: log_likelihood(noise_scale, y))(
        jnp.array(1.0)
    )

    assert jnp.isfinite(jitted_log_likelihood)
    assert jnp.isfinite(noise_scale_gradient)
    check_grads(
        lambda noise_scale: log_likelihood(noise_scale, y),
        (jnp.array(1.0),),
        order=1,
        modes=["rev"],
    )


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

    def test_diagonal_transition_and_obs_avoid_materialization(self, getkey):
        N, T = 3, 6
        A_diag = jnp.array([0.9, 0.8, 0.7])
        H_diag = jnp.array([1.0, 0.5, 1.5])
        A = jnp.diag(A_diag)
        H = jnp.diag(H_diag)
        Q = 0.1 * jnp.eye(N)
        R = 0.5 * jnp.eye(N)
        y = jr.normal(getkey(), (T, N))
        x0, P0 = jnp.zeros(N), jnp.eye(N)

        ref = kalman_filter(A, H, Q, R, y, x0, P0)
        op = kalman_filter(
            LazyDiagonal(A_diag),
            LazyDiagonal(H_diag),
            Q,
            R,
            y,
            x0,
            P0,
        )

        assert tree_allclose(ref.filtered_means, op.filtered_means, rtol=1e-5)
        assert tree_allclose(ref.filtered_covs, op.filtered_covs, rtol=1e-5)
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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
    s_means, s_covs = rts_smoother(state, A_seq)

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


def test_mask_invalid_shape_raises(getkey):
    """Wrong-shape mask must raise a clear ValueError before the scan."""
    import pytest

    N, M, T = 2, 1, 4
    A = jnp.eye(N)
    H = jnp.array([[1.0, 0.0]])
    Q = 0.1 * jnp.eye(N)
    R = 0.2 * jnp.eye(M)
    y = jr.normal(getkey(), (T, M))
    bad_mask = jnp.ones((T - 1,), dtype=bool)
    with pytest.raises(ValueError, match=r"mask must be"):
        kalman_filter(A, H, Q, R, y, jnp.zeros(N), jnp.eye(N), mask=bad_mask)


def test_mask_scalar_broadcasts(getkey):
    """Scalar mask should broadcast across T (ergonomic shortcut)."""
    N, M, T = 2, 1, 4
    A = 0.9 * jnp.eye(N)
    H = jnp.array([[1.0, 0.0]])
    Q = 0.1 * jnp.eye(N)
    R = 0.2 * jnp.eye(M)
    y = jr.normal(getkey(), (T, M))
    full = kalman_filter(A, H, Q, R, y, jnp.zeros(N), jnp.eye(N), mask=jnp.array(True))
    ref = kalman_filter(A, H, Q, R, y, jnp.zeros(N), jnp.eye(N))
    assert tree_allclose(full.filtered_means, ref.filtered_means, rtol=1e-6)


def test_mixed_numpy_3d_with_operator_raises(getkey):
    """numpy.ndarray (not jax.Array) 3D stack must trigger the same TypeError."""
    import numpy as np
    import pytest

    N, M, T = 2, 1, 4
    A_seq_np = np.broadcast_to(0.9 * np.eye(N), (T, N, N))
    H = jnp.array([[1.0, 0.0]])
    Q_op = lx.DiagonalLinearOperator(jnp.array([0.1, 0.2]))
    R = 0.2 * jnp.eye(M)
    y = jr.normal(getkey(), (T, M))
    with pytest.raises(TypeError, match="Time-varying"):
        kalman_filter(A_seq_np, H, Q_op, R, y, jnp.zeros(N), jnp.eye(N))


@pytest.mark.slow
def test_kalman_filter_float32_inputs_stay_float32():
    """float32 inputs must survive the ``lax.cond`` gate under x64.

    The predict-only branch used to return a bare ``jnp.array(0.0)`` — a
    float64 scalar under ``jax_enable_x64`` — which made ``lax.cond``
    reject the branches outright on float32 inputs. Regression for
    gh-219.
    """
    f32 = jnp.float32
    N, M, T = 3, 2, 5

    state = kalman_filter(
        jnp.eye(N, dtype=f32) * 0.9,
        jnp.eye(M, N, dtype=f32),
        (0.05 * jnp.eye(N)).astype(f32),
        (0.1 * jnp.eye(M)).astype(f32),
        jnp.zeros((T, M), dtype=f32),
        jnp.zeros(N, dtype=f32),
        jnp.eye(N, dtype=f32),
    )

    assert state.filtered_means.dtype == f32
    assert state.filtered_covs.dtype == f32
    assert state.predicted_means.dtype == f32
    assert state.predicted_covs.dtype == f32
    assert state.log_likelihood.dtype == f32


def test_kalman_filter_float32_with_partial_mask():
    """The gated path is the one that broke; exercise it explicitly."""
    f32 = jnp.float32
    N, M, T = 3, 2, 6
    mask = jnp.array([True, False, True, True, False, True])

    state = kalman_filter(
        jnp.eye(N, dtype=f32) * 0.9,
        jnp.eye(M, N, dtype=f32),
        (0.05 * jnp.eye(N)).astype(f32),
        (0.1 * jnp.eye(M)).astype(f32),
        jnp.zeros((T, M), dtype=f32),
        jnp.zeros(N, dtype=f32),
        jnp.eye(N, dtype=f32),
        mask=mask,
    )

    assert state.filtered_means.dtype == f32
    assert state.log_likelihood.dtype == f32
    assert jnp.isfinite(state.log_likelihood)


@pytest.mark.parametrize("woodbury", [False, True], ids=["dense", "woodbury"])
def test_float32_covariances_exactly_symmetric(woodbury):
    # gh-388: round-off (and the one-sided Woodbury update) left P slightly
    # asymmetric, which numpyro's positive_definite constraint rejects.
    kern = MaternSDE(
        variance=jnp.array(1.0, dtype=jnp.float32),
        lengthscale=jnp.array(0.3, dtype=jnp.float32),
        order=2,
    )
    A, Q = kern.discretise(jnp.array(0.01, dtype=jnp.float32))
    P0 = kern.sde_params().P_inf
    H = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    R = jnp.array([[0.01]], dtype=jnp.float32)
    # Any data shows the effect; the key is pinned for reproducibility.
    y = jr.normal(jr.key(0), (100, 1), dtype=jnp.float32)
    state = kalman_filter(
        A, H, Q, R, y, jnp.zeros(3, jnp.float32), P0, woodbury_innovation=woodbury
    )
    smoothed = rts_smoother(state, A)[1]
    for P in (state.filtered_covs, state.predicted_covs, smoothed):
        assert P.dtype == jnp.float32
        assert jnp.array_equal(P, jnp.swapaxes(P, -1, -2))


_PF_A = jnp.array([[0.5, 0.2], [0.0, 0.7]])
_PF_H = jnp.array([[1.0, 0.0]])
_PF_Q = 0.3 * jnp.eye(2)
_PF_R = jnp.array([[0.1]])
_PF_M0 = jnp.array([1.0, -1.0])
_PF_P0 = jnp.eye(2)
_PF_Y = jnp.array([[0.4]])

_FILTERS = {
    "kalman_filter": lambda A, Q: kalman_filter(
        A, _PF_H, Q, _PF_R, _PF_Y, _PF_M0, _PF_P0
    ),
    "parallel_covariance": lambda A, Q: parallel_kalman_filter(
        A, _PF_H, Q, _PF_R, _PF_Y, _PF_M0, _PF_P0
    ),
    "parallel_psd_project": lambda A, Q: parallel_kalman_filter(
        A, _PF_H, Q, _PF_R, _PF_Y, _PF_M0, _PF_P0, psd_project=True
    ),
}


@pytest.mark.parametrize("name", [*_FILTERS, "nonlinear_kalman_filter"])
def test_first_observation_sees_one_transition(name):
    # gh-346: every filter predicts before it updates, so observations[0]
    # is scored against H A x₀, not H x₀.
    if name == "nonlinear_kalman_filter":
        state = nonlinear_kalman_filter(
            lambda x: _PF_A @ x,
            lambda x: _PF_H @ x,
            _PF_Q,
            _PF_R,
            _PF_Y,
            _PF_M0,
            _PF_P0,
        )
    else:
        state = _FILTERS[name](_PF_A, _PF_Q)
    P_pred = _PF_A @ _PF_P0 @ _PF_A.T + _PF_Q
    expected = mvn.logpdf(
        _PF_Y[0], _PF_H @ _PF_A @ _PF_M0, _PF_H @ P_pred @ _PF_H.T + _PF_R
    )
    assert jnp.allclose(state.log_likelihood, expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(state.predicted_means[0], _PF_A @ _PF_M0, atol=1e-12)


@pytest.mark.parametrize("name", list(_FILTERS))
def test_identity_first_transition_observes_the_prior(name):
    # The documented recipe: A₀ = I, Q₀ = 0 scores observations[0] against x₀.
    state = _FILTERS[name](jnp.eye(2)[None], jnp.zeros((1, 2, 2)))
    expected = mvn.logpdf(_PF_Y[0], _PF_H @ _PF_M0, _PF_H @ _PF_P0 @ _PF_H.T + _PF_R)
    assert jnp.allclose(state.log_likelihood, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_kalman_filter_grad_matches_fd():
    # gh-412: a fast-tier gradient check on a genuinely time-varying model.
    T, N, M = 6, 3, 2
    k = jr.split(jr.key(0), 5)
    A = 0.8 * jnp.eye(N) + 0.1 * jr.normal(k[0], (T, N, N))
    H = jr.normal(k[1], (T, M, N))
    Lq = 0.3 * jr.normal(k[2], (T, N, N))
    Q = Lq @ jnp.swapaxes(Lq, -1, -2) + 0.1 * jnp.eye(N)
    R = 0.3 * jnp.eye(M)
    y = jr.normal(k[3], (T, M))
    m0 = jr.normal(k[4], (N,))

    def ll(q_scale, r_scale):
        return kalman_filter(
            A, H, q_scale * Q, r_scale * R, y, m0, jnp.eye(N)
        ).log_likelihood

    check_grads(ll, (1.0, 1.0), order=1, modes=["rev"])
    grad = jax.grad(ll, argnums=(0, 1))(1.0, 1.0)
    h = 1e-5
    fd = (
        (ll(1.0 + h, 1.0) - ll(1.0 - h, 1.0)) / (2 * h),
        (ll(1.0, 1.0 + h) - ll(1.0, 1.0 - h)) / (2 * h),
    )
    for g, f in zip(grad, fd, strict=True):
        assert jnp.allclose(g, f, rtol=1e-6)
