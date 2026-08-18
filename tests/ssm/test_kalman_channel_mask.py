"""Per-dimension ``(T, M)`` observation mask for the Kalman filters.

The masked update is verified against two independent references:

- a **row-deleted** filter that physically drops the unobserved rows of
  ``H`` / ``R`` / ``y`` at each step, and
- the **dense** ``(TM, TM)`` joint Gaussian, restricted to the observed
  entries.

The log-likelihood correction gets its own regression tests. The dummy
unit block contributes a spurious ``-0.5 * log(2 pi)`` per masked channel;
under a *fixed* mask that is a constant offset, so it never changes an
argmax and a fixed-mask test passes with the bug present. The tests below
vary the mask and pin the offset exactly.

Sequential-filter correctness stays in the fast lane; the
`parallel_kalman_filter` cases are marked ``slow`` (the associative-scan
trace alone costs ~10 s to compile), matching how ``test_parallel_kalman.py``
already marks its suite.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gaussx import (
    kalman_filter,
    parallel_kalman_filter,
    parallel_rts_smoother,
    rts_smoother,
)


_LOG_2PI = float(np.log(2.0 * np.pi))


class _RowDeleted(NamedTuple):
    """Moments and log-likelihood of the row-deleted reference filter."""

    means: np.ndarray
    covs: np.ndarray
    pred_means: np.ndarray
    pred_covs: np.ndarray
    log_likelihood: float


def _tol(reference, rtol=1e-14):
    """Machine-precision tolerance scaled to the reference's magnitude.

    ``getkey`` reseeds per run, so a bare absolute bound is a function of
    the draw: filtered covariances here range over roughly ``1``-``20``.
    Scaling keeps the ``1e-14`` target meaningful across draws.
    """
    return rtol * max(1.0, float(np.abs(np.asarray(reference)).max()))


def _make_model(key, N=3, M=4, T=10):
    """A well-conditioned, genuinely coupled time-invariant LGSSM."""
    k = jr.split(key, 7)
    A = 0.85 * jnp.eye(N) + 0.05 * jr.normal(k[0], (N, N))
    H = jr.normal(k[1], (M, N))
    Q_h = jr.normal(k[2], (N, N))
    Q = Q_h @ Q_h.T + 0.3 * jnp.eye(N)
    R_h = jr.normal(k[3], (M, M))
    R = R_h @ R_h.T + 0.5 * jnp.eye(M)
    m0 = jr.normal(k[4], (N,))
    P0_h = jr.normal(k[5], (N, N))
    P0 = P0_h @ P0_h.T + jnp.eye(N)
    y = jr.normal(k[6], (T, M))
    return A, H, Q, R, y, m0, P0


def _partial_mask(key, T, M):
    """A mask with a fully-masked row, a fully-observed row, and gaps."""
    mask = jr.bernoulli(key, 0.6, (T, M))
    return mask.at[2].set(False).at[5].set(True)


def _row_deleted_filter(A, H, Q, R, y, m0, P0, mask):
    """NumPy reference that physically deletes unobserved rows."""
    A, H, Q, R, y, mask = (np.asarray(v) for v in (A, H, Q, R, y, mask))
    x, P, ll = np.asarray(m0), np.asarray(P0), 0.0
    means, covs, pred_means, pred_covs = [], [], [], []
    for t in range(y.shape[0]):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        pred_means.append(x_pred)
        pred_covs.append(P_pred)
        idx = np.where(mask[t])[0]
        if len(idx):
            H_s, R_s = H[idx], R[np.ix_(idx, idx)]
            v = y[t][idx] - H_s @ x_pred
            S = H_s @ P_pred @ H_s.T + R_s
            K = P_pred @ H_s.T @ np.linalg.inv(S)
            x = x_pred + K @ v
            P = P_pred - K @ S @ K.T
            ll += -0.5 * (
                v @ np.linalg.solve(S, v)
                + np.linalg.slogdet(S)[1]
                + len(idx) * _LOG_2PI
            )
        else:
            x, P = x_pred, P_pred
        means.append(x)
        covs.append(P)
    return _RowDeleted(
        np.array(means),
        np.array(covs),
        np.array(pred_means),
        np.array(pred_covs),
        ll,
    )


def _rts_reference(ref, A):
    """NumPy RTS smoother over reference filter moments."""
    A = np.asarray(A)
    T = ref.means.shape[0]
    s_means = [None] * T
    s_covs = [None] * T
    s_means[T - 1], s_covs[T - 1] = ref.means[T - 1], ref.covs[T - 1]
    for t in range(T - 2, -1, -1):
        G = ref.covs[t] @ A.T @ np.linalg.inv(ref.pred_covs[t + 1])
        s_means[t] = ref.means[t] + G @ (s_means[t + 1] - ref.pred_means[t + 1])
        s_covs[t] = ref.covs[t] + G @ (s_covs[t + 1] - ref.pred_covs[t + 1]) @ G.T
    return np.array(s_means), np.array(s_covs)


def _dummy_block_ll(A, H, Q, R, y, m0, P0, mask, r_dummy, *, corrected):
    """Masked-``H`` filter with an arbitrary dummy variance ``r_dummy``.

    With ``corrected=False`` this is the naive formulation carrying the
    likelihood bug; with ``corrected=True`` it strips the dummy block's
    contribution per step and must agree with `kalman_filter` for
    every ``r_dummy``.
    """
    A, H, Q, R, y, mask = (np.asarray(v) for v in (A, H, Q, R, y, mask))
    M = y.shape[1]
    x, P, ll = np.asarray(m0), np.asarray(P0), 0.0
    for t in range(y.shape[0]):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        mk = mask[t].astype(float)
        H_m = H * mk[:, None]
        R_m = R * (mk[:, None] * mk[None, :]) + np.diag(r_dummy * (1.0 - mk))
        v = (y[t] - H_m @ x_pred) * mk
        S = H_m @ P_pred @ H_m.T + R_m
        K = P_pred @ H_m.T @ np.linalg.inv(S)
        x = x_pred + K @ v
        P = P_pred - K @ S @ K.T
        ll += -0.5 * (
            v @ np.linalg.solve(S, v) + np.linalg.slogdet(S)[1] + M * _LOG_2PI
        )
        if corrected:
            ll += 0.5 * (M - mk.sum()) * (_LOG_2PI + np.log(r_dummy))
    return ll


def _dense_joint(A, H, Q, R, m0, P0, T):
    """Mean and covariance of the dense ``(TM, TM)`` joint over ``y``."""
    A, H, Q, R, m0, P0 = (np.asarray(v) for v in (A, H, Q, R, m0, P0))
    N, M = A.shape[0], H.shape[0]
    means_x = np.zeros((T, N))
    covs_x = np.zeros((T, N, N))
    m, P = m0, P0
    for t in range(T):
        m = A @ m
        P = A @ P @ A.T + Q
        means_x[t], covs_x[t] = m, P
    mu = (H @ means_x.T).T.reshape(-1)
    Sigma = np.zeros((T * M, T * M))
    for s in range(T):
        for t in range(s, T):
            # Cov(x_s, x_t) = P_s (A^{t-s})^T for s <= t.
            block = H @ covs_x[s] @ np.linalg.matrix_power(A, t - s).T @ H.T
            if s == t:
                block = block + R
            Sigma[s * M : (s + 1) * M, t * M : (t + 1) * M] = block
            Sigma[t * M : (t + 1) * M, s * M : (s + 1) * M] = block.T
    return mu, Sigma


def _mvn_logpdf(x, mu, Sigma):
    residual = x - mu
    return -0.5 * (
        residual @ np.linalg.solve(Sigma, residual)
        + np.linalg.slogdet(Sigma)[1]
        + len(residual) * _LOG_2PI
    )


class TestRowDeletedEquivalence:
    """The masked filter must reproduce a row-deleted filter exactly."""

    def test_filtered_means(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        ref = _row_deleted_filter(A, H, Q, R, y, m0, P0, mask)
        assert jnp.abs(out.filtered_means - ref.means).max() <= _tol(ref.means)

    def test_filtered_covs(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        ref = _row_deleted_filter(A, H, Q, R, y, m0, P0, mask)
        assert jnp.abs(out.filtered_covs - ref.covs).max() <= _tol(ref.covs)

    def test_log_likelihood(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        ref = _row_deleted_filter(A, H, Q, R, y, m0, P0, mask)
        assert abs(float(out.log_likelihood) - ref.log_likelihood) <= 1e-13

    def test_matches_dense_joint_marginal(self, getkey):
        """The exact marginal of the dense joint over observed entries."""
        A, H, Q, R, y, m0, P0 = _make_model(getkey(), T=8)
        T, M = y.shape
        mask = _partial_mask(getkey(), T, M)
        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)

        mu, Sigma = _dense_joint(A, H, Q, R, m0, P0, T)
        idx = np.where(np.asarray(mask).reshape(-1))[0]
        exact = _mvn_logpdf(
            np.asarray(y).reshape(-1)[idx], mu[idx], Sigma[np.ix_(idx, idx)]
        )
        assert abs(float(out.log_likelihood) - exact) <= 1e-13


class TestLikelihoodCorrection:
    """Regression tests for the dummy-block likelihood bug."""

    @pytest.mark.parametrize("r_dummy", [1.0, 4.0, 100.0])
    def test_invariant_to_dummy_variance(self, getkey, r_dummy):
        """The corrected likelihood must not depend on the dummy variance."""
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        corrected = _dummy_block_ll(
            A, H, Q, R, y, m0, P0, mask, r_dummy, corrected=True
        )
        assert abs(float(out.log_likelihood) - corrected) <= 1e-13

    def test_gap_from_naive_is_exactly_n_missing_half_log_2pi(self, getkey):
        """Without the correction the likelihood over-counts by a fixed offset."""
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        n_missing = int((~mask).sum())
        assert n_missing > 0

        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        naive = _dummy_block_ll(A, H, Q, R, y, m0, P0, mask, 1.0, corrected=False)
        gap = float(out.log_likelihood) - naive
        assert gap == pytest.approx(n_missing * 0.5 * _LOG_2PI, abs=1e-12)

    def test_varying_mask_does_not_reward_incomplete_series(self, getkey):
        """The bug makes the LL scale with the number of *missing* channels.

        Under a fixed mask that is invisible (constant offset). Across a
        batch with differing completeness it becomes a per-example bias,
        so compare the naive gap between two masks of different density.
        """
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        T, M = y.shape
        dense_mask = jnp.ones((T, M), dtype=bool).at[0, 0].set(False)
        sparse_mask = _partial_mask(getkey(), T, M)

        ll_dense = float(
            kalman_filter(A, H, Q, R, y, m0, P0, mask=dense_mask).log_likelihood
        )
        ll_sparse = float(
            kalman_filter(A, H, Q, R, y, m0, P0, mask=sparse_mask).log_likelihood
        )
        naive_dense = _dummy_block_ll(
            A, H, Q, R, y, m0, P0, dense_mask, 1.0, corrected=False
        )
        naive_sparse = _dummy_block_ll(
            A, H, Q, R, y, m0, P0, sparse_mask, 1.0, corrected=False
        )
        # The naive bias differs between the two masks; the corrected
        # values carry no such offset.
        naive_bias = (naive_dense - ll_dense) - (naive_sparse - ll_sparse)
        n_gap = int((~dense_mask).sum()) - int((~sparse_mask).sum())
        assert naive_bias == pytest.approx(-n_gap * 0.5 * _LOG_2PI, abs=1e-12)
        assert abs(naive_bias) > 1.0


class TestEdgeCases:
    def test_fully_unobserved_step_is_predict_only(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        T, M = y.shape
        mask = jnp.ones((T, M), dtype=bool).at[3].set(False)
        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)

        assert jnp.isfinite(out.filtered_means).all()
        assert jnp.isfinite(out.filtered_covs).all()
        assert jnp.isfinite(out.log_likelihood)
        assert jnp.allclose(out.filtered_means[3], out.predicted_means[3])
        assert jnp.allclose(out.filtered_covs[3], out.predicted_covs[3])

    def test_all_true_channel_mask_matches_unmasked(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        T, M = y.shape
        unmasked = kalman_filter(A, H, Q, R, y, m0, P0)
        masked = kalman_filter(A, H, Q, R, y, m0, P0, mask=jnp.ones((T, M), dtype=bool))
        assert masked.log_likelihood == unmasked.log_likelihood
        assert jnp.array_equal(masked.filtered_means, unmasked.filtered_means)
        assert jnp.array_equal(masked.filtered_covs, unmasked.filtered_covs)

    def test_broadcast_step_mask_matches_step_gate(self, getkey):
        """An all-False row is equivalent to a False entry in the (T,) form."""
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        T, M = y.shape
        step_mask = jnp.ones((T,), dtype=bool).at[1].set(False).at[4].set(False)
        gated = kalman_filter(A, H, Q, R, y, m0, P0, mask=step_mask)
        per_channel = kalman_filter(
            A,
            H,
            Q,
            R,
            y,
            m0,
            P0,
            mask=jnp.broadcast_to(step_mask[:, None], (T, M)),
        )
        assert abs(float(gated.log_likelihood - per_channel.log_likelihood)) <= 1e-13
        assert jnp.abs(gated.filtered_means - per_channel.filtered_means).max() <= 1e-14

    def test_masked_entries_may_be_nan(self, getkey):
        """Unobserved slots are never read, so NaN there is harmless."""
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        clean = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        poisoned = kalman_filter(
            A, H, Q, R, jnp.where(mask, y, jnp.nan), m0, P0, mask=mask
        )
        assert poisoned.log_likelihood == clean.log_likelihood
        assert jnp.array_equal(poisoned.filtered_means, clean.filtered_means)

    def test_wrong_shape_channel_mask_raises(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        T, M = y.shape
        with pytest.raises(ValueError, match=r"mask must be"):
            kalman_filter(A, H, Q, R, y, m0, P0, mask=jnp.ones((T, M + 1), dtype=bool))

    @pytest.mark.slow
    def test_float32_tracks_row_deleted_reference(self, getkey):
        """Tight noise in float32 is where the dummy block could bite.

        The bound is set from a 30-seed sweep: relative log-likelihood
        error is ~3e-6 median with a 1.3e-5 tail, which is float32
        accumulation over 60 steps rather than a defect. ``5e-5`` clears
        that tail with room to spare while staying orders of magnitude
        below what this test exists to catch — the missing likelihood
        correction shifts this model by ~100 nats, a relative error of
        order 1.
        """
        T, M, N = 60, 6, 3
        k = jr.split(getkey(), 4)
        A = (0.95 * jnp.eye(N) + 0.01 * jr.normal(k[0], (N, N))).astype(jnp.float32)
        H = jr.normal(k[1], (M, N)).astype(jnp.float32)
        Q = (1e-4 * jnp.eye(N)).astype(jnp.float32)
        R = (1e-3 * jnp.eye(M)).astype(jnp.float32)
        m0 = jnp.zeros((N,), dtype=jnp.float32)
        P0 = jnp.eye(N, dtype=jnp.float32)
        y = (0.1 * jr.normal(k[2], (T, M))).astype(jnp.float32)
        mask = jr.bernoulli(k[3], 0.7, (T, M))

        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        assert out.log_likelihood.dtype == jnp.float32

        ref = _row_deleted_filter(
            *(np.asarray(v, dtype=np.float64) for v in (A, H, Q, R, y, m0, P0)),
            np.asarray(mask),
        )
        ref_ll = ref.log_likelihood
        rel = abs(float(out.log_likelihood) - ref_ll) / abs(ref_ll)
        assert rel <= 5e-5
        assert np.abs(np.asarray(out.filtered_means) - ref.means).mean() <= 1e-5

    @pytest.mark.slow
    def test_jit_and_grad(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])

        def loss(H_param):
            return -kalman_filter(A, H_param, Q, R, y, m0, P0, mask=mask).log_likelihood

        grad = jax.jit(jax.grad(loss))(H)
        assert jnp.isfinite(grad).all()
        assert jnp.abs(grad).max() > 0.0


class TestBackCompat:
    def test_step_mask_path_unchanged(self, getkey):
        """A genuinely partial (T,) mask must still take the gated path."""
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        T = y.shape[0]
        mask = jnp.ones((T,), dtype=bool).at[1].set(False).at[6].set(False)
        out = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)

        ref_mask = jnp.broadcast_to(mask[:, None], y.shape)
        ref = _row_deleted_filter(A, H, Q, R, y, m0, P0, ref_mask)
        assert abs(float(out.log_likelihood) - ref.log_likelihood) <= 1e-13
        # Masked steps are predict-only.
        for t in (1, 6):
            assert jnp.allclose(out.filtered_means[t], out.predicted_means[t])


class TestSmoother:
    """`rts_smoother` needs no mask of its own — it reads only moments."""

    def test_matches_row_deleted_reference(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        filtered = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        s_means, s_covs = rts_smoother(filtered, A, Q)

        ref = _row_deleted_filter(A, H, Q, R, y, m0, P0, mask)
        ref_means, ref_covs = _rts_reference(ref, A)
        assert jnp.abs(s_means - ref_means).max() <= _tol(ref_means, 1e-12)
        assert jnp.abs(s_covs - ref_covs).max() <= _tol(ref_covs, 1e-12)

    def test_fully_unobserved_step_smooths_without_nan(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        T, M = y.shape
        mask = jnp.ones((T, M), dtype=bool).at[4].set(False)
        filtered = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        s_means, s_covs = rts_smoother(filtered, A, Q)
        assert jnp.isfinite(s_means).all()
        assert jnp.isfinite(s_covs).all()

    @pytest.mark.slow
    def test_parallel_smoother_agrees(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        seq = rts_smoother(kalman_filter(A, H, Q, R, y, m0, P0, mask=mask), A, Q)
        par = parallel_rts_smoother(
            parallel_kalman_filter(A, H, Q, R, y, m0, P0, mask=mask), A, Q
        )
        assert jnp.abs(seq[0] - par[0]).max() <= 1e-11
        assert jnp.abs(seq[1] - par[1]).max() <= 1e-11


class TestParallelParity:
    @pytest.mark.slow
    def test_matches_sequential(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        seq = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        par = parallel_kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)

        assert jnp.abs(seq.filtered_means - par.filtered_means).max() <= 1e-12
        assert jnp.abs(seq.filtered_covs - par.filtered_covs).max() <= 1e-12
        assert abs(float(seq.log_likelihood - par.log_likelihood)) <= 1e-12

    @pytest.mark.slow
    def test_partially_observed_first_step(self, getkey):
        """Step 0 absorbs the prior, so a partial mask there is its own case."""
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        T, M = y.shape
        mask = jnp.ones((T, M), dtype=bool).at[0].set(jnp.arange(M) % 2 == 0)
        seq = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        par = parallel_kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        assert jnp.abs(seq.filtered_means - par.filtered_means).max() <= 1e-12
        assert abs(float(seq.log_likelihood - par.log_likelihood)) <= 1e-12

    @pytest.mark.slow
    def test_woodbury_innovation_delegates(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        seq = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask)
        wood = kalman_filter(A, H, Q, R, y, m0, P0, mask=mask, woodbury_innovation=True)
        assert abs(float(seq.log_likelihood - wood.log_likelihood)) <= 1e-11

    def test_sqrt_form_rejects_channel_mask(self, getkey):
        A, H, Q, R, y, m0, P0 = _make_model(getkey())
        mask = _partial_mask(getkey(), y.shape[0], y.shape[1])
        with pytest.raises(NotImplementedError, match=r"per-channel"):
            parallel_kalman_filter(A, H, Q, R, y, m0, P0, mask=mask, form="sqrt")
