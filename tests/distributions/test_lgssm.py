"""Tests for the LGSSM / MaskedLGSSM / LGSSMFactory distributions."""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np

from gaussx import LGSSM, LGSSMFactory, MaskedLGSSM
from gaussx._operators import Kronecker


_LOG_2PI = float(np.log(2.0 * np.pi))


def _make_params(key, N=3, M=4):
    k = jr.split(key, 6)
    A = 0.85 * jnp.eye(N) + 0.05 * jr.normal(k[0], (N, N))
    H = jr.normal(k[1], (M, N))
    Q_h = jr.normal(k[2], (N, N))
    Q = Q_h @ Q_h.T + 0.3 * jnp.eye(N)
    R_h = jr.normal(k[3], (M, M))
    R = R_h @ R_h.T + 0.5 * jnp.eye(M)
    m0 = jr.normal(k[4], (N,))
    P0_h = jr.normal(k[5], (N, N))
    P0 = P0_h @ P0_h.T + jnp.eye(N)
    return A, H, Q, R, m0, P0


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


class TestLGSSM:
    def test_log_prob_matches_dense_joint(self, getkey):
        T = 8
        A, H, Q, R, m0, P0 = _make_params(getkey())
        d = LGSSM(A, H, Q, R, m0, P0, n_steps=T)
        y = d.sample(getkey())

        mu, Sigma = _dense_joint(A, H, Q, R, m0, P0, T)
        expected = _mvn_logpdf(np.asarray(y).reshape(-1), mu, Sigma)
        assert abs(float(d.log_prob(y)) - expected) <= 1e-13

    def test_log_prob_is_scalar(self, getkey):
        """``event_shape`` rank is what makes this a scalar — easy to get wrong."""
        A, H, Q, R, m0, P0 = _make_params(getkey())
        d = LGSSM(A, H, Q, R, m0, P0, n_steps=6)
        assert d.event_shape == (6, H.shape[0])
        assert d.batch_shape == ()
        assert d.log_prob(d.sample(getkey())).shape == ()

    @pytest.mark.slow
    def test_sample_shapes(self, getkey):
        T, M = 7, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        d = LGSSM(A, H, Q, R, m0, P0, n_steps=T)
        assert d.sample(getkey()).shape == (T, M)
        assert d.sample(getkey(), (5,)).shape == (5, T, M)
        assert d.sample(getkey(), (2, 3)).shape == (2, 3, T, M)

    def test_sample_requires_key(self, getkey):
        A, H, Q, R, m0, P0 = _make_params(getkey())
        d = LGSSM(A, H, Q, R, m0, P0, n_steps=4)
        with pytest.raises(ValueError, match="PRNG key"):
            d.sample(None)

    def test_mean_and_variance_match_dense_joint(self, getkey):
        T = 6
        A, H, Q, R, m0, P0 = _make_params(getkey())
        d = LGSSM(A, H, Q, R, m0, P0, n_steps=T)
        mu, Sigma = _dense_joint(A, H, Q, R, m0, P0, T)

        assert d.mean.shape == d.event_shape
        assert d.variance.shape == d.event_shape
        assert np.abs(np.asarray(d.mean).reshape(-1) - mu).max() <= 1e-12
        variance_flat = np.asarray(d.variance).reshape(-1)
        assert np.abs(variance_flat - np.diag(Sigma)).max() <= 1e-12

    @pytest.mark.slow
    def test_batched_log_prob(self, getkey):
        A, H, Q, R, m0, P0 = _make_params(getkey())
        d = LGSSM(A, H, Q, R, m0, P0, n_steps=5)
        ys = d.sample(getkey(), (4,))
        batched = d.log_prob(ys)
        assert batched.shape == (4,)
        assert jnp.allclose(batched[1], d.log_prob(ys[1]))

    @pytest.mark.slow
    def test_jit_vmap_over_batch(self, getkey):
        A, H, Q, R, m0, P0 = _make_params(getkey())
        d = LGSSM(A, H, Q, R, m0, P0, n_steps=5)
        ys = d.sample(getkey(), (16,))
        out = eqx.filter_jit(jax.vmap(d.log_prob))(ys)
        assert out.shape == (16,)
        assert jnp.isfinite(out).all()


class TestOperatorInputs:
    """Structured operators must be accepted wherever kalman_filter takes them."""

    def _operator_model(self, getkey, T=6):
        """Same model expressed densely and as operators."""
        N1, N2, M1, M2 = 2, 2, 2, 2
        k = jr.split(getkey(), 4)
        A_op = Kronecker(
            lx.MatrixLinearOperator(0.9 * jnp.eye(N1)),
            lx.MatrixLinearOperator(0.8 * jnp.eye(N2)),
        )
        H_op = Kronecker(
            lx.MatrixLinearOperator(jr.normal(k[0], (M1, N1))),
            lx.MatrixLinearOperator(jr.normal(k[1], (M2, N2))),
        )
        Q_op = Kronecker(
            lx.MatrixLinearOperator(0.2 * jnp.eye(N1), lx.positive_semidefinite_tag),
            lx.MatrixLinearOperator(0.3 * jnp.eye(N2), lx.positive_semidefinite_tag),
        )
        R_op = lx.DiagonalLinearOperator(0.4 * jnp.ones(M1 * M2))
        P0 = jnp.eye(N1 * N2)
        m0 = jr.normal(k[2], (N1 * N2,))
        dense = tuple(op.as_matrix() for op in (A_op, H_op, Q_op, R_op))
        return (A_op, H_op, Q_op, R_op, P0, m0), dense, T

    @pytest.mark.slow
    def test_log_prob_matches_dense(self, getkey):
        ops, dense, T = self._operator_model(getkey)
        A_op, H_op, Q_op, R_op, P0, m0 = ops
        A, H, Q, R = dense

        d_op = LGSSM(A_op, H_op, Q_op, R_op, m0, P0, n_steps=T)
        d_dense = LGSSM(A, H, Q, R, m0, P0, n_steps=T)
        assert d_op.event_shape == d_dense.event_shape

        y = d_dense.sample(getkey())
        assert abs(float(d_op.log_prob(y) - d_dense.log_prob(y))) <= 1e-10

    @pytest.mark.slow
    def test_sample_and_moments_match_dense(self, getkey):
        ops, dense, T = self._operator_model(getkey)
        A_op, H_op, Q_op, R_op, P0, m0 = ops
        A, H, Q, R = dense

        d_op = LGSSM(A_op, H_op, Q_op, R_op, m0, P0, n_steps=T)
        d_dense = LGSSM(A, H, Q, R, m0, P0, n_steps=T)

        key = getkey()
        assert jnp.abs(d_op.sample(key) - d_dense.sample(key)).max() <= 1e-10
        assert jnp.abs(d_op.mean - d_dense.mean).max() <= 1e-10
        assert jnp.abs(d_op.variance - d_dense.variance).max() <= 1e-10

    def test_masked_operator_log_prob(self, getkey):
        ops, dense, T = self._operator_model(getkey)
        A_op, H_op, Q_op, R_op, P0, m0 = ops
        A, H, Q, R = dense
        M = H.shape[0]
        mask = jr.bernoulli(getkey(), 0.6, (T, M))

        d_op = MaskedLGSSM(A_op, H_op, Q_op, R_op, m0, P0, n_steps=T, obs_mask=mask)
        d_dense = MaskedLGSSM(A, H, Q, R, m0, P0, n_steps=T, obs_mask=mask)
        y = d_dense.sample(getkey())
        assert abs(float(d_op.log_prob(y) - d_dense.log_prob(y))) <= 1e-10


class TestMaskedLGSSM:
    def test_log_prob_is_exact_marginal(self, getkey):
        T, M = 8, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        mask = jr.bernoulli(getkey(), 0.6, (T, M))
        d = MaskedLGSSM(A, H, Q, R, m0, P0, n_steps=T, obs_mask=mask)
        y = d.sample(getkey())

        mu, Sigma = _dense_joint(A, H, Q, R, m0, P0, T)
        idx = np.where(np.asarray(mask).reshape(-1))[0]
        expected = _mvn_logpdf(
            np.asarray(y).reshape(-1)[idx], mu[idx], Sigma[np.ix_(idx, idx)]
        )
        assert abs(float(d.log_prob(y)) - expected) <= 1e-13

    def test_all_true_mask_equals_lgssm(self, getkey):
        T, M = 8, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        base = LGSSM(A, H, Q, R, m0, P0, n_steps=T)
        masked = MaskedLGSSM(
            A, H, Q, R, m0, P0, n_steps=T, obs_mask=jnp.ones((T, M), dtype=bool)
        )
        y = base.sample(getkey())
        assert masked.log_prob(y) == base.log_prob(y)

    def test_fully_unobserved_timestep(self, getkey):
        T, M = 8, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        mask = jnp.ones((T, M), dtype=bool).at[3].set(False)
        d = MaskedLGSSM(A, H, Q, R, m0, P0, n_steps=T, obs_mask=mask)
        assert jnp.isfinite(d.log_prob(d.sample(getkey())))

    def test_masked_entries_may_be_nan(self, getkey):
        T, M = 8, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        mask = jr.bernoulli(getkey(), 0.6, (T, M))
        d = MaskedLGSSM(A, H, Q, R, m0, P0, n_steps=T, obs_mask=mask)
        y = d.sample(getkey())
        assert d.log_prob(jnp.where(mask, y, jnp.nan)) == d.log_prob(y)

    def test_wrong_mask_shape_raises(self, getkey):
        T, M = 8, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        with pytest.raises(ValueError, match="obs_mask must have shape"):
            MaskedLGSSM(
                A,
                H,
                Q,
                R,
                m0,
                P0,
                n_steps=T,
                obs_mask=jnp.ones((T, M + 1), dtype=bool),
            )

    def test_sample_returns_complete_series(self, getkey):
        """The mask governs scoring, not generation."""
        T, M = 8, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        mask = jnp.zeros((T, M), dtype=bool).at[0, 0].set(True)
        d = MaskedLGSSM(A, H, Q, R, m0, P0, n_steps=T, obs_mask=mask)
        y = d.sample(getkey())
        assert y.shape == (T, M)
        assert jnp.isfinite(y).all()


class TestPytree:
    """Guards against the hand-written ``tree_unflatten`` trap.

    ``equinox.partition`` walks the tree with boolean sentinel leaves, so
    any ``__init__`` that infers shapes from its arguments explodes on
    reconstruction. A ``log_prob``-only suite passes with that bug present.
    """

    @pytest.mark.parametrize("masked", [False, True])
    def test_round_trip(self, getkey, masked):
        T, M = 6, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        if masked:
            d = MaskedLGSSM(
                A,
                H,
                Q,
                R,
                m0,
                P0,
                n_steps=T,
                obs_mask=jr.bernoulli(getkey(), 0.6, (T, M)),
            )
        else:
            d = LGSSM(A, H, Q, R, m0, P0, n_steps=T)
        y = d.sample(getkey())

        leaves, treedef = jax.tree_util.tree_flatten(d)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.event_shape == d.event_shape
        assert rebuilt.log_prob(y) == d.log_prob(y)

    @pytest.mark.parametrize("masked", [False, True])
    def test_eqx_partition(self, getkey, masked):
        T, M = 6, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        if masked:
            d = MaskedLGSSM(
                A,
                H,
                Q,
                R,
                m0,
                P0,
                n_steps=T,
                obs_mask=jr.bernoulli(getkey(), 0.6, (T, M)),
            )
        else:
            d = LGSSM(A, H, Q, R, m0, P0, n_steps=T)

        arrays, static = eqx.partition(d, eqx.is_inexact_array)
        assert len(jax.tree_util.tree_leaves(arrays)) == 6
        assert eqx.combine(arrays, static).event_shape == d.event_shape

    @pytest.mark.slow
    @pytest.mark.parametrize("masked", [False, True])
    def test_filter_grad_reaches_ssm_params(self, getkey, masked):
        T, M = 6, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        if masked:
            d = MaskedLGSSM(
                A,
                H,
                Q,
                R,
                m0,
                P0,
                n_steps=T,
                obs_mask=jr.bernoulli(getkey(), 0.6, (T, M)),
            )
        else:
            d = LGSSM(A, H, Q, R, m0, P0, n_steps=T)
        y = d.sample(getkey())

        grads = eqx.filter_grad(lambda dist: -dist.log_prob(y))(d)
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
        assert len(leaves) == 6  # A, H, Q, R, m0, P0
        assert all(jnp.isfinite(g).all() for g in leaves)
        norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
        assert float(norm) > 0.0


class TestFactory:
    def test_builds_masked_lgssm(self, getkey):
        T, M = 6, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        mask = jr.bernoulli(getkey(), 0.6, (T, M))
        factory = LGSSMFactory(A, H, Q, R, m0, P0, n_steps=T)
        d = factory(mask)

        assert isinstance(d, MaskedLGSSM)
        assert d.event_shape == (T, M)
        direct = MaskedLGSSM(A, H, Q, R, m0, P0, n_steps=T, obs_mask=mask)
        y = direct.sample(getkey())
        assert d.log_prob(y) == direct.log_prob(y)

    @pytest.mark.slow
    def test_filter_grad_through_factory(self, getkey):
        """The whole point of the Module: params stay visible to filter_grad."""
        T, M = 6, 4
        A, H, Q, R, m0, P0 = _make_params(getkey(), M=M)
        mask = jr.bernoulli(getkey(), 0.6, (T, M))
        factory = LGSSMFactory(A, H, Q, R, m0, P0, n_steps=T)
        y = factory(mask).sample(getkey())

        grads = eqx.filter_grad(lambda f: -f(mask).log_prob(y))(factory)
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
        assert len(leaves) == 6
        norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
        assert float(norm) > 0.0


@pytest.mark.slow
def test_sample_moments_match_analytic(getkey):
    """Empirical moments of 200k samples against the analytic joint."""
    T, N, M = 5, 2, 2
    A = jnp.array([[0.8, 0.1], [0.0, 0.7]])
    H = jnp.array([[1.0, 0.3], [0.2, 1.0]])
    Q = 0.2 * jnp.eye(N)
    R = 0.3 * jnp.eye(M)
    m0 = jnp.array([0.5, -0.2])
    P0 = jnp.eye(N)
    d = LGSSM(A, H, Q, R, m0, P0, n_steps=T)

    samples = d.sample(getkey(), (200_000,)).reshape(200_000, -1)
    mu, Sigma = _dense_joint(A, H, Q, R, m0, P0, T)

    # Monte-Carlo error at 200k samples is ~1e-2 for these magnitudes.
    assert np.abs(np.asarray(samples.mean(0)) - mu).max() <= 2e-2
    assert np.abs(np.asarray(jnp.cov(samples.T)) - Sigma).max() <= 2e-2


@pytest.mark.slow
@pytest.mark.integration
def test_numpyro_svi_reduces_loss(getkey):
    """LGSSM used as an observed site inside a NumPyro model."""
    import numpyro
    import numpyro.distributions as ndist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    T, N, M = 8, 2, 2
    A_true = jnp.array([[0.8, 0.1], [0.0, 0.7]])
    H = jnp.eye(M)
    Q = 0.2 * jnp.eye(N)
    R = 0.3 * jnp.eye(M)
    m0 = jnp.zeros(N)
    P0 = jnp.eye(N)
    y = LGSSM(A_true, H, Q, R, m0, P0, n_steps=T).sample(getkey())

    def model(obs):
        a = numpyro.sample("a", ndist.Normal(0.0, 1.0).expand([N, N]).to_event(2))
        numpyro.sample("y", LGSSM(a, H, Q, R, m0, P0, n_steps=T), obs=obs)

    svi = SVI(model, AutoNormal(model), numpyro.optim.Adam(1e-2), Trace_ELBO())
    result = svi.run(jr.key(0), 200, y, progress_bar=False)

    losses = np.asarray(result.losses)
    assert np.isfinite(losses).all()
    assert losses[-20:].mean() < losses[:20].mean()
