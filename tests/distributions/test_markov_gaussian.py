"""Tests for the MarkovGaussian chain distribution."""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpyro
import numpyro.handlers as handlers

from gaussx import MarkovGaussian
from gaussx._einx import rearrange
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._ssm._kalman import kalman_filter, rts_smoother
from gaussx._ssm._spingp import spingp_posterior
from gaussx._ssm._udl import udl_decomposition
from gaussx._testing import assert_sample_moments


def _make_chain(key, T, d, *, with_offset=True):
    """Random stable chain with a non-trivial offset."""
    k1, k2, k3, k4, k5 = jr.split(key, 5)
    A = 0.8 * jnp.eye(d)[None] + 0.05 * jr.normal(k1, (T - 1, d, d))
    raw = jr.normal(k2, (T - 1, d, d))
    Q = jax.vmap(lambda M: M @ M.T)(raw) + 0.1 * jnp.eye(d)[None]
    mu0 = jr.normal(k3, (d,))
    raw0 = jr.normal(k4, (d, d))
    P0 = raw0 @ raw0.T + 0.1 * jnp.eye(d)
    b = jr.normal(k5, (T - 1, d)) if with_offset else None
    return MarkovGaussian(A, Q, mu0, P0, b=b)


def _dense_moments(chain):
    """Reference joint mean / covariance by explicit propagation."""
    T, d = chain.horizon, chain.state_dim
    # x = G eps + m with eps = (x0 - mu0, eps_0, ..., eps_{T-2}); G is the
    # (Td, Td) linear map from the stacked noise to the trajectory.
    n = T * d
    G = jnp.zeros((n, n))
    G = G.at[:d, :d].set(jnp.eye(d))
    means = [chain.mu0]
    for k in range(T - 1):
        r = (k + 1) * d
        G = G.at[r : r + d, :].set(chain.A[k] @ G[k * d : r, :])
        G = G.at[r : r + d, r : r + d].set(jnp.eye(d))
        means.append(chain.A[k] @ means[-1] + chain.b[k])
    S = jax.scipy.linalg.block_diag(chain.P0, *chain.Q)
    return jnp.concatenate(means), G @ S @ G.T


class TestConstruction:
    def test_shapes_and_defaults(self):
        chain = _make_chain(jr.key(0), 6, 2, with_offset=False)
        assert chain.event_shape == (6, 2)
        assert chain.batch_shape == ()
        assert chain.horizon == 6
        assert chain.state_dim == 2
        assert chain.b.shape == (5, 2)
        assert jnp.all(chain.b == 0.0)

    def test_rejects_inconsistent_shapes(self):
        d = 2
        A = jnp.zeros((4, d, d))
        Q = jnp.zeros((4, d, d))
        with pytest.raises(ValueError, match="Q must have shape"):
            MarkovGaussian(A, Q[:3], jnp.zeros(d), jnp.eye(d))
        with pytest.raises(ValueError, match="mu0 must have shape"):
            MarkovGaussian(A, Q, jnp.zeros(d + 1), jnp.eye(d))
        with pytest.raises(ValueError, match="P0 must have shape"):
            MarkovGaussian(A, Q, jnp.zeros(d), jnp.eye(d + 1))
        with pytest.raises(ValueError, match="b must have shape"):
            MarkovGaussian(A, Q, jnp.zeros(d), jnp.eye(d), b=jnp.zeros((3, d)))

    def test_is_a_pytree(self):
        chain = _make_chain(jr.key(1), 4, 2)
        leaves = jax.tree_util.tree_leaves(chain)
        assert len(leaves) == 5
        doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, chain)
        assert isinstance(doubled, MarkovGaussian)
        assert jnp.allclose(doubled.A, 2.0 * chain.A)


class TestMoments:
    def test_marginals_match_dense(self):
        chain = _make_chain(jr.key(1), 6, 2)
        mean_ref, cov_ref = _dense_moments(chain)
        means, covs = chain.marginals()
        T, d = chain.horizon, chain.state_dim
        assert jnp.allclose(rearrange(means, "T d -> (T d)"), mean_ref, atol=1e-10)
        assert jnp.allclose(chain.mean, means)
        for k in range(T):
            r = k * d
            assert jnp.allclose(covs[k], cov_ref[r : r + d, r : r + d], atol=1e-10)
        assert jnp.allclose(chain.variance, jnp.diagonal(covs, axis1=-2, axis2=-1))

    def test_covariance_matrix_matches_dense(self):
        chain = _make_chain(jr.key(2), 5, 3)
        _, cov_ref = _dense_moments(chain)
        assert jnp.allclose(chain.covariance_matrix, cov_ref, atol=1e-8)

    def test_pairwise_marginals_match_dense(self):
        chain = _make_chain(jr.key(3), 5, 2)
        mean_ref, cov_ref = _dense_moments(chain)
        joint_means, joint_covs = chain.pairwise_marginals()
        d = chain.state_dim
        assert joint_means.shape == (4, 2 * d)
        assert joint_covs.shape == (4, 2 * d, 2 * d)
        for k in range(4):
            r = k * d
            assert jnp.allclose(joint_means[k], mean_ref[r : r + 2 * d], atol=1e-10)
            assert jnp.allclose(
                joint_covs[k], cov_ref[r : r + 2 * d, r : r + 2 * d], atol=1e-8
            )

    def test_cross_covariances(self):
        chain = _make_chain(jr.key(4), 5, 2)
        _, cov_ref = _dense_moments(chain)
        cross = chain.cross_covariances()
        d = chain.state_dim
        for k in range(4):
            r, c = (k + 1) * d, k * d
            assert jnp.allclose(cross[k], cov_ref[r : r + d, c : c + d], atol=1e-8)


class TestPrecisionForm:
    def test_precision_is_inverse_of_cov(self):
        chain = _make_chain(jr.key(5), 6, 2)
        prec = chain.precision
        assert isinstance(prec, BlockTriDiag)
        _, cov_ref = _dense_moments(chain)
        assert jnp.allclose(prec.as_matrix() @ cov_ref, jnp.eye(12), atol=1e-8)

    def test_roundtrip_is_identity(self):
        """SSM -> precision -> SSM recovers (A, b, Q, mu0, P0) to 1e-8."""
        chain = _make_chain(jr.key(6), 7, 3)
        back = MarkovGaussian.from_precision_form(*chain.to_precision_form())
        assert jnp.allclose(back.A, chain.A, atol=1e-8)
        assert jnp.allclose(back.b, chain.b, atol=1e-8)
        assert jnp.allclose(back.Q, chain.Q, atol=1e-8)
        assert jnp.allclose(back.mu0, chain.mu0, atol=1e-8)
        assert jnp.allclose(back.P0, chain.P0, atol=1e-8)

    def test_from_precision_form_accepts_stacked_mean(self):
        chain = _make_chain(jr.key(7), 4, 2)
        mean, prec = chain.to_precision_form()
        back = MarkovGaussian.from_precision_form(
            rearrange(mean, "(T d) -> T d", T=4, d=2), prec
        )
        assert jnp.allclose(back.A, chain.A, atol=1e-8)
        assert jnp.allclose(back.mean, chain.mean, atol=1e-8)

    def test_spingp_posterior_matches_rts_smoother(self):
        """Precision-form posterior -> chain reproduces the RTS marginals.

        The Kalman filter's ``T`` states are x_1..x_T (it predicts before
        the first update), so the matching prior chain starts at
        x_1 ~ N(A m0, A P0 A^T + Q). ``spingp_posterior`` assumes a
        zero-mean prior; the prior-mean term is added through the UDL
        solve.
        """
        T, d, m = 8, 2, 1
        A = jnp.array([[0.9, 0.1], [0.0, 0.8]])
        Q = jnp.array([[0.2, 0.05], [0.05, 0.3]])
        H = jnp.array([[1.0, 0.0]])
        R = 0.1 * jnp.eye(m)
        m0 = jnp.array([0.5, -0.5])
        P0 = jnp.eye(d)
        y = jr.normal(jr.key(8), (T, m))

        prior = MarkovGaussian(
            jnp.broadcast_to(A, (T - 1, d, d)),
            jnp.broadcast_to(Q, (T - 1, d, d)),
            A @ m0,
            A @ P0 @ A.T + Q,
        )
        prior_mean, prior_prec = prior.to_precision_form()
        zero_mean_post, post_prec = spingp_posterior(
            prior_prec, H, lx.MatrixLinearOperator(R), y
        )
        eta = post_prec.mv(zero_mean_post) + prior_prec.mv(prior_mean)
        post_mean = udl_decomposition(post_prec).solve(eta)
        posterior = MarkovGaussian.from_precision_form(post_mean, post_prec)
        means, covs = posterior.marginals()

        state = kalman_filter(A, H, Q, R, y, m0, P0)
        s_means, s_covs = rts_smoother(state, A, Q)
        assert jnp.allclose(means, s_means, atol=1e-8)
        assert jnp.allclose(covs, s_covs, atol=1e-8)

        # And the extracted chain samples the smoothing posterior.
        samples = posterior.sample(jr.key(9), (8192,))
        # Band from the estimator's own sampling distribution (7 sigma).
        assert_sample_moments(
            rearrange(samples, "S T d -> S (T d)"),
            post_mean,
            jnp.linalg.inv(post_prec.as_matrix()),
        )


class TestDensityAndSampling:
    def test_log_prob_matches_dense_mvn(self):
        chain = _make_chain(jr.key(10), 6, 2)
        mean_ref, cov_ref = _dense_moments(chain)
        xs = jr.normal(jr.key(11), (6, 2))
        expected = jax.scipy.stats.multivariate_normal.logpdf(
            rearrange(xs, "T d -> (T d)"), mean_ref, cov_ref
        )
        assert chain.log_prob(xs).shape == ()
        assert jnp.allclose(chain.log_prob(xs), expected, atol=1e-8)

    def test_log_prob_batches(self):
        chain = _make_chain(jr.key(12), 5, 2)
        xs = chain.sample(jr.key(13), (3, 4))
        assert xs.shape == (3, 4, 5, 2)
        lp = chain.log_prob(xs)
        assert lp.shape == (3, 4)
        assert jnp.allclose(lp[1, 2], chain.log_prob(xs[1, 2]))

    def test_sample_moments(self):
        chain = _make_chain(jr.key(14), 5, 2)
        samples = chain.sample(jr.key(15), (4096,))
        assert samples.shape == (4096, 5, 2)
        assert chain.sample(jr.key(16)).shape == (5, 2)
        mean_ref, cov_ref = _dense_moments(chain)
        # Band from the estimator's own sampling distribution (7 sigma).
        assert_sample_moments(rearrange(samples, "S T d -> S (T d)"), mean_ref, cov_ref)

    def test_sample_requires_key(self):
        chain = _make_chain(jr.key(17), 4, 2)
        with pytest.raises(ValueError, match="PRNG key"):
            chain.sample(None)

    def test_usable_as_numpyro_site(self):
        chain = _make_chain(jr.key(18), 6, 2)

        def model():
            numpyro.sample("x", chain)

        trace = handlers.trace(handlers.seed(model, jr.key(19))).get_trace()
        site = trace["x"]
        assert site["value"].shape == (6, 2)
        assert site["fn"].log_prob(site["value"]).shape == ()

    @pytest.mark.slow
    def test_jit_grad_through_precision_form(self):
        chain = _make_chain(jr.key(20), 6, 2)
        xs = chain.sample(jr.key(21))

        def loss(c):
            mean, prec = c.to_precision_form()
            back = MarkovGaussian.from_precision_form(mean, prec)
            return back.log_prob(xs)

        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(chain)
        assert jnp.allclose(value, chain.log_prob(xs), atol=1e-8)
        assert jnp.all(jnp.isfinite(grads.A))
        assert jnp.all(jnp.isfinite(grads.Q))
