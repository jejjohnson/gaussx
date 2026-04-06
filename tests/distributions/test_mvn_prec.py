"""Tests for MultivariateNormalPrecision distribution."""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._distributions import MultivariateNormal, MultivariateNormalPrecision
from gaussx._testing import tree_allclose


def _make_psd(key, n):
    """Create a random PSD matrix."""
    A = jr.normal(key, (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


class TestLogProb:
    def test_matches_manual(self, getkey):
        n = 5
        mu = jr.normal(getkey(), (n,))
        Lambda = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        x = jr.normal(getkey(), (n,))

        d = MultivariateNormalPrecision(mu, op)
        lp = d.log_prob(x)

        # Manual: -0.5 * (N*log(2pi) - log|Lambda| + (x-mu)^T Lambda (x-mu))
        residual = x - mu
        quad = residual @ Lambda @ residual
        ld = jnp.linalg.slogdet(Lambda)[1]
        lp_expected = -0.5 * (n * jnp.log(2.0 * jnp.pi) - ld + quad)

        assert tree_allclose(lp, lp_expected, rtol=1e-5)

    def test_matches_covariance_form(self, getkey):
        """Precision-parameterized log_prob should match covariance form."""
        n = 4
        mu = jr.normal(getkey(), (n,))
        Sigma = _make_psd(getkey(), n)
        Lambda = jnp.linalg.inv(Sigma)
        x = jr.normal(getkey(), (n,))

        cov_op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        prec_op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)

        lp_cov = MultivariateNormal(mu, cov_op).log_prob(x)
        lp_prec = MultivariateNormalPrecision(mu, prec_op).log_prob(x)

        assert tree_allclose(lp_cov, lp_prec, rtol=1e-4)

    def test_batched_loc_matches_numpyro(self, getkey):
        import numpyro.distributions as dist

        n = 4
        batch = 5
        mu = jr.normal(getkey(), (batch, n))
        Lambda = _make_psd(getkey(), n)
        x = jr.normal(getkey(), (batch, n))

        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        lp_ours = MultivariateNormalPrecision(mu, op).log_prob(x)
        lp_numpyro = dist.MultivariateNormal(mu, precision_matrix=Lambda).log_prob(x)

        assert tree_allclose(lp_ours, lp_numpyro, rtol=1e-5)


class TestSample:
    def test_sample_shape(self, getkey):
        n = 4
        Lambda = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(jnp.zeros(n), op)

        samples = d.sample(getkey(), sample_shape=(100,))
        assert samples.shape == (100, n)

    def test_sample_statistics(self, getkey):
        n = 3
        mu = jnp.array([1.0, -0.5, 2.0])
        Sigma = _make_psd(getkey(), n)
        Lambda = jnp.linalg.inv(Sigma)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(mu, op)

        samples = d.sample(getkey(), sample_shape=(50_000,))
        sample_mean = jnp.mean(samples, axis=0)
        sample_cov = jnp.cov(samples.T)

        assert jnp.allclose(sample_mean, mu, atol=0.1)
        assert jnp.allclose(sample_cov, Sigma, atol=0.3)

    def test_batched_loc_sample_shape(self, getkey):
        n = 3
        batch = 4
        Lambda = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        mu = jr.normal(getkey(), (batch, n))
        d = MultivariateNormalPrecision(mu, op)

        sample = d.sample(getkey())
        assert sample.shape == (batch, n)

    def test_log_prob_multi_sample_shape_matches_numpyro(self, getkey):
        import numpyro.distributions as dist

        n = 3
        Lambda = _make_psd(getkey(), n)
        mu = jr.normal(getkey(), (n,))
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(mu, op)
        samples = d.sample(getkey(), sample_shape=(2, 3))

        lp_ours = d.log_prob(samples)
        lp_numpyro = dist.MultivariateNormal(mu, precision_matrix=Lambda).log_prob(
            samples
        )

        assert lp_ours.shape == (2, 3)
        assert tree_allclose(lp_ours, lp_numpyro, rtol=1e-5)


class TestProperties:
    def test_mean(self, getkey):
        n = 4
        mu = jr.normal(getkey(), (n,))
        Lambda = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(mu, op)

        assert tree_allclose(d.mean, mu)

    def test_variance(self, getkey):
        n = 4
        Sigma = _make_psd(getkey(), n)
        Lambda = jnp.linalg.inv(Sigma)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(jnp.zeros(n), op)

        assert tree_allclose(d.variance, jnp.diag(Sigma), rtol=1e-4)

    def test_variance_broadcasts_over_batched_loc(self, getkey):
        n = 4
        batch = 3
        Sigma = _make_psd(getkey(), n)
        Lambda = jnp.linalg.inv(Sigma)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        mu = jr.normal(getkey(), (batch, n))
        d = MultivariateNormalPrecision(mu, op)

        expected = jnp.broadcast_to(jnp.diag(Sigma), (batch, n))
        assert tree_allclose(d.variance, expected, rtol=1e-4)

    def test_entropy_matches_covariance_form(self, getkey):
        n = 4
        Sigma = _make_psd(getkey(), n)
        Lambda = jnp.linalg.inv(Sigma)

        cov_op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        prec_op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)

        h_cov = MultivariateNormal(jnp.zeros(n), cov_op).entropy()
        h_prec = MultivariateNormalPrecision(jnp.zeros(n), prec_op).entropy()

        assert tree_allclose(h_cov, h_prec, rtol=1e-4)

    def test_event_shape(self, getkey):
        n = 5
        Lambda = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(jnp.zeros(n), op)

        assert d.event_shape == (n,)
        assert d.batch_shape == ()


class TestVmapVsNumpyro:
    """Verify vmapped precision form matches numpyro's native batching."""

    def test_vmap_log_prob_batched_precision(self, getkey):
        import numpyro.distributions as dist

        n = 3
        batch = 4
        mu = jr.normal(getkey(), (n,))
        Lambdas = jnp.stack([_make_psd(getkey(), n) for _ in range(batch)])
        x_batch = jr.normal(getkey(), (batch, n))

        # numpyro: native batch
        lp_np = dist.MultivariateNormal(mu, precision_matrix=Lambdas).log_prob(x_batch)

        # ours: vmap
        def single_lp(Lambda_i, x_i):
            op = lx.MatrixLinearOperator(Lambda_i, lx.positive_semidefinite_tag)
            return MultivariateNormalPrecision(mu, op).log_prob(x_i)

        lp_ours = jax.vmap(single_lp)(Lambdas, x_batch)
        assert tree_allclose(lp_ours, lp_np, rtol=1e-5)

    def test_vmap_log_prob_matches_covariance_form(self, getkey):
        n = 3
        batch = 4
        mu = jr.normal(getkey(), (n,))
        Sigmas = jnp.stack([_make_psd(getkey(), n) for _ in range(batch)])
        Lambdas = jnp.linalg.inv(Sigmas)
        x_batch = jr.normal(getkey(), (batch, n))

        def cov_lp(Sigma_i, x_i):
            op = lx.MatrixLinearOperator(Sigma_i, lx.positive_semidefinite_tag)
            return MultivariateNormal(mu, op).log_prob(x_i)

        def prec_lp(Lambda_i, x_i):
            op = lx.MatrixLinearOperator(Lambda_i, lx.positive_semidefinite_tag)
            return MultivariateNormalPrecision(mu, op).log_prob(x_i)

        lp_cov = jax.vmap(cov_lp)(Sigmas, x_batch)
        lp_prec = jax.vmap(prec_lp)(Lambdas, x_batch)
        assert tree_allclose(lp_cov, lp_prec, rtol=1e-4)

    def test_vmap_grad_log_prob(self, getkey):
        import numpyro.distributions as dist

        n = 3
        batch = 4
        Lambda = _make_psd(getkey(), n)
        mu_batch = jr.normal(getkey(), (batch, n))
        x_batch = jr.normal(getkey(), (batch, n))

        # numpyro gradient
        def neg_lp_np(mu_batch):
            d = dist.MultivariateNormal(mu_batch, precision_matrix=Lambda)
            return -jnp.sum(d.log_prob(x_batch))

        g_np = jax.grad(neg_lp_np)(mu_batch)

        # ours via vmap
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)

        def neg_lp_ours(mu_batch):
            def single_lp(mu_i, x_i):
                return MultivariateNormalPrecision(mu_i, op).log_prob(x_i)

            return -jnp.sum(jax.vmap(single_lp)(mu_batch, x_batch))

        g_ours = jax.grad(neg_lp_ours)(mu_batch)
        assert tree_allclose(g_ours, g_np, rtol=1e-5)


class TestJIT:
    def test_log_prob_jit(self, getkey):
        n = 4
        Lambda = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(jnp.zeros(n), op)
        x = jr.normal(getkey(), (n,))

        lp_eager = d.log_prob(x)
        lp_jit = jax.jit(d.log_prob)(x)

        assert tree_allclose(lp_eager, lp_jit)

    def test_grad_log_prob(self, getkey):
        n = 3
        Lambda = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(jnp.zeros(n), op)
        x = jr.normal(getkey(), (n,))

        grad_fn = jax.grad(d.log_prob)
        g = grad_fn(x)
        assert g.shape == (n,)
        assert jnp.all(jnp.isfinite(g))
