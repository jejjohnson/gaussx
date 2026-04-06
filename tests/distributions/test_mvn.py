"""Tests for MultivariateNormal distribution."""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._distributions import MultivariateNormal
from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._strategies import AutoSolver, DenseSolver
from gaussx._testing import tree_allclose


def _make_psd(key, n):
    """Create a random PSD matrix."""
    A = jr.normal(key, (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


class TestLogProb:
    def test_matches_manual(self, getkey):
        n = 5
        mu = jr.normal(getkey(), (n,))
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        x = jr.normal(getkey(), (n,))

        d = MultivariateNormal(mu, op)
        lp = d.log_prob(x)

        # Manual computation
        residual = x - mu
        alpha = jnp.linalg.solve(Sigma, residual)
        quad = residual @ alpha
        ld = jnp.linalg.slogdet(Sigma)[1]
        lp_expected = -0.5 * (n * jnp.log(2.0 * jnp.pi) + ld + quad)

        assert tree_allclose(lp, lp_expected, rtol=1e-5)

    def test_matches_numpyro(self, getkey):
        import numpyro.distributions as dist

        n = 4
        mu = jr.normal(getkey(), (n,))
        Sigma = _make_psd(getkey(), n)
        x = jr.normal(getkey(), (n,))

        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        lp_ours = MultivariateNormal(mu, op).log_prob(x)
        lp_numpyro = dist.MultivariateNormal(mu, covariance_matrix=Sigma).log_prob(x)

        assert tree_allclose(lp_ours, lp_numpyro, rtol=1e-5)

    def test_batched_loc_matches_numpyro(self, getkey):
        import numpyro.distributions as dist

        n = 4
        batch = 5
        mu = jr.normal(getkey(), (batch, n))
        Sigma = _make_psd(getkey(), n)
        x = jr.normal(getkey(), (batch, n))

        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        lp_ours = MultivariateNormal(mu, op).log_prob(x)
        lp_numpyro = dist.MultivariateNormal(mu, covariance_matrix=Sigma).log_prob(x)

        assert tree_allclose(lp_ours, lp_numpyro, rtol=1e-5)

    def test_diagonal_operator(self, getkey):
        n = 6
        d_vals = jnp.abs(jr.normal(getkey(), (n,))) + 0.5
        op = lx.DiagonalLinearOperator(d_vals)
        mu = jnp.zeros(n)
        x = jr.normal(getkey(), (n,))

        lp = MultivariateNormal(mu, op).log_prob(x)

        # Manual: diagonal covariance
        ld = jnp.sum(jnp.log(d_vals))
        quad = jnp.sum(x**2 / d_vals)
        lp_expected = -0.5 * (n * jnp.log(2.0 * jnp.pi) + ld + quad)

        assert tree_allclose(lp, lp_expected, rtol=1e-5)


class TestSample:
    def test_sample_shape(self, getkey):
        n = 4
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(jnp.zeros(n), op)

        samples = d.sample(getkey(), sample_shape=(100,))
        assert samples.shape == (100, n)

    def test_sample_statistics(self, getkey):
        n = 3
        mu = jnp.array([1.0, -0.5, 2.0])
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(mu, op)

        samples = d.sample(getkey(), sample_shape=(50_000,))
        sample_mean = jnp.mean(samples, axis=0)
        sample_cov = jnp.cov(samples.T)

        assert jnp.allclose(sample_mean, mu, atol=0.1)
        assert jnp.allclose(sample_cov, Sigma, atol=0.3)

    def test_single_sample(self, getkey):
        n = 3
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(jnp.zeros(n), op)

        sample = d.sample(getkey())
        assert sample.shape == (n,)

    def test_batched_loc_sample_shape(self, getkey):
        n = 3
        batch = 4
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        mu = jr.normal(getkey(), (batch, n))
        d = MultivariateNormal(mu, op)

        sample = d.sample(getkey())
        assert sample.shape == (batch, n)

    def test_log_prob_multi_sample_shape_matches_numpyro(self, getkey):
        import numpyro.distributions as dist

        n = 3
        Sigma = _make_psd(getkey(), n)
        mu = jr.normal(getkey(), (n,))
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(mu, op)
        samples = d.sample(getkey(), sample_shape=(2, 3))

        lp_ours = d.log_prob(samples)
        lp_numpyro = dist.MultivariateNormal(mu, covariance_matrix=Sigma).log_prob(
            samples
        )

        assert lp_ours.shape == (2, 3)
        assert tree_allclose(lp_ours, lp_numpyro, rtol=1e-5)


class TestProperties:
    def test_mean(self, getkey):
        n = 4
        mu = jr.normal(getkey(), (n,))
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(mu, op)

        assert tree_allclose(d.mean, mu)

    def test_variance(self, getkey):
        n = 4
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(jnp.zeros(n), op)

        assert tree_allclose(d.variance, jnp.diag(Sigma), rtol=1e-5)

    def test_variance_broadcasts_over_batched_loc(self, getkey):
        n = 4
        batch = 3
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        mu = jr.normal(getkey(), (batch, n))
        d = MultivariateNormal(mu, op)

        expected = jnp.broadcast_to(jnp.diag(Sigma), (batch, n))
        assert tree_allclose(d.variance, expected, rtol=1e-5)

    def test_entropy_matches_manual(self, getkey):
        n = 4
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(jnp.zeros(n), op)

        ld = jnp.linalg.slogdet(Sigma)[1]
        expected = 0.5 * (n * (1.0 + jnp.log(2.0 * jnp.pi)) + ld)

        assert tree_allclose(d.entropy(), expected, rtol=1e-5)

    def test_event_shape(self, getkey):
        n = 5
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(jnp.zeros(n), op)

        assert d.event_shape == (n,)
        assert d.batch_shape == ()


class TestStructuredOperators:
    def test_kronecker(self, getkey):
        A = _make_psd(getkey(), 2)
        B = _make_psd(getkey(), 3)
        A_op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
        B_op = lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag)
        kron = Kronecker(A_op, B_op)

        mu = jnp.zeros(6)
        x = jr.normal(getkey(), (6,))

        d = MultivariateNormal(mu, kron)
        lp = d.log_prob(x)

        # Reference: dense Kronecker product
        Sigma_dense = jnp.kron(A, B)
        residual = x - mu
        alpha = jnp.linalg.solve(Sigma_dense, residual)
        ld = jnp.linalg.slogdet(Sigma_dense)[1]
        lp_expected = -0.5 * (6 * jnp.log(2.0 * jnp.pi) + ld + residual @ alpha)

        assert tree_allclose(lp, lp_expected, rtol=1e-4)

    def test_block_diag(self, getkey):
        A = _make_psd(getkey(), 2)
        B = _make_psd(getkey(), 3)
        A_op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
        B_op = lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag)
        bd = BlockDiag(A_op, B_op)

        mu = jnp.zeros(5)
        x = jr.normal(getkey(), (5,))

        d = MultivariateNormal(mu, bd)
        lp = d.log_prob(x)

        # Reference: dense block-diagonal
        Sigma_dense = bd.as_matrix()
        residual = x - mu
        alpha = jnp.linalg.solve(Sigma_dense, residual)
        ld = jnp.linalg.slogdet(Sigma_dense)[1]
        lp_expected = -0.5 * (5 * jnp.log(2.0 * jnp.pi) + ld + residual @ alpha)

        assert tree_allclose(lp, lp_expected, rtol=1e-4)

    def test_low_rank_update(self, getkey):
        n = 5
        d_vals = jnp.abs(jr.normal(getkey(), (n,))) + 1.0
        base = lx.DiagonalLinearOperator(d_vals)
        U = jr.normal(getkey(), (n, 2)) * 0.1
        lr = LowRankUpdate(base, U)

        mu = jnp.zeros(n)
        x = jr.normal(getkey(), (n,))

        d = MultivariateNormal(mu, lr)
        lp = d.log_prob(x)

        # Reference: dense
        Sigma_dense = lr.as_matrix()
        residual = x - mu
        alpha = jnp.linalg.solve(Sigma_dense, residual)
        ld = jnp.linalg.slogdet(Sigma_dense)[1]
        lp_expected = -0.5 * (n * jnp.log(2.0 * jnp.pi) + ld + residual @ alpha)

        assert tree_allclose(lp, lp_expected, rtol=1e-4)


class TestJIT:
    def test_log_prob_jit(self, getkey):
        n = 4
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(jnp.zeros(n), op)
        x = jr.normal(getkey(), (n,))

        lp_eager = d.log_prob(x)
        lp_jit = jax.jit(d.log_prob)(x)

        assert tree_allclose(lp_eager, lp_jit)

    def test_sample_jit(self, getkey):
        n = 4
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(jnp.zeros(n), op)

        sample = jax.jit(d.sample)(getkey())
        assert sample.shape == (n,)

    def test_grad_log_prob(self, getkey):
        n = 3
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d = MultivariateNormal(jnp.zeros(n), op)
        x = jr.normal(getkey(), (n,))

        grad_fn = jax.grad(d.log_prob)
        g = grad_fn(x)
        assert g.shape == (n,)
        assert jnp.all(jnp.isfinite(g))


class TestVmapVsNumpyro:
    """Verify vmapped gaussx matches numpyro's native batching."""

    def test_vmap_log_prob_batched_means(self, getkey):
        import numpyro.distributions as dist

        n = 4
        batch = 5
        Sigma = _make_psd(getkey(), n)
        mu_batch = jr.normal(getkey(), (batch, n))
        x_batch = jr.normal(getkey(), (batch, n))

        # numpyro: native batch
        lp_np = dist.MultivariateNormal(mu_batch, covariance_matrix=Sigma).log_prob(
            x_batch
        )

        # ours: vmap
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        def single_lp(mu_i, x_i):
            return MultivariateNormal(mu_i, op).log_prob(x_i)

        lp_ours = jax.vmap(single_lp)(mu_batch, x_batch)
        assert tree_allclose(lp_ours, lp_np, rtol=1e-5)

    def test_vmap_log_prob_batched_covariances(self, getkey):
        import numpyro.distributions as dist

        n = 3
        batch = 4
        mu = jr.normal(getkey(), (n,))
        Sigmas = jnp.stack([_make_psd(getkey(), n) for _ in range(batch)])
        x_batch = jr.normal(getkey(), (batch, n))

        # numpyro: native batch
        lp_np = dist.MultivariateNormal(mu, covariance_matrix=Sigmas).log_prob(x_batch)

        # ours: vmap over covariance matrices
        def single_lp(Sigma_i, x_i):
            op = lx.MatrixLinearOperator(Sigma_i, lx.positive_semidefinite_tag)
            return MultivariateNormal(mu, op).log_prob(x_i)

        lp_ours = jax.vmap(single_lp)(Sigmas, x_batch)
        assert tree_allclose(lp_ours, lp_np, rtol=1e-5)

    def test_vmap_entropy(self, getkey):
        import numpyro.distributions as dist

        n = 3
        batch = 4
        mu = jnp.zeros(n)
        Sigmas = jnp.stack([_make_psd(getkey(), n) for _ in range(batch)])

        # numpyro: native batch
        h_np = dist.MultivariateNormal(mu, covariance_matrix=Sigmas).entropy()

        # ours: vmap
        def single_h(Sigma_i):
            op = lx.MatrixLinearOperator(Sigma_i, lx.positive_semidefinite_tag)
            return MultivariateNormal(mu, op).entropy()

        h_ours = jax.vmap(single_h)(Sigmas)
        assert tree_allclose(h_ours, h_np, rtol=1e-5)

    def test_vmap_grad_log_prob(self, getkey):
        import numpyro.distributions as dist

        n = 3
        batch = 4
        Sigma = _make_psd(getkey(), n)
        mu_batch = jr.normal(getkey(), (batch, n))
        x_batch = jr.normal(getkey(), (batch, n))

        # numpyro gradient
        def neg_lp_np(mu_batch):
            d = dist.MultivariateNormal(mu_batch, covariance_matrix=Sigma)
            return -jnp.sum(d.log_prob(x_batch))

        g_np = jax.grad(neg_lp_np)(mu_batch)

        # ours via vmap
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        def neg_lp_ours(mu_batch):
            def single_lp(mu_i, x_i):
                return MultivariateNormal(mu_i, op).log_prob(x_i)

            return -jnp.sum(jax.vmap(single_lp)(mu_batch, x_batch))

        g_ours = jax.grad(neg_lp_ours)(mu_batch)
        assert tree_allclose(g_ours, g_np, rtol=1e-5)

    def test_vmap_sample_shape(self, getkey):
        n = 4
        batch = 3
        Sigma = _make_psd(getkey(), n)
        mu_batch = jr.normal(getkey(), (batch, n))
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        keys = jr.split(getkey(), batch)

        def single_sample(mu_i, key_i):
            return MultivariateNormal(mu_i, op).sample(key_i)

        samples = jax.vmap(single_sample)(mu_batch, keys)
        assert samples.shape == (batch, n)


class TestSolverChoice:
    def test_explicit_solver(self, getkey):
        n = 4
        Sigma = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        x = jr.normal(getkey(), (n,))

        d_auto = MultivariateNormal(jnp.zeros(n), op, solver=AutoSolver())
        d_dense = MultivariateNormal(jnp.zeros(n), op, solver=DenseSolver())

        assert tree_allclose(d_auto.log_prob(x), d_dense.log_prob(x), rtol=1e-5)
