"""Tests for MultivariateNormalPrecision distribution."""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._distributions import MultivariateNormal, MultivariateNormalPrecision
from gaussx._testing import assert_sample_moments, tree_allclose


def _make_psd(key, n):
    """Create a random PSD matrix."""
    A = jr.normal(key, (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


class TestLogProb:
    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_sample_statistics(self, getkey):
        n = 3
        mu = jnp.array([1.0, -0.5, 2.0])
        Sigma = _make_psd(getkey(), n)
        Lambda = jnp.linalg.inv(Sigma)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        d = MultivariateNormalPrecision(mu, op)

        samples = d.sample(getkey(), sample_shape=(10_000,))

        # Bounded by the estimators' own sampling distributions, not a
        # fixed atol: ``Sigma`` is drawn per-seed, so an absolute
        # tolerance is a different number of sigmas on every run (gh-220).
        assert_sample_moments(samples, mu, Sigma)

    def test_batched_loc_sample_shape(self, getkey):
        n = 3
        batch = 4
        Lambda = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
        mu = jr.normal(getkey(), (batch, n))
        d = MultivariateNormalPrecision(mu, op)

        sample = d.sample(getkey())
        assert sample.shape == (batch, n)

    @pytest.mark.slow
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

    @pytest.mark.slow
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


# ---------------------------------------------------------------------------
# gh-298: structured precisions keep their Cholesky; non-finite falls back
# ---------------------------------------------------------------------------


def _psd(matrix):
    return lx.MatrixLinearOperator(jnp.asarray(matrix), lx.positive_semidefinite_tag)


def _scaled_kronecker_precision():
    from gaussx import Kronecker

    A = _psd([[2.0, 0.5], [0.5, 1.0]])
    B = _psd([[1.0, 0.2, 0.0], [0.2, 1.5, 0.3], [0.0, 0.3, 1.2]])
    return 2.0 * Kronecker(A, B)


def test_scaled_kronecker_precision_keeps_its_structure(monkeypatch):
    """2 * Kronecker used to densify: the Cholesky now sees the Kronecker."""
    import gaussx._distributions._mvn_prec as prec_module

    factored = []
    original = prec_module._cholesky

    def spy(operator):
        factored.append(type(operator).__name__)
        return original(operator)

    monkeypatch.setattr(prec_module, "_cholesky", spy)
    precision = _scaled_kronecker_precision()
    d = MultivariateNormalPrecision(jnp.zeros(6), precision)
    draws = d.sample(jr.key(0), (20_000,))
    assert factored == ["Kronecker"]
    cov = jnp.linalg.inv(precision.as_matrix())
    assert_sample_moments(draws, jnp.zeros(6), cov)


def test_non_finite_cholesky_falls_back_to_dense(monkeypatch):
    """A factor that fails numerically gives draws from the dense route."""
    import gaussx._distributions._mvn_prec as prec_module

    def nan_factor(operator):
        n = operator.in_size()
        return lx.MatrixLinearOperator(
            jnp.full((n, n), jnp.nan), lx.lower_triangular_tag
        )

    monkeypatch.setattr(prec_module, "_cholesky", nan_factor)
    precision = _psd([[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.5]])
    d = MultivariateNormalPrecision(jnp.zeros(3), precision)
    draws = d.sample(jr.key(0), (20_000,))
    assert jnp.all(jnp.isfinite(draws))
    assert_sample_moments(draws, jnp.zeros(3), jnp.linalg.inv(precision.as_matrix()))


def test_precision_sample_under_jit():
    d = MultivariateNormalPrecision(jnp.zeros(6), _scaled_kronecker_precision())
    draws = jax.jit(lambda key: d.sample(key, (3,)))(jr.key(0))
    assert draws.shape == (3, 6)
    assert jnp.all(jnp.isfinite(draws))


def _sparse_precision():
    import numpy as np

    from gaussx import SparseOperator

    rows = np.array([0, 1, 2, 0, 1, 1, 2])
    cols = np.array([0, 1, 2, 1, 0, 2, 1])
    values = jnp.array([2.0, 2.0, 2.0, -0.5, -0.5, -0.5, -0.5])
    return SparseOperator.from_coo(
        rows, cols, values, (3, 3), tags=frozenset({lx.positive_semidefinite_tag})
    )


def test_scaled_composite_with_a_sparse_block_still_samples():
    """gh-298 review: 2 * BlockDiag(sparse, dense) must not be unwrapped into
    BlockDiag's Cholesky, which cannot hold a sparse factor."""
    from gaussx import BlockDiag

    dense = lx.MatrixLinearOperator(2 * jnp.eye(2), lx.positive_semidefinite_tag)
    precision = 2.0 * BlockDiag(_sparse_precision(), dense)
    draws = MultivariateNormalPrecision(jnp.zeros(5), precision).sample(jr.key(0), (4,))
    assert draws.shape == (4, 5)
    assert jnp.all(jnp.isfinite(draws))


def test_sparse_precision_sample_stages_no_dense_fallback():
    """gh-298 review: lax.cond stages both branches under jit, so the dense
    fallback is only added for dense precisions."""
    d = MultivariateNormalPrecision(jnp.zeros(3), _sparse_precision())
    jaxpr = str(jax.make_jaxpr(lambda key: d.sample(key, (2,)))(jr.key(0)))
    assert "eigh" not in jaxpr
    assert jnp.all(jnp.isfinite(d.sample(jr.key(0), (2,))))
