"""Tests for distribution-level KL divergence."""

import jax
import jax.numpy as jnp
import lineax as lx

import gaussx


def _make_psd_op(key, n):
    """Create a random PSD operator."""
    M = jax.random.normal(key, (n, n))
    mat = M @ M.T + jnp.eye(n)
    return lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)


class TestDistKLDivergence:
    def test_kl_same_distribution(self):
        """KL(p || p) = 0."""
        n = 4
        key = jax.random.PRNGKey(0)
        mu = jax.random.normal(key, (n,))
        cov = _make_psd_op(jax.random.PRNGKey(1), n)
        kl = gaussx.dist_kl_divergence(mu, cov, mu, cov)
        assert jnp.allclose(kl, 0.0, atol=1e-4)

    def test_kl_positive(self):
        """KL is non-negative."""
        n = 5
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
        mu_p = jax.random.normal(k1, (n,))
        mu_q = jax.random.normal(k2, (n,))
        cov_p = _make_psd_op(k3, n)
        cov_q = _make_psd_op(k4, n)
        kl = gaussx.dist_kl_divergence(mu_p, cov_p, mu_q, cov_q)
        assert kl >= -1e-6

    def test_kl_known_value(self):
        """KL between two known Gaussians matches closed-form."""
        n = 3
        mu_p = jnp.zeros(n)
        mu_q = jnp.ones(n)
        cov_p = lx.MatrixLinearOperator(jnp.eye(n), lx.positive_semidefinite_tag)
        cov_q = lx.MatrixLinearOperator(2.0 * jnp.eye(n), lx.positive_semidefinite_tag)
        kl = gaussx.dist_kl_divergence(mu_p, cov_p, mu_q, cov_q)
        # KL(N(0,I) || N(1, 2I)) = 0.5 * (tr(0.5 I) + 0.5*||1||^2 - 3 + 3*ln2)
        # = 0.5 * (1.5 + 1.5 - 3 + 3*ln2) = 0.5 * 3 * ln2 = 1.5 * ln2 ≈ 1.0397
        expected = 0.5 * (n / 2.0 + 0.5 * n - n + n * jnp.log(2.0))
        assert jnp.allclose(kl, expected, atol=1e-4)

    def test_kl_diagonal_operators(self):
        """KL with diagonal covariance operators."""
        n = 4
        mu_p = jnp.zeros(n)
        mu_q = jnp.zeros(n)
        diag_p = jnp.array([1.0, 2.0, 3.0, 4.0])
        diag_q = jnp.array([2.0, 3.0, 4.0, 5.0])
        cov_p = lx.DiagonalLinearOperator(diag_p)
        cov_q = lx.DiagonalLinearOperator(diag_q)
        kl = gaussx.dist_kl_divergence(mu_p, cov_p, mu_q, cov_q)
        # Closed-form for diagonal
        expected = 0.5 * (
            jnp.sum(diag_p / diag_q) - n + jnp.sum(jnp.log(diag_q / diag_p))
        )
        assert jnp.allclose(kl, expected, atol=1e-5)

    def test_kl_asymmetric(self):
        """KL(p||q) != KL(q||p) in general."""
        n = 3
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
        mu_p = jax.random.normal(k1, (n,))
        mu_q = jax.random.normal(k2, (n,))
        cov_p = _make_psd_op(k3, n)
        cov_q = _make_psd_op(k4, n)
        kl_pq = gaussx.dist_kl_divergence(mu_p, cov_p, mu_q, cov_q)
        kl_qp = gaussx.dist_kl_divergence(mu_q, cov_q, mu_p, cov_p)
        assert not jnp.allclose(kl_pq, kl_qp, atol=1e-3)
