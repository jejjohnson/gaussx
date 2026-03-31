"""Tests for conditional Gaussian distribution."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import gaussx


def _make_psd_mat(key, n):
    """Create a random PSD matrix."""
    M = jax.random.normal(key, (n, n))
    return M @ M.T + jnp.eye(n)


class TestConditional:
    def test_conditional_mean(self):
        """Conditional mean matches closed-form for a known case."""
        # Joint: N([0,0], [[1, 0.5], [0.5, 1]])
        mu = jnp.zeros(2)
        Sigma = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        # Observe x_1 = 1.0
        cond_mean, cond_cov = gaussx.conditional(
            mu, cov, jnp.array([1]), jnp.array([1.0])
        )
        # mu_0|1 = 0 + 0.5 * 1.0 / 1.0 * (1.0 - 0) = 0.5
        assert jnp.allclose(cond_mean, jnp.array([0.5]), atol=1e-5)
        # Sigma_0|1 = 1 - 0.5^2 / 1 = 0.75
        assert jnp.allclose(cond_cov.as_matrix(), jnp.array([[0.75]]), atol=1e-5)

    def test_conditional_recovers_marginal(self):
        """Conditioning on nothing should return something close to prior."""
        n = 5
        key = jax.random.PRNGKey(0)
        mu = jax.random.normal(key, (n,))
        Sigma = _make_psd_mat(jax.random.PRNGKey(1), n)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        # Observe 2 variables
        obs_idx = jnp.array([1, 3])
        obs_values = mu[obs_idx]  # Observe at the mean

        cond_mean, _cond_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)
        # Conditional mean should equal prior mean at free indices
        free_idx = jnp.array([0, 2, 4])
        assert jnp.allclose(cond_mean, mu[free_idx], atol=1e-4)

    def test_conditional_cov_psd(self):
        """Conditional covariance should be PSD."""
        n = 6
        key = jax.random.PRNGKey(42)
        mu = jnp.zeros(n)
        Sigma = _make_psd_mat(key, n)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        obs_idx = jnp.array([0, 2, 4])
        obs_values = jnp.array([1.0, -0.5, 0.3])
        _, cond_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)

        eigs = jnp.linalg.eigvalsh(cond_cov.as_matrix())
        assert jnp.all(eigs > -1e-6)

    def test_conditional_reduces_variance(self):
        """Conditional variance <= marginal variance."""
        n = 4
        key = jax.random.PRNGKey(7)
        mu = jnp.zeros(n)
        Sigma = _make_psd_mat(key, n)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        obs_idx = jnp.array([1])
        obs_values = jnp.array([0.5])
        _, cond_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)

        free_idx = jnp.array([0, 2, 3])
        marginal_var = jnp.diag(Sigma)[free_idx]
        cond_var = jnp.diag(cond_cov.as_matrix())
        assert jnp.all(cond_var <= marginal_var + 1e-6)

    def test_conditional_against_dense(self):
        """Test against manual computation."""
        n = 5
        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        mu = jax.random.normal(k1, (n,))
        Sigma = _make_psd_mat(k2, n)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        obs_idx = jnp.array([0, 3])
        free_idx = jnp.array([1, 2, 4])
        obs_values = jnp.array([1.0, -1.0])

        cond_mean, cond_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)

        # Manual computation
        Sigma_AA = Sigma[jnp.ix_(free_idx, free_idx)]
        Sigma_AB = Sigma[jnp.ix_(free_idx, obs_idx)]
        Sigma_BB = Sigma[jnp.ix_(obs_idx, obs_idx)]
        mu_A = mu[free_idx]
        mu_B = mu[obs_idx]

        expected_mean = mu_A + Sigma_AB @ jnp.linalg.solve(Sigma_BB, obs_values - mu_B)
        expected_cov = Sigma_AA - Sigma_AB @ jnp.linalg.solve(Sigma_BB, Sigma_AB.T)

        assert jnp.allclose(cond_mean, expected_mean, atol=1e-4)
        assert jnp.allclose(cond_cov.as_matrix(), expected_cov, atol=1e-4)

    def test_duplicate_obs_idx_raises(self):
        """Duplicate observed indices should raise a clear validation error."""
        mu = jnp.zeros(4)
        cov = lx.MatrixLinearOperator(jnp.eye(4), lx.positive_semidefinite_tag)

        with pytest.raises(ValueError, match="duplicates"):
            gaussx.conditional(
                mu,
                cov,
                jnp.array([1, 1, 2]),
                jnp.array([0.0, 0.0, 0.0]),
            )

    def test_obs_idx_out_of_bounds_raises(self):
        """Observed indices must lie within the state dimension."""
        mu = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3), lx.positive_semidefinite_tag)

        with pytest.raises(ValueError, match="bounds"):
            gaussx.conditional(
                mu,
                cov,
                jnp.array([0, 3]),
                jnp.array([0.0, 1.0]),
            )

    def test_obs_values_shape_mismatch_raises(self):
        """Observed values must align one-to-one with observed indices."""
        mu = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3), lx.positive_semidefinite_tag)

        with pytest.raises(ValueError, match="same shape"):
            gaussx.conditional(
                mu,
                cov,
                jnp.array([0, 2]),
                jnp.array([1.0]),
            )
