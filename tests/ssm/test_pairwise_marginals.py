"""Tests for pairwise marginals."""

import jax
import jax.numpy as jnp

from gaussx import pairwise_marginals


class TestPairwiseMarginals:
    def test_output_shapes(self, getkey):
        """Output shapes are correct."""
        T, d = 5, 3
        means = jax.random.normal(getkey(), (T, d))
        covs = jnp.tile(jnp.eye(d), (T, 1, 1))
        cross_covs = jnp.tile(0.9 * jnp.eye(d), (T - 1, 1, 1))
        joint_means, joint_covs = pairwise_marginals(means, covs, cross_covs)
        assert joint_means.shape == (T - 1, 2 * d)
        assert joint_covs.shape == (T - 1, 2 * d, 2 * d)

    def test_marginals_recover_individual(self, getkey):
        """Marginalizing the joint recovers individual marginals."""
        T, d = 4, 2
        means = jax.random.normal(getkey(), (T, d))
        covs = jnp.tile(jnp.eye(d), (T, 1, 1))
        cross_covs = jnp.tile(0.25 * jnp.eye(d), (T - 1, 1, 1))
        joint_means, joint_covs = pairwise_marginals(means, covs, cross_covs)
        # First marginal of each joint should be means[k]
        for k in range(T - 1):
            assert jnp.allclose(joint_means[k, :d], means[k], atol=1e-6)
            assert jnp.allclose(joint_means[k, d:], means[k + 1], atol=1e-6)
            assert jnp.allclose(joint_covs[k, :d, :d], covs[k], atol=1e-6)
            assert jnp.allclose(joint_covs[k, d:, d:], covs[k + 1], atol=1e-6)

    def test_joint_covariance_symmetric(self, getkey):
        """Joint covariance should be symmetric."""
        T, d = 4, 3
        means = jax.random.normal(getkey(), (T, d))
        covs = jnp.tile(jnp.eye(d), (T, 1, 1))
        cross_covs = jnp.tile(0.8 * jnp.eye(d), (T - 1, 1, 1))
        _, joint_covs = pairwise_marginals(means, covs, cross_covs)
        for k in range(T - 1):
            assert jnp.allclose(joint_covs[k], joint_covs[k].T, atol=1e-8)

    def test_cross_covariance_correct(self, getkey):
        """Cross-covariance block equals the supplied Cov[x_{k+1}, x_k]."""
        T, d = 3, 2
        means = jax.random.normal(getkey(), (T, d))
        P = jax.random.normal(getkey(), (d, d))
        P = P @ P.T + jnp.eye(d)
        covs = jnp.tile(P, (T, 1, 1))
        C = jax.random.normal(getkey(), (d, d)) * 0.1
        cross_covs = jnp.tile(C, (T - 1, 1, 1))
        _, joint_covs = pairwise_marginals(means, covs, cross_covs)
        assert jnp.allclose(joint_covs[0, d:, :d], C, atol=1e-6)
        assert jnp.allclose(joint_covs[0, :d, d:], C.T, atol=1e-6)

    def test_joint_psd(self, getkey):
        """Joint covariance should be positive semi-definite."""
        T, d = 4, 2
        means = jax.random.normal(getkey(), (T, d))
        P = jax.random.normal(getkey(), (d, d))
        P = P @ P.T + jnp.eye(d)
        covs = jnp.tile(P, (T, 1, 1))
        cross_covs = jnp.tile(0.5 * P, (T - 1, 1, 1))
        _, joint_covs = pairwise_marginals(means, covs, cross_covs)
        for k in range(T - 1):
            eigvals = jnp.linalg.eigvalsh(joint_covs[k])
            assert jnp.all(eigvals >= -1e-8)
