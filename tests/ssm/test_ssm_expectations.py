"""Tests for SSM <-> expectation parameter transformations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._ssm._ssm_natural import (
    expectations_to_ssm,
    ssm_to_expectations,
)
from gaussx._testing import tree_allclose


def _make_smoothed_marginals(getkey, N=4, d=2):
    """Build synthetic smoothed marginals for testing."""
    means = jax.random.normal(getkey(), (N, d))

    # PD covariances
    raw = jax.random.normal(getkey(), (N, d, d))
    covs = jax.vmap(lambda M: M @ M.T + 0.1 * jnp.eye(d))(raw)

    # Cross-covariances (not necessarily PD, just (N-1, d, d))
    cross_covs = 0.3 * jax.random.normal(getkey(), (N - 1, d, d))

    return means, covs, cross_covs


class TestSSMToExpectations:
    def test_shapes(self, getkey):
        """Output shapes should be correct."""
        N, d = 5, 3
        means, covs, cross_covs = _make_smoothed_marginals(getkey, N, d)

        eta1, eta2 = ssm_to_expectations(means, covs, cross_covs)

        assert eta1.shape == (N * d,)
        assert isinstance(eta2, BlockTriDiag)
        assert eta2._num_blocks == N
        assert eta2._block_size == d

    def test_diagonal_blocks(self, getkey):
        """Diagonal blocks should be E[x_k x_k^T] = P_k + m_k m_k^T."""
        N, d = 4, 2
        means, covs, cross_covs = _make_smoothed_marginals(getkey, N, d)

        _, eta2 = ssm_to_expectations(means, covs, cross_covs)

        expected_diag = covs + jax.vmap(jnp.outer)(means, means)
        assert tree_allclose(eta2.diagonal, expected_diag, rtol=1e-6)

    def test_sub_diagonal_blocks(self, getkey):
        """Sub-diagonal blocks should be E[x_{k+1} x_k^T] = C_k + m_{k+1} m_k^T."""
        N, d = 4, 2
        means, covs, cross_covs = _make_smoothed_marginals(getkey, N, d)

        _, eta2 = ssm_to_expectations(means, covs, cross_covs)

        expected_sub = cross_covs + jax.vmap(jnp.outer)(means[1:], means[:-1])
        assert tree_allclose(eta2.sub_diagonal, expected_sub, rtol=1e-6)

    def test_eta1_is_concatenated_means(self, getkey):
        """eta1 should be the flattened means."""
        N, d = 3, 4
        means, covs, cross_covs = _make_smoothed_marginals(getkey, N, d)

        eta1, _ = ssm_to_expectations(means, covs, cross_covs)

        assert tree_allclose(eta1, means.reshape(N * d), atol=1e-12)


class TestExpectationsToSSM:
    def test_roundtrip(self, getkey):
        """ssm -> expectations -> ssm should recover original parameters."""
        N, d = 4, 2
        means, covs, cross_covs = _make_smoothed_marginals(getkey, N, d)

        eta1, eta2 = ssm_to_expectations(means, covs, cross_covs)
        means_rec, covs_rec, cross_covs_rec = expectations_to_ssm(eta1, eta2)

        assert tree_allclose(means_rec, means, rtol=1e-6)
        assert tree_allclose(covs_rec, covs, rtol=1e-6)
        assert tree_allclose(cross_covs_rec, cross_covs, rtol=1e-6)

    def test_roundtrip_single_step(self, getkey):
        """Roundtrip with N=2 (minimal case with one cross-covariance)."""
        N, d = 2, 3
        means, covs, cross_covs = _make_smoothed_marginals(getkey, N, d)

        eta1, eta2 = ssm_to_expectations(means, covs, cross_covs)
        means_rec, covs_rec, cross_covs_rec = expectations_to_ssm(eta1, eta2)

        assert tree_allclose(means_rec, means, rtol=1e-6)
        assert tree_allclose(covs_rec, covs, rtol=1e-6)
        assert tree_allclose(cross_covs_rec, cross_covs, rtol=1e-6)

    def test_shapes(self, getkey):
        """Recovered shapes should match original."""
        N, d = 5, 3
        means, covs, cross_covs = _make_smoothed_marginals(getkey, N, d)

        eta1, eta2 = ssm_to_expectations(means, covs, cross_covs)
        means_rec, covs_rec, cross_covs_rec = expectations_to_ssm(eta1, eta2)

        assert means_rec.shape == (N, d)
        assert covs_rec.shape == (N, d, d)
        assert cross_covs_rec.shape == (N - 1, d, d)
