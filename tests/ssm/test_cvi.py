"""Tests for CVI — Conjugate-computation Variational Inference sites."""

import jax
import jax.numpy as jnp

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._ssm._cvi import GaussianSites, cvi_update_sites, sites_to_precision


def _make_sites(getkey, N=4, d=2):
    """Build test Gaussian sites."""
    nat1 = jax.random.normal(getkey(), (N, d))
    nat2_raw = jax.random.normal(getkey(), (N, d, d))
    nat2 = -0.5 * jax.vmap(lambda M: M @ M.T + 0.1 * jnp.eye(d))(nat2_raw)
    return GaussianSites(nat1=nat1, nat2=nat2)


class TestGaussianSites:
    def test_is_pytree(self, getkey):
        """GaussianSites should be a valid JAX pytree."""
        sites = _make_sites(getkey)
        leaves = jax.tree.leaves(sites)
        assert len(leaves) == 2

    def test_shapes(self, getkey):
        """Fields should have correct shapes."""
        N, d = 5, 3
        nat1 = jnp.zeros((N, d))
        nat2 = jnp.zeros((N, d, d))
        sites = GaussianSites(nat1=nat1, nat2=nat2)
        assert sites.nat1.shape == (N, d)
        assert sites.nat2.shape == (N, d, d)


class TestCVIUpdateSites:
    def test_rho_zero_no_change(self, getkey):
        """rho=0 should return original sites."""
        sites = _make_sites(getkey)
        grad1 = jax.random.normal(getkey(), sites.nat1.shape)
        grad2 = jax.random.normal(getkey(), sites.nat2.shape)

        updated = cvi_update_sites(sites, grad1, grad2, rho=0.0)
        assert jnp.allclose(updated.nat1, sites.nat1)
        assert jnp.allclose(updated.nat2, sites.nat2)

    def test_rho_one_returns_grad(self, getkey):
        """rho=1 should return the gradient values."""
        sites = _make_sites(getkey)
        grad1 = jax.random.normal(getkey(), sites.nat1.shape)
        grad2 = jax.random.normal(getkey(), sites.nat2.shape)

        updated = cvi_update_sites(sites, grad1, grad2, rho=1.0)
        assert jnp.allclose(updated.nat1, grad1)
        assert jnp.allclose(updated.nat2, grad2)

    def test_interpolation(self, getkey):
        """Should linearly interpolate between old and gradient."""
        sites = _make_sites(getkey)
        grad1 = jax.random.normal(getkey(), sites.nat1.shape)
        grad2 = jax.random.normal(getkey(), sites.nat2.shape)
        rho = 0.3

        updated = cvi_update_sites(sites, grad1, grad2, rho=rho)
        expected_nat1 = 0.7 * sites.nat1 + 0.3 * grad1
        assert jnp.allclose(updated.nat1, expected_nat1)


class TestSitesToPrecision:
    def test_returns_block_tridiag(self, getkey):
        """Should return a BlockTriDiag."""
        sites = _make_sites(getkey)
        prec = sites_to_precision(sites)
        assert isinstance(prec, BlockTriDiag)

    def test_correct_dimensions(self, getkey):
        """BlockTriDiag dimensions should match sites."""
        N, d = 5, 3
        sites = GaussianSites(
            nat1=jnp.zeros((N, d)),
            nat2=-0.5 * jnp.tile(jnp.eye(d)[None], (N, 1, 1)),
        )
        prec = sites_to_precision(sites)
        assert prec._num_blocks == N
        assert prec._block_size == d

    def test_zero_sub_diagonals(self, getkey):
        """Sub-diagonal blocks should all be zero."""
        sites = _make_sites(getkey)
        prec = sites_to_precision(sites)
        assert jnp.allclose(prec.sub_diagonal, 0.0)

    def test_diagonal_is_precision(self, getkey):
        """Diagonal blocks should be -2 * nat2 (raw precision)."""
        sites = _make_sites(getkey)
        prec = sites_to_precision(sites)
        expected = -2.0 * sites.nat2
        assert jnp.allclose(prec.diagonal, expected)

    def test_addable_to_prior(self, getkey):
        """Should be addable to a prior BlockTriDiag."""
        N, d = 4, 2
        diag_raw = jax.random.normal(getkey(), (N, d, d))
        diag = jax.vmap(lambda A: A @ A.T + 3.0 * jnp.eye(d))(diag_raw)
        sub = 0.1 * jax.random.normal(getkey(), (N - 1, d, d))
        prior = BlockTriDiag(diag, sub)

        sites = _make_sites(getkey, N, d)
        site_prec = sites_to_precision(sites)

        posterior = prior + site_prec
        assert isinstance(posterior, BlockTriDiag)
        assert jnp.allclose(posterior.diagonal, prior.diagonal + site_prec.diagonal)
