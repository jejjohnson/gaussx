"""Tests for natural gradient primitives."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._sugar._natural_gradient import (
    damped_natural_update,
    gauss_newton_precision,
    riemannian_psd_correction,
)


def _make_block_tridiag(getkey, N=3, d=2):
    """Helper to build a test BlockTriDiag."""
    diag_raw = jax.random.normal(getkey(), (N, d, d))
    diag = jax.vmap(lambda A: A @ A.T + 2.0 * jnp.eye(d))(diag_raw)
    sub = 0.1 * jax.random.normal(getkey(), (N - 1, d, d))
    return BlockTriDiag(diag, sub)


class TestDampedNaturalUpdate:
    def test_lr_zero_returns_old(self, getkey):
        """lr=0 should return old parameters unchanged."""
        nat1_old = jax.random.normal(getkey(), (4,))
        nat2_old = jax.random.normal(getkey(), (4, 4))
        nat1_target = jax.random.normal(getkey(), (4,))
        nat2_target = jax.random.normal(getkey(), (4, 4))

        nat1_new, nat2_new = damped_natural_update(
            nat1_old, nat2_old, nat1_target, nat2_target, lr=0.0
        )
        assert jnp.allclose(nat1_new, nat1_old)
        assert jnp.allclose(nat2_new, nat2_old)

    def test_lr_one_returns_target(self, getkey):
        """lr=1 should return target parameters."""
        nat1_old = jax.random.normal(getkey(), (4,))
        nat2_old = jax.random.normal(getkey(), (4, 4))
        nat1_target = jax.random.normal(getkey(), (4,))
        nat2_target = jax.random.normal(getkey(), (4, 4))

        nat1_new, nat2_new = damped_natural_update(
            nat1_old, nat2_old, nat1_target, nat2_target, lr=1.0
        )
        assert jnp.allclose(nat1_new, nat1_target)
        assert jnp.allclose(nat2_new, nat2_target)

    def test_block_tridiag_preserved(self, getkey):
        """BlockTriDiag structure should be preserved."""
        nat1_old = jax.random.normal(getkey(), (6,))
        nat2_old = _make_block_tridiag(getkey)
        nat1_target = jax.random.normal(getkey(), (6,))
        nat2_target = _make_block_tridiag(getkey)

        _, nat2_new = damped_natural_update(
            nat1_old, nat2_old, nat1_target, nat2_target, lr=0.5
        )
        assert isinstance(nat2_new, BlockTriDiag)

    def test_block_tridiag_interpolation(self, getkey):
        """BlockTriDiag interpolation should match dense interpolation."""
        nat2_old = _make_block_tridiag(getkey)
        nat2_target = _make_block_tridiag(getkey)
        nat1 = jnp.zeros(6)

        _, nat2_new = damped_natural_update(nat1, nat2_old, nat1, nat2_target, lr=0.3)

        expected = 0.7 * nat2_old.as_matrix() + 0.3 * nat2_target.as_matrix()
        assert jnp.allclose(nat2_new.as_matrix(), expected, atol=1e-10)

    def test_generic_operator(self, getkey):
        """Generic operators should materialize to MatrixLinearOperator."""
        nat1 = jnp.zeros(3)
        nat2_old = lx.MatrixLinearOperator(jnp.eye(3))
        nat2_target = lx.MatrixLinearOperator(2.0 * jnp.eye(3))

        _, nat2_new = damped_natural_update(nat1, nat2_old, nat1, nat2_target, lr=0.5)
        assert isinstance(nat2_new, lx.MatrixLinearOperator)
        assert jnp.allclose(nat2_new.as_matrix(), 1.5 * jnp.eye(3))


class TestRiemannianPSDCorrection:
    def test_shape(self, getkey):
        """Output should match input shape."""
        d = 4
        H = jax.random.normal(getkey(), (d, d))
        S_prec = jax.random.normal(getkey(), (d, d))
        S_prec = S_prec @ S_prec.T
        S_cov = jnp.linalg.inv(S_prec)

        result = riemannian_psd_correction(H, S_prec, S_cov, lr=1.0)
        assert result.shape == (d, d)

    def test_zero_lr_returns_hessian(self, getkey):
        """lr=0 should return the original Hessian."""
        d = 3
        H = jax.random.normal(getkey(), (d, d))
        S_prec = jnp.eye(d)
        S_cov = jnp.eye(d)

        result = riemannian_psd_correction(H, S_prec, S_cov, lr=0.0)
        assert jnp.allclose(result, H)


class TestGaussNewtonPrecision:
    def test_low_rank_when_obs_lt_latent(self, getkey):
        """Should return LowRankUpdate when D_obs < D_latent."""
        from gaussx._operators._low_rank_update import LowRankUpdate

        J = jax.random.normal(getkey(), (3, 10))
        op = gauss_newton_precision(J)
        assert isinstance(op, LowRankUpdate)

    def test_dense_when_obs_ge_latent(self, getkey):
        """Should return MatrixLinearOperator when D_obs >= D_latent."""
        J = jax.random.normal(getkey(), (10, 3))
        op = gauss_newton_precision(J)
        assert isinstance(op, lx.MatrixLinearOperator)

    def test_matches_dense(self, getkey):
        """Should match J^T J."""
        J = jax.random.normal(getkey(), (5, 8))
        op = gauss_newton_precision(J)
        expected = J.T @ J
        assert jnp.allclose(op.as_matrix(), expected, atol=1e-6)

    def test_psd(self, getkey):
        """Result should always be PSD."""
        J = jax.random.normal(getkey(), (3, 6))
        op = gauss_newton_precision(J)
        eigvals = jnp.linalg.eigvalsh(op.as_matrix())
        assert jnp.all(eigvals >= -1e-10)
