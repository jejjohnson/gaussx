"""Tests for natural gradient primitives."""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx._einx import einsum, rearrange
from gaussx._inference._natural_gradient import (
    damped_natural_update,
    gauss_newton_precision,
    riemannian_psd_correction,
)
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._testing import key_sequence, random_pd_matrix


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

    @pytest.mark.slow
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

    def test_generic_operator(self):
        """Generic operators should materialize to MatrixLinearOperator."""
        nat1 = jnp.zeros(3)
        nat2_old = lx.MatrixLinearOperator(jnp.eye(3))
        nat2_target = lx.MatrixLinearOperator(2.0 * jnp.eye(3))

        _, nat2_new = damped_natural_update(nat1, nat2_old, nat1, nat2_target, lr=0.5)
        assert isinstance(nat2_new, lx.MatrixLinearOperator)
        assert jnp.allclose(nat2_new.as_matrix(), 1.5 * jnp.eye(3))

    def test_numpy_arrays_are_accepted(self):
        """gh-394: NumPy arrays dispatch as arrays, not as a type mismatch."""
        nat1, nat2 = damped_natural_update(
            np.zeros(3), np.eye(3), np.ones(3), 2.0 * np.eye(3), 0.5
        )
        assert isinstance(nat2, jax.Array)
        assert jnp.allclose(nat2, 1.5 * jnp.eye(3))
        assert jnp.allclose(nat1, 0.5 * jnp.ones(3))

    def test_mixed_types_error_names_both(self):
        """gh-394: an array mixed with an operator names both types."""
        with pytest.raises(
            TypeError,
            match=r"both be arrays or both be linear operators.*"
            r"ArrayImpl and MatrixLinearOperator",
        ):
            damped_natural_update(
                jnp.zeros(3),
                jnp.eye(3),
                jnp.zeros(3),
                lx.MatrixLinearOperator(jnp.eye(3)),
                0.5,
            )

    def test_traced_learning_rate(self):
        """gh-394: ``lr`` may be a traced scalar (``jax.grad`` through it)."""

        def loss(lr):
            _, nat2 = damped_natural_update(
                jnp.zeros(2), jnp.eye(2), jnp.zeros(2), 3.0 * jnp.eye(2), lr
            )
            return jnp.trace(nat2)

        assert jnp.allclose(jax.grad(loss)(jnp.asarray(0.5)), 4.0)


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

    # gh-401: what the correction guarantees is not that the corrected Hessian
    # is negative semi-definite (for H = diag(-1, 2), Lambda = I, lr = 0.1 it
    # has eigenvalues [-1, 1.55]) but that the damped precision update
    # (1 - lr) Lambda - lr H_psd is positive definite: with Sigma = Lambda^-1
    # and G = Lambda + H it equals
    # 1/2 Lambda + 1/2 (I - lr G Sigma) Lambda (I - lr Sigma G),
    # PD plus PSD. The 1e-12 tolerance is float64 round-off on a 2 x 2 or
    # 3 x 3 problem.
    @pytest.mark.x64_only(reason="identity checked to float64 round-off")
    @pytest.mark.parametrize("lr", [0.1, 0.5, 1.0])
    def test_damped_precision_update_is_positive_definite(self, lr):
        H = jnp.diag(jnp.array([-1.0, 2.0]))
        Lam = jnp.eye(2)

        H_psd = riemannian_psd_correction(H, Lam, jnp.linalg.inv(Lam), lr=lr)
        updated = (1.0 - lr) * Lam - lr * H_psd

        assert jnp.min(jnp.linalg.eigvalsh(updated)) > 0.0
        if lr == 0.1:
            # The corrected Hessian itself stays indefinite.
            assert jnp.allclose(
                jnp.linalg.eigvalsh(H_psd), jnp.array([-1.0, 1.55]), atol=1e-12
            )

    @pytest.mark.x64_only(reason="identity checked to float64 round-off")
    @pytest.mark.parametrize("lr", [0.1, 0.5, 1.0])
    def test_damped_precision_update_identity(self, lr):
        nextkey = key_sequence(0)
        d = 3
        A = jax.random.normal(nextkey(), (d, d))
        H = 0.5 * (A + rearrange(A, "i j -> j i"))  # symmetric, indefinite
        Lam = random_pd_matrix(nextkey(), d, jitter=1.0)
        Sigma = jnp.linalg.inv(Lam)
        G = Lam + H

        H_psd = riemannian_psd_correction(H, Lam, Sigma, lr=lr)
        updated = (1.0 - lr) * Lam - lr * H_psd

        left = jnp.eye(d) - lr * einsum(G, Sigma, "i k, k j -> i j")
        right = jnp.eye(d) - lr * einsum(Sigma, G, "i k, k j -> i j")
        expected = 0.5 * Lam + 0.5 * einsum(left, Lam, right, "i a, a b, b j -> i j")
        assert jnp.min(jnp.linalg.eigvalsh(H)) < 0.0
        assert jnp.allclose(updated, expected, atol=1e-12, rtol=0.0)
        assert jnp.min(jnp.linalg.eigvalsh(updated)) > 0.0


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


class TestGaussNewtonPrecisionWithBase:
    """gh-334: a prior precision makes J^T J invertible and keeps Woodbury."""

    @pytest.mark.parametrize("shape", [(2, 4), (5, 3)], ids=["wide", "tall"])
    def test_matches_dense_posterior_precision(self, shape):
        from gaussx._operators._low_rank_update import LowRankUpdate

        J = jax.random.normal(jax.random.key(0), shape)
        D_latent = shape[1]
        prior = lx.DiagonalLinearOperator(2.0 * jnp.ones(D_latent))
        op = gauss_newton_precision(J, base=prior)
        assert isinstance(op, LowRankUpdate)
        assert op.base is prior
        assert lx.is_symmetric(op)
        M = J.T @ J + 2.0 * jnp.eye(D_latent)
        b = jnp.ones(D_latent)
        assert jnp.allclose(op.as_matrix(), M, atol=1e-12)
        assert jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(M, b), atol=1e-12)
        assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(M)[1], atol=1e-12)

    def test_psd_tag_follows_the_base(self):
        J = jax.random.normal(jax.random.key(0), (2, 4))
        tagged = lx.MatrixLinearOperator(2.0 * jnp.eye(4), lx.positive_semidefinite_tag)
        assert lx.is_positive_semidefinite(gauss_newton_precision(J, base=tagged))
        untagged = lx.MatrixLinearOperator(2.0 * jnp.eye(4))
        assert not lx.is_positive_semidefinite(gauss_newton_precision(J, base=untagged))

    def test_base_shape_mismatch_raises(self):
        J = jax.random.normal(jax.random.key(0), (2, 4))
        with pytest.raises(ValueError, match="base must be"):
            gauss_newton_precision(J, base=lx.DiagonalLinearOperator(jnp.ones(3)))

    def test_without_base_is_documented_singular(self):
        # Pinned: rank D_obs < D_latent, so there is no inverse to compute.
        J = jax.random.normal(jax.random.key(0), (2, 4))
        op = gauss_newton_precision(J)
        assert jnp.linalg.matrix_rank(op.as_matrix()) == 2
        assert not jnp.all(jnp.isfinite(gaussx.solve(op, jnp.ones(4))))
