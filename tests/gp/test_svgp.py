"""Tests for whitened SVGP forward pass."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._gp._svgp import whitened_svgp_predict


class TestWhitenedSVGPPredict:
    def test_basic_shapes(self, getkey):
        """Output shapes should match number of test points."""
        M, N = 5, 10
        K_zz = jax.random.normal(getkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.01 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = jax.random.normal(getkey(), (N, M))
        u_mean = jax.random.normal(getkey(), (M,))
        u_chol = jnp.linalg.cholesky(jnp.eye(M))
        K_xx_diag = jnp.ones(N)

        f_loc, f_var = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        assert f_loc.shape == (N,)
        assert f_var.shape == (N,)

    def test_nonnegative_variance(self, getkey):
        """Predictive variances should be non-negative."""
        M, N = 4, 8
        K_zz = jax.random.normal(getkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = jax.random.normal(getkey(), (N, M))
        u_mean = jax.random.normal(getkey(), (M,))
        u_chol = 0.5 * jnp.linalg.cholesky(jnp.eye(M))
        K_xx_diag = 2.0 * jnp.ones(N)

        _, f_var = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        assert jnp.all(f_var >= 0.0)

    def test_zero_u_mean_gives_zero_mean(self, getkey):
        """With u_mean=0, predictive mean should be zero."""
        M, N = 4, 6
        K_zz = jax.random.normal(getkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = jax.random.normal(getkey(), (N, M))
        u_mean = jnp.zeros(M)
        u_chol = jnp.linalg.cholesky(jnp.eye(M))
        K_xx_diag = jnp.ones(N)

        f_loc, _ = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        assert jnp.allclose(f_loc, 0.0, atol=1e-10)

    def test_identity_chol_reduces_variance(self, getkey):
        """With identity u_chol, posterior should reduce prior variance."""
        M, N = 4, 6
        K_zz = jax.random.normal(getkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = 0.5 * jax.random.normal(getkey(), (N, M))
        u_mean = jnp.zeros(M)
        # u_chol = 0 means zero posterior covariance in whitened space
        u_chol = jnp.zeros((M, M))
        K_xx_diag = 2.0 * jnp.ones(N)

        _, f_var = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        # Variance should be less than prior
        assert jnp.all(f_var <= K_xx_diag + 1e-6)

    def test_jit(self, getkey):
        """Should be JIT-compatible."""
        M, N = 3, 5
        K_zz = jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = jax.random.normal(getkey(), (N, M))
        u_mean = jnp.zeros(M)
        u_chol = jnp.eye(M)
        K_xx_diag = jnp.ones(N)

        f_loc1, f_var1 = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        f_loc2, f_var2 = jax.jit(whitened_svgp_predict)(
            K_zz_op, K_xz, u_mean, u_chol, K_xx_diag
        )
        assert jnp.allclose(f_loc1, f_loc2, atol=1e-10)
        assert jnp.allclose(f_var1, f_var2, atol=1e-10)
