"""Tests for whitened SVGP forward pass."""

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._gp._base_conditional import sparse_conditional
from gaussx._gp._svgp import whitened_svgp_predict
from gaussx._testing import key_sequence, psd_operator, random_pd_matrix


class TestWhitenedSVGPPredict:
    @pytest.mark.slow
    def test_basic_shapes(self):
        """Output shapes should match number of test points."""
        nextkey = key_sequence(0)
        M, N = 5, 10
        K_zz = jax.random.normal(nextkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.01 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = jax.random.normal(nextkey(), (N, M))
        u_mean = jax.random.normal(nextkey(), (M,))
        u_chol = jnp.linalg.cholesky(jnp.eye(M))
        K_xx_diag = jnp.ones(N)

        f_loc, f_var = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        assert f_loc.shape == (N,)
        assert f_var.shape == (N,)

    @pytest.mark.slow
    def test_nonnegative_variance(self):
        """Predictive variances should be non-negative."""
        nextkey = key_sequence(0)
        M, N = 4, 8
        K_zz = jax.random.normal(nextkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = jax.random.normal(nextkey(), (N, M))
        u_mean = jax.random.normal(nextkey(), (M,))
        u_chol = 0.5 * jnp.linalg.cholesky(jnp.eye(M))
        K_xx_diag = 2.0 * jnp.ones(N)

        _, f_var = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        assert jnp.all(f_var >= 0.0)

    def test_zero_u_mean_gives_zero_mean(self):
        """With u_mean=0, predictive mean should be zero."""
        nextkey = key_sequence(0)
        M, N = 4, 6
        K_zz = jax.random.normal(nextkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = jax.random.normal(nextkey(), (N, M))
        u_mean = jnp.zeros(M)
        u_chol = jnp.linalg.cholesky(jnp.eye(M))
        K_xx_diag = jnp.ones(N)

        f_loc, _ = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        assert jnp.allclose(f_loc, 0.0, atol=1e-10)

    def test_identity_chol_reduces_variance(self):
        """With identity u_chol, posterior should reduce prior variance."""
        nextkey = key_sequence(0)
        M, N = 4, 6
        K_zz = jax.random.normal(nextkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = 0.5 * jax.random.normal(nextkey(), (N, M))
        u_mean = jnp.zeros(M)
        # u_chol = 0 means zero posterior covariance in whitened space
        u_chol = jnp.zeros((M, M))
        K_xx_diag = 2.0 * jnp.ones(N)

        _, f_var = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        # Variance should be less than prior
        assert jnp.all(f_var <= K_xx_diag + 1e-6)

    @pytest.mark.slow
    def test_jit(self):
        """Should be JIT-compatible."""
        nextkey = key_sequence(0)
        M, N = 3, 5
        K_zz = jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        K_xz = jax.random.normal(nextkey(), (N, M))
        u_mean = jnp.zeros(M)
        u_chol = jnp.eye(M)
        K_xx_diag = jnp.ones(N)

        f_loc1, f_var1 = whitened_svgp_predict(K_zz_op, K_xz, u_mean, u_chol, K_xx_diag)
        f_loc2, f_var2 = jax.jit(whitened_svgp_predict)(
            K_zz_op, K_xz, u_mean, u_chol, K_xx_diag
        )
        assert jnp.allclose(f_loc1, f_loc2, atol=1e-10)
        assert jnp.allclose(f_var1, f_var2, atol=1e-10)


@pytest.mark.x64_only(reason="1e-12 agreement in float64")
def test_matches_sparse_conditional_whitened():
    """gh-392: whitened_svgp_predict == sparse_conditional(white=True), R = 1."""
    M, N = 4, 6
    k = jr.split(jr.key(0), 5)
    K_zz = random_pd_matrix(k[0], M)
    K_xz = jr.normal(k[1], (N, M))
    # A valid prior diagonal: diag(K_post) + diag(Q_xx) with K_post PD.
    K_xx_diag = jnp.diag(
        random_pd_matrix(k[2], N) + K_xz @ jnp.linalg.solve(K_zz, K_xz.T)
    )
    u = jr.normal(k[3], (M,))
    L_u = jnp.tril(jr.normal(k[4], (M, M))) + 0.5 * jnp.eye(M)
    m1, v1 = whitened_svgp_predict(psd_operator(K_zz), K_xz, u, L_u, K_xx_diag)
    m2, v2 = sparse_conditional(K_zz, K_xz, K_xx_diag, u, q_sqrt=L_u, white=True)
    assert jnp.allclose(m1, m2, rtol=0, atol=1e-12)
    assert jnp.allclose(v1, v2, rtol=0, atol=1e-12)
