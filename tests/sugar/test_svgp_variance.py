"""Tests for SVGP variance adjustment operator."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._sugar._svgp_variance import svgp_variance_adjustment


class TestSVGPVarianceAdjustment:
    def test_basic_shape(self, getkey):
        """Output operator should have correct shape."""
        M = 5
        K_zz = jax.random.normal(getkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        S_u = lx.MatrixLinearOperator(jnp.eye(M), lx.positive_semidefinite_tag)

        Q = svgp_variance_adjustment(K_zz_op, S_u)
        Q_mat = Q.as_matrix()
        assert Q_mat.shape == (M, M)

    def test_identity_covariance(self, getkey):
        """With S_u = K_zz, Q should be zero (posterior = prior)."""
        M = 4
        K_zz = jax.random.normal(getkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        # S_u = K_zz means q(u) = p(u), so Q = K^{-1} K K^{-1} - K^{-1} = 0
        S_u = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)

        Q = svgp_variance_adjustment(K_zz_op, S_u)
        Q_mat = Q.as_matrix()
        assert jnp.allclose(Q_mat, 0.0, atol=1e-5)

    def test_reference_dense(self, getkey):
        """Should match dense numpy computation."""
        M = 4
        K_zz = jax.random.normal(getkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        S_u = jax.random.normal(getkey(), (M, M))
        S_u = S_u @ S_u.T + 0.01 * jnp.eye(M)

        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        S_u_op = lx.MatrixLinearOperator(S_u, lx.positive_semidefinite_tag)

        Q = svgp_variance_adjustment(K_zz_op, S_u_op)
        Q_mat = Q.as_matrix()

        # Dense reference
        K_inv = jnp.linalg.inv(K_zz)
        Q_ref = K_inv @ S_u @ K_inv - K_inv
        assert jnp.allclose(Q_mat, Q_ref, atol=1e-4)

    def test_matvec(self, getkey):
        """Operator matvec should match dense result."""
        M = 5
        K_zz = jax.random.normal(getkey(), (M, M))
        K_zz = K_zz @ K_zz.T + 0.1 * jnp.eye(M)
        S_u = jax.random.normal(getkey(), (M, M))
        S_u = S_u @ S_u.T + 0.01 * jnp.eye(M)

        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        S_u_op = lx.MatrixLinearOperator(S_u, lx.positive_semidefinite_tag)

        Q = svgp_variance_adjustment(K_zz_op, S_u_op)
        v = jax.random.normal(getkey(), (M,))

        result = Q.mv(v)
        K_inv = jnp.linalg.inv(K_zz)
        expected = (K_inv @ S_u @ K_inv - K_inv) @ v
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_jit(self, getkey):
        """Should be JIT-compatible."""
        M = 3
        K_zz = jnp.eye(M)
        K_zz_op = lx.MatrixLinearOperator(K_zz, lx.positive_semidefinite_tag)
        S_u = 0.5 * jnp.eye(M)
        S_u_op = lx.MatrixLinearOperator(S_u, lx.positive_semidefinite_tag)

        Q1 = svgp_variance_adjustment(K_zz_op, S_u_op)
        Q2 = jax.jit(svgp_variance_adjustment)(K_zz_op, S_u_op)
        assert jnp.allclose(Q1.as_matrix(), Q2.as_matrix(), atol=1e-10)
