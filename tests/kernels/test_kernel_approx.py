"""Tests for kernel approximation sugar: Nystrom, RFF, centering, HSIC, MMD."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._kernels._kernel_approx import (
    center_kernel,
    centering_operator,
    hsic,
    mmd_squared,
    nystrom_operator,
    rff_operator,
)


def _rbf_kernel(x, y, lengthscale=1.0):
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2) / lengthscale**2)


class TestNystromOperator:
    def test_shape(self, getkey):
        """Output should be N x N."""
        N, M = 10, 3
        K_XZ = jax.random.normal(getkey(), (N, M))
        K_ZZ = jax.random.normal(getkey(), (M, M))
        K_ZZ = K_ZZ @ K_ZZ.T + 0.1 * jnp.eye(M)
        K_ZZ_op = lx.MatrixLinearOperator(K_ZZ, lx.positive_semidefinite_tag)

        op = nystrom_operator(K_XZ, K_ZZ_op)
        mat = op.as_matrix()
        assert mat.shape == (N, N)

    def test_approximation(self, getkey):
        """Nystrom should approximate K_XZ K_ZZ^{-1} K_ZX."""
        N, M, D = 8, 3, 2
        X = jax.random.normal(getkey(), (N, D))
        Z = jax.random.normal(getkey(), (M, D))

        K_XZ = jax.vmap(lambda x: jax.vmap(lambda z: _rbf_kernel(x, z))(Z))(X)
        K_ZZ = jax.vmap(lambda z1: jax.vmap(lambda z2: _rbf_kernel(z1, z2))(Z))(Z)
        K_ZZ = K_ZZ + 0.01 * jnp.eye(M)
        K_ZZ_op = lx.MatrixLinearOperator(K_ZZ, lx.positive_semidefinite_tag)

        op = nystrom_operator(K_XZ, K_ZZ_op)
        approx = op.as_matrix()

        # Reference
        ref = K_XZ @ jnp.linalg.solve(K_ZZ, K_XZ.T)
        assert jnp.allclose(approx, ref, atol=1e-4)

    def test_symmetric_psd(self, getkey):
        """Result should be symmetric and PSD."""
        N, M = 6, 3
        K_XZ = jax.random.normal(getkey(), (N, M))
        K_ZZ = jax.random.normal(getkey(), (M, M))
        K_ZZ = K_ZZ @ K_ZZ.T + 0.1 * jnp.eye(M)
        K_ZZ_op = lx.MatrixLinearOperator(K_ZZ, lx.positive_semidefinite_tag)

        op = nystrom_operator(K_XZ, K_ZZ_op)
        assert lx.is_symmetric(op)
        assert lx.is_positive_semidefinite(op)


class TestRFFOperator:
    def test_shape(self, getkey):
        """Output should be N x N."""
        N, D, D_rff = 10, 3, 20
        X = jax.random.normal(getkey(), (N, D))
        omega = jax.random.normal(getkey(), (D_rff, D))
        b = jax.random.uniform(getkey(), (D_rff,), maxval=2 * jnp.pi)

        op = rff_operator(X, omega, b)
        mat = op.as_matrix()
        assert mat.shape == (N, N)

    def test_symmetric_psd(self, getkey):
        """Result should be symmetric and PSD."""
        N, D, D_rff = 8, 2, 15
        X = jax.random.normal(getkey(), (N, D))
        omega = jax.random.normal(getkey(), (D_rff, D))
        b = jax.random.uniform(getkey(), (D_rff,), maxval=2 * jnp.pi)

        op = rff_operator(X, omega, b)
        mat = op.as_matrix()
        assert jnp.allclose(mat, mat.T, atol=1e-6)
        eigvals = jnp.linalg.eigvalsh(mat)
        assert jnp.all(eigvals >= -1e-6)


class TestCenteringOperator:
    def test_shape(self):
        """Should be n x n."""
        n = 5
        op = centering_operator(n)
        assert op.as_matrix().shape == (n, n)

    def test_idempotent(self):
        """H^2 = H (centering is a projection)."""
        n = 6
        H = centering_operator(n).as_matrix()
        assert jnp.allclose(H @ H, H, atol=1e-6)

    def test_centers_vector(self):
        """H @ x should have zero mean."""
        n = 5
        H = centering_operator(n).as_matrix()
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        centered = H @ x
        assert jnp.allclose(jnp.mean(centered), 0.0, atol=1e-10)


class TestCenterKernel:
    def test_row_col_mean_zero(self, getkey):
        """Centered kernel should have zero row and column means."""
        N = 6
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T
        K_op = lx.MatrixLinearOperator(K, lx.symmetric_tag)

        K_c = center_kernel(K_op).as_matrix()
        assert jnp.allclose(jnp.mean(K_c, axis=0), 0.0, atol=1e-10)
        assert jnp.allclose(jnp.mean(K_c, axis=1), 0.0, atol=1e-10)

    def test_symmetric(self, getkey):
        """Centered kernel should be symmetric if input is."""
        N = 5
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T
        K_op = lx.MatrixLinearOperator(K, lx.symmetric_tag)

        K_c = center_kernel(K_op)
        assert lx.is_symmetric(K_c)


class TestHSIC:
    def test_independent_near_zero(self, getkey):
        """HSIC of independent features should be near zero."""
        N = 50
        X = jax.random.normal(getkey(), (N, 1))
        Y = jax.random.normal(getkey(), (N, 1))
        K_f = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        K_q = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(Y))(Y)
        K_f_op = lx.MatrixLinearOperator(K_f, lx.symmetric_tag)
        K_q_op = lx.MatrixLinearOperator(K_q, lx.symmetric_tag)

        h = hsic(K_f_op, K_q_op)
        assert jnp.abs(h) < 0.1

    def test_self_hsic_positive(self, getkey):
        """HSIC(K, K) should be positive."""
        N = 20
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T + 0.1 * jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.symmetric_tag)

        h = hsic(K_op, K_op)
        assert h > 0

    def test_scalar(self, getkey):
        """Should return a scalar."""
        N = 10
        K = jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.symmetric_tag)
        h = hsic(K_op, K_op)
        assert h.shape == ()


class TestMMDSquared:
    def test_same_distribution_zero(self, getkey):
        """MMD^2 of identical distributions should be near zero."""
        N = 20
        X = jax.random.normal(getkey(), (N, 2))
        K_xx = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        m = mmd_squared(K_xx, K_xx, K_xx)
        assert jnp.allclose(m, 0.0, atol=1e-6)

    def test_different_distributions_positive(self, getkey):
        """MMD^2 of different distributions should be positive."""
        N = 20
        X = jax.random.normal(getkey(), (N, 2))
        Y = jax.random.normal(getkey(), (N, 2)) + 3.0
        K_xx = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        K_yy = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(Y))(Y)
        K_xy = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(Y))(X)
        m = mmd_squared(K_xx, K_yy, K_xy)
        assert m > 0

    def test_scalar(self, getkey):
        """Should return a scalar."""
        N = 5
        K_xx = jnp.eye(N)
        K_yy = jnp.eye(N)
        K_xy = jnp.zeros((N, N))
        m = mmd_squared(K_xx, K_yy, K_xy)
        assert m.shape == ()
