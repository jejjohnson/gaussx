"""Tests for KroneckerSum operator."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import gaussx


def _make_psd(key, n):
    """Create a random PSD matrix."""
    M = jax.random.normal(key, (n, n))
    return M @ M.T + jnp.eye(n)


@pytest.fixture()
def kron_sum():
    """A (+) B with small PSD factors."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    A_mat = _make_psd(k1, 3)
    B_mat = _make_psd(k2, 4)
    A = lx.MatrixLinearOperator(A_mat, lx.positive_semidefinite_tag)
    B = lx.MatrixLinearOperator(B_mat, lx.positive_semidefinite_tag)
    return gaussx.KroneckerSum(A, B)


class TestKroneckerSum:
    def test_mv(self, kron_sum):
        n = kron_sum.in_size()
        x = jnp.ones(n)
        result = kron_sum.mv(x)
        expected = kron_sum.as_matrix() @ x
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_mv_random(self, kron_sum):
        key = jax.random.PRNGKey(99)
        x = jax.random.normal(key, (kron_sum.in_size(),))
        result = kron_sum.mv(x)
        expected = kron_sum.as_matrix() @ x
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_as_matrix_shape(self, kron_sum):
        mat = kron_sum.as_matrix()
        assert mat.shape == (12, 12)

    def test_as_matrix_symmetric(self, kron_sum):
        mat = kron_sum.as_matrix()
        assert jnp.allclose(mat, mat.T, atol=1e-7)

    def test_transpose(self, kron_sum):
        mat = kron_sum.as_matrix()
        mat_t = kron_sum.T.as_matrix()
        assert jnp.allclose(mat.T, mat_t, atol=1e-7)

    def test_solve(self, kron_sum):
        key = jax.random.PRNGKey(1)
        b = jax.random.normal(key, (kron_sum.in_size(),))
        x = gaussx.solve(kron_sum, b)
        residual = kron_sum.mv(x) - b
        assert jnp.allclose(residual, 0.0, atol=1e-4)

    def test_logdet(self, kron_sum):
        ld = gaussx.logdet(kron_sum)
        mat = kron_sum.as_matrix()
        expected = jnp.linalg.slogdet(mat)[1]
        assert jnp.allclose(ld, expected, atol=1e-3)

    def test_eigendecompose(self, kron_sum):
        evals, Q = kron_sum.eigendecompose()
        # Q @ diag(evals) @ Q^T should reconstruct the matrix
        reconstructed = Q @ jnp.diag(evals) @ Q.T
        expected = kron_sum.as_matrix()
        assert jnp.allclose(reconstructed, expected, atol=1e-4)

    def test_tags(self, kron_sum):
        assert gaussx.is_kronecker_sum(kron_sum)
        assert lx.is_symmetric(kron_sum)
        assert lx.is_positive_semidefinite(kron_sum)

    def test_in_out_size(self, kron_sum):
        # 3 * 4 = 12
        assert kron_sum.in_size() == 12
        assert kron_sum.out_size() == 12

    def test_non_square_raises(self):
        A = lx.MatrixLinearOperator(jnp.ones((2, 3)))
        B = lx.MatrixLinearOperator(jnp.eye(2))
        with pytest.raises(ValueError, match="square"):
            gaussx.KroneckerSum(A, B)
