"""Tests for BlockTriDiag and LowerBlockTriDiag operators."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import gaussx


def _make_spd_block_tridiag(N, d, key):
    """Create a random SPD block-tridiagonal matrix.

    Makes the diagonal blocks dominant enough to ensure positive definiteness.
    """
    k1, k2 = jax.random.split(key)
    # Sub-diagonal blocks: random
    sub = 0.3 * jax.random.normal(k1, (N - 1, d, d))
    # Diagonal blocks: SPD with diagonal dominance
    raw = jax.random.normal(k2, (N, d, d))
    diags = jax.vmap(lambda M: M @ M.T)(raw) + 5.0 * jnp.eye(d)[None]
    # Make symmetric
    diags = 0.5 * (diags + jnp.swapaxes(diags, -2, -1))
    return diags, sub


@pytest.fixture()
def block_tridiag():
    """SPD block-tridiagonal operator."""
    N, d = 5, 3
    diags, sub = _make_spd_block_tridiag(N, d, jax.random.PRNGKey(42))
    return gaussx.BlockTriDiag(diags, sub)


class TestBlockTriDiag:
    def test_mv(self, block_tridiag):
        n = block_tridiag.in_size()
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (n,))
        result = block_tridiag.mv(x)
        expected = block_tridiag.as_matrix() @ x
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_as_matrix_shape(self, block_tridiag):
        mat = block_tridiag.as_matrix()
        assert mat.shape == (15, 15)

    def test_as_matrix_band_structure(self, block_tridiag):
        """Verify entries outside the band are zero."""
        mat = block_tridiag.as_matrix()
        d = block_tridiag._block_size
        N = block_tridiag._num_blocks
        for i in range(N):
            for j in range(N):
                if abs(i - j) > 1:
                    block = mat[i * d : (i + 1) * d, j * d : (j + 1) * d]
                    assert jnp.allclose(block, 0.0, atol=1e-10)

    def test_transpose(self, block_tridiag):
        mat = block_tridiag.as_matrix()
        mat_t = block_tridiag.T.as_matrix()
        assert jnp.allclose(mat.T, mat_t, atol=1e-7)

    def test_cholesky(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        assert isinstance(L, gaussx.LowerBlockTriDiag)
        # L L^T should reconstruct the original
        reconstructed = L.as_matrix() @ L.as_matrix().T
        expected = block_tridiag.as_matrix()
        assert jnp.allclose(reconstructed, expected, atol=1e-4)

    def test_solve(self, block_tridiag):
        key = jax.random.PRNGKey(1)
        b = jax.random.normal(key, (block_tridiag.in_size(),))
        x = gaussx.solve(block_tridiag, b)
        residual = block_tridiag.mv(x) - b
        assert jnp.allclose(residual, 0.0, atol=1e-4)

    def test_logdet(self, block_tridiag):
        ld = gaussx.logdet(block_tridiag)
        mat = block_tridiag.as_matrix()
        expected = jnp.linalg.slogdet(mat)[1]
        assert jnp.allclose(ld, expected, atol=1e-3)

    def test_diag(self, block_tridiag):
        d = gaussx.diag(block_tridiag)
        expected = jnp.diag(block_tridiag.as_matrix())
        assert jnp.allclose(d, expected, atol=1e-7)

    def test_trace(self, block_tridiag):
        tr = gaussx.trace(block_tridiag)
        expected = jnp.trace(block_tridiag.as_matrix())
        assert jnp.allclose(tr, expected, atol=1e-5)

    def test_add(self, block_tridiag):
        result = block_tridiag.add(block_tridiag)
        mat_sum = result.as_matrix()
        expected = 2.0 * block_tridiag.as_matrix()
        assert jnp.allclose(mat_sum, expected, atol=1e-7)

    def test_scalar_mul_requires_true_scalar(self, block_tridiag):
        with pytest.raises(TypeError, match="scalar"):
            block_tridiag * jnp.array([1.0, 2.0])

    def test_tags(self, block_tridiag):
        assert gaussx.is_block_tridiagonal(block_tridiag)

    def test_reports_symmetric(self, block_tridiag):
        assert lx.is_symmetric(block_tridiag)

    def test_in_out_size(self, block_tridiag):
        assert block_tridiag.in_size() == 15
        assert block_tridiag.out_size() == 15

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="3 dimensions"):
            gaussx.BlockTriDiag(jnp.ones((3, 3)), jnp.ones((2, 3, 3)))

        with pytest.raises(ValueError, match="square"):
            gaussx.BlockTriDiag(jnp.ones((3, 2, 3)), jnp.ones((2, 2, 3)))

        with pytest.raises(ValueError, match="2 blocks"):
            gaussx.BlockTriDiag(jnp.ones((3, 2, 2)), jnp.ones((3, 2, 2)))


class TestLowerBlockTriDiag:
    def test_mv(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        n = L.in_size()
        key = jax.random.PRNGKey(5)
        x = jax.random.normal(key, (n,))
        result = L.mv(x)
        expected = L.as_matrix() @ x
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_transpose_gives_upper(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        U = L.T
        assert isinstance(U, gaussx.UpperBlockTriDiag)
        assert jnp.allclose(U.as_matrix(), L.as_matrix().T, atol=1e-7)

    def test_logdet(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        ld = gaussx.logdet(L)
        expected = jnp.linalg.slogdet(L.as_matrix())[1]
        assert jnp.allclose(ld, expected, atol=1e-4)

    def test_solve_forward(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        key = jax.random.PRNGKey(3)
        b = jax.random.normal(key, (L.in_size(),))
        x = gaussx.solve(L, b)
        residual = L.mv(x) - b
        assert jnp.allclose(residual, 0.0, atol=1e-4)

    def test_solve_backward(self, block_tridiag):
        L = gaussx.cholesky(block_tridiag)
        U = L.T
        key = jax.random.PRNGKey(4)
        b = jax.random.normal(key, (U.in_size(),))
        x = gaussx.solve(U, b)
        residual = U.mv(x) - b
        assert jnp.allclose(residual, 0.0, atol=1e-4)
