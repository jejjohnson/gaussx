"""Tests for SVDLowRankUpdate operator."""

import jax.numpy as jnp
import lineax as lx
import pytest

import gaussx


@pytest.fixture()
def psd_operator():
    """Symmetric PSD SVDLowRankUpdate: diag + U S U^T."""
    n, k = 6, 3
    diag = jnp.ones(n) * 2.0
    key = jax.random.PRNGKey(42)
    U, _, _ = jnp.linalg.svd(jax.random.normal(key, (n, k)), full_matrices=False)
    S = jnp.array([3.0, 2.0, 1.0])
    base = lx.DiagonalLinearOperator(diag)
    base = lx.TaggedLinearOperator(base, lx.positive_semidefinite_tag)
    return gaussx.SVDLowRankUpdate(base, U, S)


@pytest.fixture()
def nonsym_operator():
    """Non-symmetric SVDLowRankUpdate: diag + U S V^T."""
    n, k = 5, 2
    diag = jnp.ones(n) * 3.0
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    U, _, _ = jnp.linalg.svd(jax.random.normal(k1, (n, k)), full_matrices=False)
    V, _, _ = jnp.linalg.svd(jax.random.normal(k2, (n, k)), full_matrices=False)
    S = jnp.array([2.0, 1.0])
    base = lx.DiagonalLinearOperator(diag)
    return gaussx.SVDLowRankUpdate(base, U, S, V)


import jax


class TestSVDLowRankUpdate:
    def test_mv(self, psd_operator):
        x = jnp.ones(6)
        result = psd_operator.mv(x)
        expected = psd_operator.as_matrix() @ x
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_as_matrix(self, psd_operator):
        mat = psd_operator.as_matrix()
        assert mat.shape == (6, 6)
        # Should be symmetric
        assert jnp.allclose(mat, mat.T, atol=1e-7)

    def test_transpose(self, nonsym_operator):
        mat = nonsym_operator.as_matrix()
        mat_t = nonsym_operator.T.as_matrix()
        assert jnp.allclose(mat.T, mat_t, atol=1e-7)

    def test_rank(self, psd_operator):
        assert psd_operator.rank == 3

    def test_solve(self, psd_operator):
        b = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        x = gaussx.solve(psd_operator, b)
        # Verify: A x = b
        residual = psd_operator.mv(x) - b
        assert jnp.allclose(residual, 0.0, atol=1e-5)

    def test_logdet(self, psd_operator):
        ld = gaussx.logdet(psd_operator)
        mat = psd_operator.as_matrix()
        expected = jnp.linalg.slogdet(mat)[1]
        assert jnp.allclose(ld, expected, atol=1e-4)

    def test_tags_symmetric(self, psd_operator):
        assert lx.is_symmetric(psd_operator)
        assert gaussx.is_low_rank(psd_operator)

    def test_tags_nonsym(self, nonsym_operator):
        assert not lx.is_symmetric(nonsym_operator)
        assert gaussx.is_low_rank(nonsym_operator)

    def test_in_out_size(self, psd_operator):
        assert psd_operator.in_size() == 6
        assert psd_operator.out_size() == 6
