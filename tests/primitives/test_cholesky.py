"""Tests for gaussx.cholesky with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives import cholesky


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def _make_pd(getkey, n):
    """Generate a positive-definite matrix."""
    A = jr.normal(getkey(), (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


def test_cholesky_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    L = cholesky(op)
    # L @ L^T should reconstruct A
    assert isinstance(L, lx.DiagonalLinearOperator)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, op.as_matrix())


def test_cholesky_block_diag(getkey):
    A = _make_pd(getkey, 2)
    B = _make_pd(getkey, 3)
    bd = BlockDiag(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    L = cholesky(bd)
    assert isinstance(L, BlockDiag)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, bd.as_matrix())


def test_cholesky_kronecker(getkey):
    A = _make_pd(getkey, 2)
    B = _make_pd(getkey, 3)
    K = Kronecker(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    L = cholesky(K)
    assert isinstance(L, Kronecker)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, K.as_matrix(), rtol=1e-4)


def test_cholesky_dense(getkey):
    A = _make_pd(getkey, 4)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    L = cholesky(op)
    assert isinstance(L, lx.MatrixLinearOperator)
    assert lx.is_lower_triangular(L)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, op.as_matrix())


def test_cholesky_filter_jit(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)

    @eqx.filter_jit
    def f(op):
        return cholesky(op)

    L = f(op)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, op.as_matrix())
