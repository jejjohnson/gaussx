"""Tests for gaussx.sqrt with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives import sqrt


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def _make_pd(getkey, n):
    A = jr.normal(getkey(), (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


def test_sqrt_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    S = sqrt(op)
    assert isinstance(S, lx.DiagonalLinearOperator)
    # S @ S should reconstruct A
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, op.as_matrix())


def test_sqrt_block_diag(getkey):
    A = _make_pd(getkey, 2)
    B = _make_pd(getkey, 3)
    bd = BlockDiag(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    S = sqrt(bd)
    assert isinstance(S, BlockDiag)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, bd.as_matrix())


def test_sqrt_kronecker(getkey):
    A = _make_pd(getkey, 2)
    B = _make_pd(getkey, 3)
    K = Kronecker(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    S = sqrt(K)
    assert isinstance(S, Kronecker)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, K.as_matrix(), rtol=1e-4)


def test_sqrt_dense(getkey):
    A = _make_pd(getkey, 4)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    S = sqrt(op)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, op.as_matrix(), rtol=1e-4)
