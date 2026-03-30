"""Tests for gaussx.diag with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives import diag


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def _dense_diag(op):
    return jnp.diag(op.as_matrix())


def test_diag_diagonal(getkey):
    d = jr.normal(getkey(), (4,))
    op = lx.DiagonalLinearOperator(d)
    assert tree_allclose(diag(op), d)


def test_diag_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    bd = BlockDiag(A, B)
    assert tree_allclose(diag(bd), _dense_diag(bd))


def test_diag_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    assert tree_allclose(diag(K), _dense_diag(K))


def test_diag_dense_fallback(getkey):
    mat = jr.normal(getkey(), (3, 3))
    op = lx.MatrixLinearOperator(mat)
    assert tree_allclose(diag(op), jnp.diag(mat))
