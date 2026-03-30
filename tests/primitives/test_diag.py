"""Tests for gaussx.diag with structural dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives import diag
from gaussx._testing import dense_diag, tree_allclose


def test_diag_diagonal(getkey):
    d = jr.normal(getkey(), (4,))
    op = lx.DiagonalLinearOperator(d)
    assert tree_allclose(diag(op), d)


def test_diag_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    bd = BlockDiag(A, B)
    assert tree_allclose(diag(bd), dense_diag(bd))


def test_diag_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    assert tree_allclose(diag(K), dense_diag(K))


def test_diag_dense_fallback(getkey):
    mat = jr.normal(getkey(), (3, 3))
    op = lx.MatrixLinearOperator(mat)
    assert tree_allclose(diag(op), jnp.diag(mat))
