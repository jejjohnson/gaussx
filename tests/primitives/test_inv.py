"""Tests for gaussx.inv with structural dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives import inv
from gaussx._primitives._inv import InverseOperator
from gaussx._testing import dense_inv, tree_allclose


def test_inv_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    inv_op = inv(op)
    assert isinstance(inv_op, lx.DiagonalLinearOperator)
    assert tree_allclose(inv_op.as_matrix(), dense_inv(op))


def test_inv_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    bd = BlockDiag(A, B)
    inv_op = inv(bd)
    assert isinstance(inv_op, BlockDiag)
    assert tree_allclose(inv_op.as_matrix(), dense_inv(bd), rtol=1e-4)


def test_inv_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    inv_op = inv(K)
    assert isinstance(inv_op, Kronecker)
    assert tree_allclose(inv_op.as_matrix(), dense_inv(K), rtol=1e-4)


def test_inv_dense_returns_inverse_operator(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    inv_op = inv(op)
    assert isinstance(inv_op, InverseOperator)


def test_inv_dense_mv(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    inv_op = inv(op)
    v = jr.normal(getkey(), (3,))
    # inv_op.mv(v) should equal A^{-1} v
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(inv_op.mv(v), expected)


def test_inv_dense_as_matrix(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    inv_op = inv(op)
    assert tree_allclose(inv_op.as_matrix(), dense_inv(op))


def test_inv_roundtrip(getkey):
    """inv(A) @ A @ v should give back v."""
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    inv_op = inv(op)
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(inv_op.mv(op.mv(v)), v)
