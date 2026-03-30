"""Tests for DenseSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._strategies import DenseSolver
from gaussx._testing import tree_allclose


def test_solve_diagonal(getkey):
    ds = DenseSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    v = jr.normal(getkey(), (4,))
    expected = jnp.linalg.solve(op.as_matrix(), v)
    assert tree_allclose(ds.solve(op, v), expected)


def test_solve_block_diag(getkey):
    ds = DenseSolver()
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    bd = BlockDiag(A, B)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(bd.as_matrix(), v)
    assert tree_allclose(ds.solve(bd, v), expected)


def test_solve_kronecker(getkey):
    ds = DenseSolver()
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (6,))
    expected = jnp.linalg.solve(K.as_matrix(), v)
    assert tree_allclose(ds.solve(K, v), expected, rtol=1e-4)


def test_solve_low_rank(getkey):
    ds = DenseSolver()
    d = jnp.abs(jr.normal(getkey(), (5,))) + 1.0
    base = lx.DiagonalLinearOperator(d)
    U = jr.normal(getkey(), (5, 2)) * 0.1
    lr = LowRankUpdate(base, U)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(lr.as_matrix(), v)
    assert tree_allclose(ds.solve(lr, v), expected, rtol=1e-4)


def test_logdet_diagonal(getkey):
    ds = DenseSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    expected = jnp.linalg.slogdet(op.as_matrix())[1]
    assert tree_allclose(ds.logdet(op), expected)


def test_logdet_kronecker(getkey):
    ds = DenseSolver()
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    expected = jnp.linalg.slogdet(K.as_matrix())[1]
    assert tree_allclose(ds.logdet(K), expected, rtol=1e-4)


def test_filter_jit(getkey):
    ds = DenseSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return ds.solve(op, v), ds.logdet(op)

    sol, ld = f(op, v)
    assert tree_allclose(sol, jnp.linalg.solve(op.as_matrix(), v))
    assert tree_allclose(ld, jnp.linalg.slogdet(op.as_matrix())[1])
