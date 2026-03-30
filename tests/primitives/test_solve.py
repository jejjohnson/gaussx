"""Tests for gaussx.solve with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._primitives import solve
from gaussx._testing import dense_solve, tree_allclose


class LazyDiagonal(lx.DiagonalLinearOperator):
    def as_matrix(self):
        raise NotImplementedError("dense materialization unavailable")


def test_solve_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(solve(op, v), dense_solve(op, v))


def test_solve_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 2 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    bd = BlockDiag(A, B)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(solve(bd, v), dense_solve(bd, v))


def test_solve_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (6,))
    assert tree_allclose(solve(K, v), dense_solve(K, v), rtol=1e-4)


def test_solve_kronecker_lazy_factors(getkey):
    a_diag = jnp.abs(jr.normal(getkey(), (2,))) + 0.5
    b_diag = jnp.abs(jr.normal(getkey(), (3,))) + 0.5
    K = Kronecker(LazyDiagonal(a_diag), LazyDiagonal(b_diag))
    v = jr.normal(getkey(), (6,))
    expected = jnp.linalg.solve(jnp.kron(jnp.diag(a_diag), jnp.diag(b_diag)), v)
    assert tree_allclose(solve(K, v), expected, rtol=1e-4)


def test_solve_low_rank(getkey):
    d = jnp.abs(jr.normal(getkey(), (5,))) + 1.0
    base = lx.DiagonalLinearOperator(d)
    U = jr.normal(getkey(), (5, 2)) * 0.1
    lr = LowRankUpdate(base, U)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(solve(lr, v), dense_solve(lr, v), rtol=1e-4)


def test_solve_dense_fallback(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    v = jr.normal(getkey(), (3,))
    assert tree_allclose(solve(op, v), dense_solve(op, v))


def test_solve_filter_jit(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return solve(op, v)

    assert tree_allclose(f(op, v), dense_solve(op, v))
