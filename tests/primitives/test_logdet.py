"""Tests for gaussx.logdet with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._primitives import logdet
from gaussx._testing import dense_logdet, tree_allclose


def test_logdet_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    assert tree_allclose(logdet(op), dense_logdet(op))


def test_logdet_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    bd = BlockDiag(A, B)
    assert tree_allclose(logdet(bd), dense_logdet(bd))


def test_logdet_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    assert tree_allclose(logdet(K), dense_logdet(K), rtol=1e-4)


def test_logdet_kronecker_three_factors(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.DiagonalLinearOperator(jnp.abs(jr.normal(getkey(), (2,))) + 0.5)
    C = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    K = Kronecker(A, B, C)
    assert tree_allclose(logdet(K), dense_logdet(K), rtol=1e-4)


def test_logdet_low_rank(getkey):
    d = jnp.abs(jr.normal(getkey(), (5,))) + 1.0
    base = lx.DiagonalLinearOperator(d)
    U = jr.normal(getkey(), (5, 2)) * 0.3
    lr = LowRankUpdate(base, U)
    assert tree_allclose(logdet(lr), dense_logdet(lr), rtol=1e-4)


def test_logdet_dense_fallback(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    assert tree_allclose(logdet(op), dense_logdet(op))


def test_logdet_filter_jit(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)

    @eqx.filter_jit
    def f(op):
        return logdet(op)

    assert tree_allclose(f(op), dense_logdet(op))
