"""Tests for gaussx.trace with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives import trace


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def _dense_trace(op):
    return jnp.trace(op.as_matrix())


def test_trace_diagonal(getkey):
    d = jr.normal(getkey(), (4,))
    op = lx.DiagonalLinearOperator(d)
    assert tree_allclose(trace(op), jnp.sum(d))


def test_trace_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    bd = BlockDiag(A, B)
    assert tree_allclose(trace(bd), _dense_trace(bd))


def test_trace_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    assert tree_allclose(trace(K), _dense_trace(K))


def test_trace_dense_fallback(getkey):
    mat = jr.normal(getkey(), (3, 3))
    op = lx.MatrixLinearOperator(mat)
    assert tree_allclose(trace(op), jnp.trace(mat))
