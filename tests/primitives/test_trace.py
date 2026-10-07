"""Tests for gaussx.trace with structural dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import (
    BlockDiag,
    Kronecker,
    LowerBlockTriDiag,
    Toeplitz,
    UpperBlockTriDiag,
)
from gaussx._primitives import trace
from gaussx._testing import dense_trace, tree_allclose


def test_trace_diagonal(getkey):
    d = jr.normal(getkey(), (4,))
    op = lx.DiagonalLinearOperator(d)
    assert tree_allclose(trace(op), jnp.sum(d))


def test_trace_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    bd = BlockDiag(A, B)
    assert tree_allclose(trace(bd), dense_trace(bd))


def test_trace_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    assert tree_allclose(trace(K), dense_trace(K))


def test_trace_dense_fallback(getkey):
    mat = jr.normal(getkey(), (3, 3))
    op = lx.MatrixLinearOperator(mat)
    assert tree_allclose(trace(op), jnp.trace(mat))


def test_trace_toeplitz_is_n_c0(monkeypatch):
    """``trace(Toeplitz) = n · c[0]`` without materialising (gh-373)."""
    op = Toeplitz(jnp.array([2.0, 0.5, 0.25, 0.125]))
    expected = jnp.trace(op.as_matrix())

    def _forbidden(self):
        raise AssertionError("Toeplitz.as_matrix called")

    monkeypatch.setattr(Toeplitz, "as_matrix", _forbidden)
    result = trace(op)
    monkeypatch.undo()
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("cls", [LowerBlockTriDiag, UpperBlockTriDiag])
def test_trace_block_bidiagonal_is_structured(monkeypatch, cls):
    """``trace`` covers the factors ``diag`` already handles (gh-391)."""
    k1, k2 = jr.split(jr.key(0))
    op = cls(jr.normal(k1, (3, 2, 2)), jr.normal(k2, (2, 2, 2)))
    expected = jnp.trace(op.as_matrix())

    def _forbidden(self):
        raise AssertionError(f"{cls.__name__}.as_matrix called")

    monkeypatch.setattr(cls, "as_matrix", _forbidden)
    result = trace(op)
    monkeypatch.undo()
    assert tree_allclose(result, expected)
