"""Tests for gaussx.diag with structural dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import (
    BlockDiag,
    DiagonalizedOperator,
    Kronecker,
    Toeplitz,
    circulant,
    circulant_from_symbol,
)
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


def _forbid_as_matrix(monkeypatch, cls):
    def _forbidden(self):
        raise AssertionError(f"{cls.__name__}.as_matrix called")

    monkeypatch.setattr(cls, "as_matrix", _forbidden)


def test_diag_toeplitz_is_constant(monkeypatch):
    """A symmetric Toeplitz matrix has the constant diagonal c[0] (gh-373)."""
    op = Toeplitz(jnp.array([2.0, 0.5, 0.25, 0.125]))
    expected = jnp.diag(op.as_matrix())
    _forbid_as_matrix(monkeypatch, Toeplitz)
    result = diag(op)
    monkeypatch.undo()
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: circulant(jnp.array([3.0, 1.0, 0.5, 1.0]), symmetric=True),
            id="circulant_even",
        ),
        pytest.param(
            lambda: circulant(jnp.array([3.0, 1.0, 0.5, 0.2, -0.4])), id="circulant"
        ),
        pytest.param(
            lambda: circulant_from_symbol(jnp.fft.fftn(jr.normal(jr.key(0), (3, 4)))),
            id="symbol_2d",
        ),
    ],
)
def test_diag_fft_diagonalised_is_constant(monkeypatch, build):
    """``F⁻¹ diag(λ) F`` has the constant diagonal ``mean(λ)`` (gh-373)."""
    op = build()
    expected = jnp.diag(op.as_matrix())
    _forbid_as_matrix(monkeypatch, DiagonalizedOperator)
    result = diag(op)
    monkeypatch.undo()
    assert tree_allclose(result, expected)
