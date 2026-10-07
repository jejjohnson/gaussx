"""Tests for gaussx.cholesky with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._einx import einsum
from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives import cholesky
from gaussx._testing import random_pd_matrix, tree_allclose


def test_cholesky_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    L = cholesky(op)
    # L @ L^T should reconstruct A
    assert isinstance(L, lx.DiagonalLinearOperator)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, op.as_matrix())


def test_cholesky_block_diag(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    bd = BlockDiag(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    L = cholesky(bd)
    assert isinstance(L, BlockDiag)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, bd.as_matrix())


def test_cholesky_kronecker(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    K = Kronecker(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    L = cholesky(K)
    assert isinstance(L, Kronecker)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, K.as_matrix(), rtol=1e-4)


def test_cholesky_dense(getkey):
    A = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    L = cholesky(op)
    assert isinstance(L, lx.MatrixLinearOperator)
    assert lx.is_lower_triangular(L)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, op.as_matrix())


def test_cholesky_filter_jit(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)

    @eqx.filter_jit
    def f(op):
        return cholesky(op)

    L = f(op)
    reconstructed = L.as_matrix() @ L.as_matrix().T
    assert tree_allclose(reconstructed, op.as_matrix())


_SCALINGS = [
    pytest.param(lambda op, c: c * op, lambda c: c, id="mul"),
    pytest.param(lambda op, c: op / c, lambda c: 1 / c, id="div"),
]


def _structured(key, cls):
    k1, k2 = jr.split(key)
    a = lx.MatrixLinearOperator(random_pd_matrix(k1, 2), lx.positive_semidefinite_tag)
    b = lx.MatrixLinearOperator(random_pd_matrix(k2, 3), lx.positive_semidefinite_tag)
    return cls(a, b)


@pytest.mark.parametrize("cls", [Kronecker, BlockDiag])
@pytest.mark.parametrize(("wrap", "factor"), _SCALINGS)
def test_cholesky_scalar_multiple_keeps_structure(
    getkey, monkeypatch, cls, wrap, factor
):
    """``chol(c A) = √c chol(A)`` without materialising ``A`` (gh-326)."""
    op = _structured(getkey(), cls)
    expected = factor(2.0) * op.as_matrix()

    def _forbidden(self):
        raise AssertionError(f"{cls.__name__}.as_matrix called")

    monkeypatch.setattr(cls, "as_matrix", _forbidden)
    L = cholesky(wrap(op, 2.0))
    monkeypatch.undo()
    assert isinstance(L, cls)
    Lm = L.as_matrix()
    assert tree_allclose(einsum(Lm, Lm, "i k, j k -> i j"), expected)


def test_cholesky_traced_scalar_has_finite_gradient(getkey):
    op = _structured(getkey(), Kronecker)

    def f(c):
        return jnp.sum(cholesky(c * op).as_matrix())

    value, grad = jax.value_and_grad(eqx.filter_jit(f))(jnp.asarray(2.0))
    assert tree_allclose(value, f(2.0))
    assert jnp.isfinite(grad)


def test_cholesky_negated_operator_raises(getkey):
    op = _structured(getkey(), Kronecker)
    with pytest.raises(ValueError, match="positive semi-definite"):
        cholesky(-op)
    with pytest.raises(ValueError, match="positive semi-definite"):
        cholesky(-2.0 * op)
    L = cholesky(lx.NegLinearOperator(-op))
    assert isinstance(L, Kronecker)


def test_cholesky_negated_nsd_operator_stays_dense(getkey):
    S = random_pd_matrix(getkey(), 3)
    op = -lx.MatrixLinearOperator(-S, lx.negative_semidefinite_tag)
    Lm = cholesky(op).as_matrix()
    assert tree_allclose(einsum(Lm, Lm, "i k, j k -> i j"), S)
