"""Tests for gaussx.logdet with structural dispatch."""

from __future__ import annotations

import re

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker, KroneckerSum, LowRankUpdate
from gaussx._primitives import logdet, solve
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


def _kronecker_sum_factors(key, *tags):
    ka, kb = jr.split(key)
    A = jr.normal(ka, (3, 3)) + 3 * jnp.eye(3)
    B = jr.normal(kb, (4, 4)) + 4 * jnp.eye(4)
    if tags:
        A, B = A + A.T, B + B.T
    return lx.MatrixLinearOperator(A, tags), lx.MatrixLinearOperator(B, tags)


def test_logdet_kronecker_sum_untagged_nonsymmetric(getkey):
    """``eigh`` reads one triangle, so untagged factors take ``eigvals`` (gh-308)."""
    op = KroneckerSum(*_kronecker_sum_factors(getkey()))
    assert tree_allclose(logdet(op), dense_logdet(op))
    # logdet and solve agree on the operator they are given.
    v = jnp.arange(12.0)
    assert tree_allclose(solve(op, v), jnp.linalg.solve(op.as_matrix(), v))


def test_logdet_kronecker_sum_symmetric_keeps_eigh(getkey):
    op = KroneckerSum(*_kronecker_sum_factors(getkey(), lx.symmetric_tag))
    assert tree_allclose(logdet(op), dense_logdet(op))
    jaxpr = str(jax.make_jaxpr(logdet)(op))
    assert re.search(r"\beigh\[", jaxpr)
    assert not re.search(r"\beig\[", jaxpr)
