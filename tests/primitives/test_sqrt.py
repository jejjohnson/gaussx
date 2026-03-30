"""Tests for gaussx.sqrt with structural dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives import sqrt
from gaussx._testing import random_pd_matrix, tree_allclose


def test_sqrt_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    S = sqrt(op)
    assert isinstance(S, lx.DiagonalLinearOperator)
    # S @ S should reconstruct A
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, op.as_matrix())


def test_sqrt_block_diag(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    bd = BlockDiag(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    S = sqrt(bd)
    assert isinstance(S, BlockDiag)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, bd.as_matrix())


def test_sqrt_kronecker(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    K = Kronecker(
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
    )
    S = sqrt(K)
    assert isinstance(S, Kronecker)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, K.as_matrix(), rtol=1e-4)


def test_sqrt_dense(getkey):
    A = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    S = sqrt(op)
    reconstructed = S.as_matrix() @ S.as_matrix()
    assert tree_allclose(reconstructed, op.as_matrix(), rtol=1e-4)
