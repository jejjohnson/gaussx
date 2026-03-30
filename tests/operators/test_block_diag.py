"""Tests for the BlockDiag operator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import BlockDiag
from gaussx._tags import is_block_diagonal
from gaussx._testing import tree_allclose


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_at_least_one_operator():
    with pytest.raises(ValueError, match="at least one"):
        BlockDiag()


def test_basic_construction(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
    bd = BlockDiag(A, B)
    assert bd.in_size() == 7
    assert bd.out_size() == 7
    assert len(bd.operators) == 2


def test_single_operator(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    bd = BlockDiag(A)
    v = jr.normal(getkey(), (3,))
    assert tree_allclose(bd.mv(v), A.mv(v))


# ---------------------------------------------------------------------------
# mv correctness — mv matches dense as_matrix
# ---------------------------------------------------------------------------


def test_mv_diagonal_blocks(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    bd = BlockDiag(A, B)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(bd.mv(v), bd.as_matrix() @ v)


def test_mv_dense_blocks(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    bd = BlockDiag(A, B)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(bd.mv(v), bd.as_matrix() @ v)


def test_mv_three_blocks(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (1,)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    C = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    bd = BlockDiag(A, B, C)
    v = jr.normal(getkey(), (6,))
    assert tree_allclose(bd.mv(v), bd.as_matrix() @ v)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


def test_as_matrix_is_block_diagonal(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (1, 1)))
    bd = BlockDiag(A, B)
    mat = bd.as_matrix()
    # Off-diagonal blocks should be zero
    assert jnp.allclose(mat[:2, 2:], 0.0)
    assert jnp.allclose(mat[2:, :2], 0.0)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def test_transpose(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    bd = BlockDiag(A, B)
    assert tree_allclose(bd.T.as_matrix(), bd.as_matrix().T)


def test_transpose_mv(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    bd = BlockDiag(A, B)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(bd.T.mv(v), bd.as_matrix().T @ v)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_has_block_diagonal_tag(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    bd = BlockDiag(A)
    assert is_block_diagonal(bd) is True


def test_symmetric_when_all_blocks_symmetric(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    bd = BlockDiag(A, B)
    assert lx.is_symmetric(bd) is True


def test_not_symmetric_when_block_not_symmetric(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    bd = BlockDiag(A)
    assert lx.is_symmetric(bd) is False


def test_psd_when_all_blocks_psd(getkey):
    mat = jr.normal(getkey(), (2, 2))
    A = lx.MatrixLinearOperator(mat.T @ mat, lx.positive_semidefinite_tag)
    B = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(jnp.abs(jr.normal(getkey(), (3,)))),
        lx.positive_semidefinite_tag,
    )
    bd = BlockDiag(A, B)
    assert lx.is_positive_semidefinite(bd) is True


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


def test_filter_jit_mv(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    bd = BlockDiag(A, B)
    v = jr.normal(getkey(), (5,))

    @eqx.filter_jit
    def f(op, v):
        return op.mv(v)

    assert tree_allclose(f(bd, v), bd.as_matrix() @ v)


def test_vmap_mv(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    bd = BlockDiag(A, B)
    vs = jr.normal(getkey(), (4, 5))
    results = jax.vmap(bd.mv)(vs)
    assert results.shape == (4, 5)
    assert tree_allclose(results[0], bd.as_matrix() @ vs[0])
