"""Tests for gaussx._testing helper utilities."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._testing import (
    random_block_diag_pd,
    random_kronecker_pd,
    random_low_rank_update,
    random_pd_matrix,
    random_pd_operator,
    tree_allclose,
)


def test_tree_allclose_true():
    x = jnp.array([1.0, 2.0])
    assert tree_allclose(x, x)


def test_tree_allclose_false():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([1.0, 3.0])
    assert not tree_allclose(x, y)


def test_random_pd_matrix_shape_and_pd(getkey):
    mat = random_pd_matrix(getkey(), 5)
    assert mat.shape == (5, 5)
    assert mat.dtype == jnp.float64
    # Positive definite: all eigenvalues > 0
    eigs = jnp.linalg.eigvalsh(mat)
    assert jnp.all(eigs > 0)


def test_random_pd_operator(getkey):
    op = random_pd_operator(getkey(), 4)
    assert isinstance(op, lx.MatrixLinearOperator)
    assert op.in_size() == 4
    assert lx.is_positive_semidefinite(op)


def test_random_kronecker_pd(getkey):
    K = random_kronecker_pd(getkey(), (3, 4))
    assert isinstance(K, Kronecker)
    assert K.in_size() == 12
    # All factors should be PSD
    for op in K.operators:
        assert lx.is_positive_semidefinite(op)


def test_random_block_diag_pd(getkey):
    BD = random_block_diag_pd(getkey(), (2, 3, 4))
    assert isinstance(BD, BlockDiag)
    assert BD.in_size() == 9
    for op in BD.operators:
        assert lx.is_positive_semidefinite(op)


def test_random_low_rank_update(getkey):
    lr = random_low_rank_update(getkey(), 10, 3)
    assert isinstance(lr, LowRankUpdate)
    assert lr.in_size() == 10
    assert lr.rank == 3
