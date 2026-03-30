"""Tests for the Kronecker operator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import Kronecker
from gaussx._tags import is_kronecker


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_at_least_two_operators(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    with pytest.raises(ValueError, match="at least two"):
        Kronecker(A)


def test_basic_construction(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    K = Kronecker(A, B)
    assert K.in_size() == 6
    assert K.out_size() == 6
    assert len(K.operators) == 2


# ---------------------------------------------------------------------------
# mv correctness — mv matches dense as_matrix
# ---------------------------------------------------------------------------


def test_mv_diagonal_factors(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (6,))
    assert tree_allclose(K.mv(v), K.as_matrix() @ v)


def test_mv_dense_factors(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (6,))
    assert tree_allclose(K.mv(v), K.as_matrix() @ v)


def test_mv_three_factors(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    C = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    K = Kronecker(A, B, C)
    v = jr.normal(getkey(), (12,))
    assert tree_allclose(K.mv(v), K.as_matrix() @ v)


def test_mv_random(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (12,))
    assert tree_allclose(K.mv(v), K.as_matrix() @ v)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


def test_as_matrix_matches_jnp_kron(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    expected = jnp.kron(A.as_matrix(), B.as_matrix())
    assert tree_allclose(K.as_matrix(), expected)


def test_as_matrix_three_factors(getkey):
    A_mat = jr.normal(getkey(), (2, 2))
    B_mat = jr.normal(getkey(), (2, 2))
    C_mat = jr.normal(getkey(), (2, 2))
    K = Kronecker(
        lx.MatrixLinearOperator(A_mat),
        lx.MatrixLinearOperator(B_mat),
        lx.MatrixLinearOperator(C_mat),
    )
    expected = jnp.kron(jnp.kron(A_mat, B_mat), C_mat)
    assert tree_allclose(K.as_matrix(), expected)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def test_transpose(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    assert tree_allclose(K.T.as_matrix(), K.as_matrix().T)


def test_transpose_mv(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (6,))
    assert tree_allclose(K.T.mv(v), K.as_matrix().T @ v)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_has_kronecker_tag(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    K = Kronecker(A, B)
    assert is_kronecker(K) is True


def test_symmetric_when_all_factors_symmetric(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    K = Kronecker(A, B)
    assert lx.is_symmetric(K) is True


def test_not_symmetric_when_factor_not_symmetric(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    K = Kronecker(A, B)
    assert lx.is_symmetric(K) is False


def test_psd_when_all_factors_psd(getkey):
    m1 = jr.normal(getkey(), (2, 2))
    A = lx.MatrixLinearOperator(m1.T @ m1, lx.positive_semidefinite_tag)
    m2 = jr.normal(getkey(), (3, 3))
    B = lx.MatrixLinearOperator(m2.T @ m2, lx.positive_semidefinite_tag)
    K = Kronecker(A, B)
    assert lx.is_positive_semidefinite(K) is True


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


def test_filter_jit_mv(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    K = Kronecker(A, B)
    v = jr.normal(getkey(), (6,))

    @eqx.filter_jit
    def f(op, v):
        return op.mv(v)

    assert tree_allclose(f(K, v), K.as_matrix() @ v)


def test_vmap_mv(getkey):
    A = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    K = Kronecker(A, B)
    vs = jr.normal(getkey(), (5, 6))
    results = jax.vmap(K.mv)(vs)
    assert results.shape == (5, 6)
    assert tree_allclose(results[0], K.as_matrix() @ vs[0])
