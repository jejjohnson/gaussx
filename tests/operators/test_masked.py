"""Tests for the MaskedOperator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import MaskedOperator
from gaussx._testing import tree_allclose


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        row_mask = jnp.array([True, True, False, True, False])
        col_mask = jnp.array([True, False, True, False, True])
        M = MaskedOperator(A, row_mask, col_mask)
        assert M.out_size() == 3
        assert M.in_size() == 3

    def test_rejects_non_square_base(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 5)))
        mask = jnp.array([True, True, False])
        with pytest.raises(ValueError, match="square"):
            MaskedOperator(A, mask, mask)

    def test_rejects_wrong_mask_shape(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        mask_bad = jnp.array([True, True, False])
        mask_ok = jnp.array([True, True, False, True])
        with pytest.raises(ValueError, match="shape"):
            MaskedOperator(A, mask_bad, mask_ok)

    def test_symmetric_masks(self, getkey):
        """Same row/col mask gives a square sub-matrix."""
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        mask = jnp.array([True, True, False, True, False])
        M = MaskedOperator(A, mask, mask)
        assert M.in_size() == 3
        assert M.out_size() == 3

    def test_all_true_mask(self, getkey):
        """All-true mask gives full operator."""
        A_mat = jr.normal(getkey(), (4, 4))
        A = lx.MatrixLinearOperator(A_mat)
        mask = jnp.ones(4, dtype=bool)
        M = MaskedOperator(A, mask, mask)
        assert M.in_size() == 4
        assert M.out_size() == 4


# ---------------------------------------------------------------------------
# mv correctness
# ---------------------------------------------------------------------------


class TestMv:
    def test_mv_matches_dense(self, getkey):
        A_mat = jr.normal(getkey(), (5, 5))
        A = lx.MatrixLinearOperator(A_mat)
        row_mask = jnp.array([True, True, False, True, False])
        col_mask = jnp.array([True, False, True, False, True])
        M = MaskedOperator(A, row_mask, col_mask)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(M.mv(v), M.as_matrix() @ v)

    def test_mv_symmetric_mask(self, getkey):
        A_mat = jr.normal(getkey(), (6, 6))
        A = lx.MatrixLinearOperator(A_mat)
        mask = jnp.array([True, False, True, True, False, True])
        M = MaskedOperator(A, mask, mask)
        v = jr.normal(getkey(), (4,))
        # Manual: A[mask][:, mask] @ v
        idx = jnp.where(mask)[0]
        expected = A_mat[jnp.ix_(idx, idx)] @ v
        assert tree_allclose(M.mv(v), expected)

    def test_mv_all_true(self, getkey):
        """All-true mask: mv matches base operator."""
        A_mat = jr.normal(getkey(), (4, 4))
        A = lx.MatrixLinearOperator(A_mat)
        mask = jnp.ones(4, dtype=bool)
        M = MaskedOperator(A, mask, mask)
        v = jr.normal(getkey(), (4,))
        assert tree_allclose(M.mv(v), A_mat @ v)

    def test_mv_with_diagonal_base(self, getkey):
        diag_vals = jr.normal(getkey(), (5,))
        A = lx.DiagonalLinearOperator(diag_vals)
        mask = jnp.array([True, False, True, True, False])
        M = MaskedOperator(A, mask, mask)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(M.mv(v), M.as_matrix() @ v)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


class TestAsMatrix:
    def test_as_matrix_subselects(self, getkey):
        A_mat = jr.normal(getkey(), (5, 5))
        A = lx.MatrixLinearOperator(A_mat)
        row_mask = jnp.array([True, True, False, False, True])
        col_mask = jnp.array([False, True, True, False, True])
        M = MaskedOperator(A, row_mask, col_mask)
        row_idx = jnp.where(row_mask)[0]
        col_idx = jnp.where(col_mask)[0]
        expected = A_mat[jnp.ix_(row_idx, col_idx)]
        assert tree_allclose(M.as_matrix(), expected)

    def test_symmetric_mask_of_symmetric_base(self, getkey):
        """Symmetric mask of symmetric base gives symmetric sub-matrix."""
        m = jr.normal(getkey(), (5, 5))
        A_mat = m + m.T
        A = lx.MatrixLinearOperator(A_mat, lx.symmetric_tag)
        mask = jnp.array([True, False, True, True, False])
        M = MaskedOperator(A, mask, mask)
        sub = M.as_matrix()
        assert tree_allclose(sub, sub.T)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_transpose(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        row_mask = jnp.array([True, True, False, True, False])
        col_mask = jnp.array([True, False, True, False, True])
        M = MaskedOperator(A, row_mask, col_mask)
        assert tree_allclose(M.T.as_matrix(), M.as_matrix().T)

    def test_transpose_mv(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        row_mask = jnp.array([True, True, False, True, False])
        col_mask = jnp.array([True, False, True, False, True])
        M = MaskedOperator(A, row_mask, col_mask)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(M.T.mv(v), M.as_matrix().T @ v)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_not_symmetric_by_default(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        mask = jnp.array([True, True, False, True])
        M = MaskedOperator(A, mask, mask)
        assert lx.is_symmetric(M) is False

    def test_symmetric_when_tagged(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        mask = jnp.array([True, True, False, True])
        M = MaskedOperator(A, mask, mask, tags=lx.symmetric_tag)
        assert lx.is_symmetric(M) is True

    def test_not_diagonal(self, getkey):
        A = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
        mask = jnp.array([True, True, False, True])
        M = MaskedOperator(A, mask, mask)
        assert lx.is_diagonal(M) is False

    def test_psd_when_tagged(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        mask = jnp.array([True, True, False, True])
        M = MaskedOperator(A, mask, mask, tags=lx.positive_semidefinite_tag)
        assert lx.is_positive_semidefinite(M) is True


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


class TestJAX:
    def test_jit(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        mask = jnp.array([True, True, False, True, False])
        M = MaskedOperator(A, mask, mask)
        v = jr.normal(getkey(), (3,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(M, v), M.as_matrix() @ v)

    def test_vmap(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        mask = jnp.array([True, True, False, True, False])
        M = MaskedOperator(A, mask, mask)
        vs = jr.normal(getkey(), (8, 3))
        results = jax.vmap(M.mv)(vs)
        assert results.shape == (8, 3)
        assert tree_allclose(results[0], M.as_matrix() @ vs[0])

    def test_grad(self, getkey):
        v = jr.normal(getkey(), (3,))
        mask = jnp.array([True, True, False, True, False])

        def loss(a_mat):
            A = lx.MatrixLinearOperator(a_mat)
            M = MaskedOperator(A, mask, mask)
            return jnp.sum(M.mv(v) ** 2)

        A_mat = jr.normal(getkey(), (5, 5))
        g = jax.grad(loss)(A_mat)
        assert g.shape == (5, 5)
        assert jnp.all(jnp.isfinite(g))
