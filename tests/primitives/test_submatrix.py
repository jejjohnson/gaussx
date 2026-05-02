"""Tests for the structured submatrix primitive."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import BlockDiag, submatrix
from gaussx._testing import random_pd_matrix


class TestSubmatrixDense:
    def test_matches_ix_indexing(self, getkey):
        n = 6
        A = random_pd_matrix(getkey(), n)
        op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
        rows = jnp.array([1, 3, 5])
        cols = jnp.array([0, 2, 4])

        result = submatrix(op, rows, cols)
        expected = A[jnp.ix_(rows, cols)]
        assert jnp.allclose(result, expected)


class TestSubmatrixDiagonal:
    def test_matches_dense(self):
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        op = lx.DiagonalLinearOperator(d)
        full = jnp.diag(d)
        rows = jnp.array([0, 2, 4])
        cols = jnp.array([0, 2, 4])
        assert jnp.allclose(submatrix(op, rows, cols), full[jnp.ix_(rows, cols)])

    def test_off_diagonal_indices_zero(self):
        # Indices chosen so no (row_idx[i], col_idx[j]) pair coincides.
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        op = lx.DiagonalLinearOperator(d)
        rows = jnp.array([0, 2])
        cols = jnp.array([1, 3])
        assert jnp.allclose(submatrix(op, rows, cols), jnp.zeros((2, 2)))


class TestSubmatrixBlockDiag:
    def test_matches_dense(self, getkey):
        # Build BlockDiag with two PSD blocks of sizes 3 and 4.
        a = random_pd_matrix(getkey(), 3)
        b = random_pd_matrix(getkey(), 4)
        block_diag = BlockDiag(
            lx.MatrixLinearOperator(a, lx.positive_semidefinite_tag),
            lx.MatrixLinearOperator(b, lx.positive_semidefinite_tag),
        )
        full = block_diag.as_matrix()

        # Mix indices across both blocks.
        key = jr.PRNGKey(0)
        rows = jr.permutation(key, jnp.arange(7))[:5]
        cols = jr.permutation(jr.fold_in(key, 1), jnp.arange(7))[:5]

        assert jnp.allclose(
            submatrix(block_diag, rows, cols), full[jnp.ix_(rows, cols)]
        )

    def test_within_single_block(self, getkey):
        a = random_pd_matrix(getkey(), 4)
        b = random_pd_matrix(getkey(), 3)
        block_diag = BlockDiag(
            lx.MatrixLinearOperator(a, lx.positive_semidefinite_tag),
            lx.MatrixLinearOperator(b, lx.positive_semidefinite_tag),
        )
        full = block_diag.as_matrix()
        rows = jnp.array([4, 5, 6])  # all in second block
        cols = jnp.array([4, 5, 6])
        assert jnp.allclose(
            submatrix(block_diag, rows, cols), full[jnp.ix_(rows, cols)]
        )
