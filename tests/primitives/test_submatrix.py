"""Tests for the structured submatrix primitive."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import (
    BlockDiag,
    Kronecker,
    LowRankUpdate,
    conditional,
    submatrix,
)
from gaussx._testing import random_pd_matrix, tree_allclose


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
    @pytest.mark.slow
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


# ---------------------------------------------------------------------------
# JIT compatibility & negative indices
# ---------------------------------------------------------------------------


class TestSubmatrixJit:
    def test_block_diag_under_jit(self, getkey):
        import jax

        a = random_pd_matrix(getkey(), 3)
        b = random_pd_matrix(getkey(), 4)
        block_diag = BlockDiag(
            lx.MatrixLinearOperator(a, lx.positive_semidefinite_tag),
            lx.MatrixLinearOperator(b, lx.positive_semidefinite_tag),
        )
        rows = jnp.array([0, 4, 6])
        cols = jnp.array([1, 5, 6])

        jitted = jax.jit(submatrix)
        result = jitted(block_diag, rows, cols)
        expected = block_diag.as_matrix()[jnp.ix_(rows, cols)]
        assert jnp.allclose(result, expected)


class TestNegativeIndices:
    def test_dense_negative_indices_match_ix(self, getkey):
        n = 6
        A = random_pd_matrix(getkey(), n)
        op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
        rows = jnp.array([-1, 0, -3])
        cols = jnp.array([-2, 1])
        normalized_rows = jnp.where(rows < 0, rows + n, rows)
        normalized_cols = jnp.where(cols < 0, cols + n, cols)
        expected = A[jnp.ix_(normalized_rows, normalized_cols)]
        assert jnp.allclose(submatrix(op, rows, cols), expected)

    def test_diagonal_negative_indices(self):
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        op = lx.DiagonalLinearOperator(d)
        rows = jnp.array([-1, -2])  # → (3, 2)
        cols = jnp.array([3, -3])  # → (3, 1)
        # (0,0): row=3, col=3 → 4.0 ; (0,1): 3 vs 1 → 0
        # (1,0): row=2, col=3 → 0   ; (1,1): 2 vs 1 → 0
        expected = jnp.array([[4.0, 0.0], [0.0, 0.0]])
        assert jnp.allclose(submatrix(op, rows, cols), expected)

    def test_block_diag_negative_indices_match_dense(self, getkey):
        a = random_pd_matrix(getkey(), 3)
        b = random_pd_matrix(getkey(), 4)
        block_diag = BlockDiag(
            lx.MatrixLinearOperator(a, lx.positive_semidefinite_tag),
            lx.MatrixLinearOperator(b, lx.positive_semidefinite_tag),
        )
        n = 7
        rows = jnp.array([-1, -7, 2])  # → (6, 0, 2)
        cols = jnp.array([-2, -5])  # → (5, 2)
        normalized_rows = jnp.where(rows < 0, rows + n, rows)
        normalized_cols = jnp.where(cols < 0, cols + n, cols)
        expected = block_diag.as_matrix()[jnp.ix_(normalized_rows, normalized_cols)]
        assert jnp.allclose(submatrix(block_diag, rows, cols), expected)


# ---------------------------------------------------------------------------
# LowRankUpdate and lineax wrappers keep structure (gh-379)
# ---------------------------------------------------------------------------

_ROWS = jnp.array([0, 5, 7, -1])
_COLS = jnp.array([1, 5, -2])


def _low_rank(key, *, general=False):
    k1, k2, k3 = jr.split(key, 3)
    U = jr.normal(k1, (9, 3))
    V = jr.normal(k2, (9, 3)) if general else None
    base = lx.DiagonalLinearOperator(jr.uniform(k3, (9,)) + 0.1)
    return LowRankUpdate(base, U, jnp.array([1.0, 0.5, 2.0]), V)


def _kron(key):
    k1, k2 = jr.split(key)
    return Kronecker(
        lx.MatrixLinearOperator(random_pd_matrix(k1, 3)),
        lx.MatrixLinearOperator(random_pd_matrix(k2, 3)),
    )


def _forbid_as_matrix(monkeypatch, *classes):
    for cls in classes:

        def _forbidden(self, cls=cls):
            raise AssertionError(f"{cls.__name__}.as_matrix called")

        monkeypatch.setattr(cls, "as_matrix", _forbidden)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda k: _low_rank(k), id="low_rank_symmetric"),
        pytest.param(lambda k: _low_rank(k, general=True), id="low_rank_general"),
        pytest.param(lambda k: 2.0 * _kron(k), id="mul"),
        pytest.param(lambda k: _kron(k) / 2.0, id="div"),
        pytest.param(lambda k: -_kron(k), id="neg"),
        pytest.param(
            lambda k: (
                _kron(k) + 0.5 * lx.IdentityLinearOperator(_kron(k).in_structure())
            ),
            id="add_identity",
        ),
    ],
)
def test_submatrix_structured_wrappers(monkeypatch, build):
    op = build(jr.key(0))
    expected = op.as_matrix()[jnp.ix_(_ROWS, _COLS)]
    _forbid_as_matrix(monkeypatch, LowRankUpdate, Kronecker)
    result = submatrix(op, _ROWS, _COLS)
    monkeypatch.undo()
    assert tree_allclose(result, expected)


def test_submatrix_traced_indices():
    op = _low_rank(jr.key(0))
    result = eqx.filter_jit(submatrix)(op, _ROWS, _COLS)
    assert tree_allclose(result, op.as_matrix()[jnp.ix_(_ROWS, _COLS)])


def test_conditional_low_rank_does_not_materialise(monkeypatch):
    op = _low_rank(jr.key(0))
    loc = jnp.zeros(9)
    obs_idx, obs_values = jnp.array([3, 6]), jnp.array([0.5, -0.5])
    dense = lx.MatrixLinearOperator(op.as_matrix(), lx.positive_semidefinite_tag)
    mean_ref, cov_ref = conditional(loc, dense, obs_idx, obs_values)
    _forbid_as_matrix(monkeypatch, LowRankUpdate)
    mean, cov = conditional(loc, op, obs_idx, obs_values)
    monkeypatch.undo()
    assert tree_allclose(mean, mean_ref)
    assert tree_allclose(cov.as_matrix(), cov_ref.as_matrix())
