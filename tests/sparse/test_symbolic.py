"""Tests for the host-side symbolic Cholesky analysis (G4)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import gaussx
from gaussx import SparsityPattern


def _path_pattern(n: int) -> SparsityPattern:
    return SparsityPattern(np.arange(1, n), np.arange(n - 1), (n, n), symmetric=True)


def _triangulated_square(side: int) -> SparsityPattern:
    """A right-triangle mesh of the unit square: the P1 FEM 7-point stencil."""
    i = np.arange(side * side)
    right = i[i % side != side - 1]
    down = i[i < (side - 1) * side]
    diagonal = right[right < (side - 1) * side]  # one diagonal per cell
    rows = np.r_[right + 1, down + side, diagonal + side + 1]
    cols = np.r_[right, down, diagonal]
    n = side * side
    return SparsityPattern(rows, cols, (n, n), symmetric=True)


class TestStructure:
    def test_path_graph_has_no_fill(self):
        sym = gaussx.symbolic_cholesky(_path_pattern(6), ordering="natural")
        assert sym.nnz == 11
        assert sym.fill_ratio == 1.0
        np.testing.assert_array_equal(sym.parent, [1, 2, 3, 4, 5, -1])
        np.testing.assert_array_equal(sym.perm, np.arange(6))

    @pytest.mark.parametrize("ordering", ["natural", "rcm"])
    @pytest.mark.parametrize("fixed_effect", [False, True])
    def test_pattern_equals_dense_factor(self, grid, ordering, fixed_effect):
        # Random weights leave no exact cancellation, so the structural
        # pattern of L is exactly the dense factor's non-zeros.
        op = grid(4, 5, fixed_effect=fixed_effect)
        sym = gaussx.symbolic_cholesky(op.pattern, ordering=ordering)
        dense = np.asarray(op.as_matrix())[np.ix_(sym.perm, sym.perm)]
        expected = np.abs(np.linalg.cholesky(dense)) > 1e-14
        found = np.zeros_like(expected)
        found[sym.rowidx, sym.colidx] = True
        np.testing.assert_array_equal(found, expected)
        # Columns are sorted with the diagonal first.
        np.testing.assert_array_equal(sym.rowidx[sym.colptr[:-1]], np.arange(sym.n))
        # parent(j) = min{i > j : L_ij != 0}
        for j in range(sym.n):
            below = np.flatnonzero(expected[j + 1 :, j])
            assert sym.parent[j] == (j + 1 + below[0] if below.size else -1)

    def test_permutation(self, grid):
        sym = gaussx.symbolic_cholesky(grid(5, 6).pattern)
        np.testing.assert_array_equal(np.sort(sym.perm), np.arange(30))
        np.testing.assert_array_equal(sym.perm[sym.iperm], np.arange(30))

    def test_rcm_fill_on_reference_mesh_is_recorded_and_bounded(self):
        # 30 x 30 triangulated square (n = 900, nnz(tril Q) = 3,481). Recorded:
        #   natural ordering:  nnz(L) = 27,870 (bandwidth 31, fill 8.0)
        #   RCM ordering:      nnz(L) = 19,315 (bandwidth 30, fill 5.5)
        #   AMD (CHOLMOD):     nnz(L) = 14,377 (fill 4.1; tests/sparse/test_cholmod.py)
        # RCM's envelope bounds it by n * (bandwidth + 1).
        pattern = _triangulated_square(30)
        natural = gaussx.symbolic_cholesky(pattern, ordering="natural")
        rcm = gaussx.symbolic_cholesky(pattern, ordering="rcm")
        assert rcm.nnz_lower == 3_481
        assert natural.nnz == 27_870
        assert rcm.nnz == 19_315
        assert rcm.nnz <= 900 * (rcm.block_size + 1)
        assert rcm.fill_ratio < 6.0
        assert rcm.banded  # the band is the cheap layout here

    def test_arrow_pattern_uses_windows(self, grid):
        # A node joined to everything puts a dense row in L: no narrow band.
        sym = gaussx.symbolic_cholesky(grid(6, 6, fixed_effect=True).pattern)
        assert not sym.banded

    def test_general_storage_is_symmetrised(self, grid):
        lower = gaussx.symbolic_cholesky(grid(3, 4).pattern)
        full = gaussx.symbolic_cholesky(grid(3, 4, symmetric=False).pattern)
        assert full.nnz == lower.nnz
        np.testing.assert_array_equal(full.rowidx, lower.rowidx)
        assert full.nnz_lower == lower.nnz_lower


class TestCaching:
    def test_cached_per_pattern_content(self):
        a = gaussx.symbolic_cholesky(_path_pattern(7))
        b = gaussx.symbolic_cholesky(_path_pattern(7))  # equal, distinct object
        assert a is b

    def test_hash_and_equality(self):
        p = _path_pattern(5)
        rcm = gaussx.symbolic_cholesky(p)
        natural = gaussx.symbolic_cholesky(p, ordering="natural")
        assert rcm != natural
        assert hash(rcm) == hash(gaussx.symbolic_cholesky(p))
        assert "nnz_L=9" in repr(rcm)


class TestInversePlan:
    def test_contains_pattern_of_q(self, grid):
        op = grid(4, 4, fixed_effect=True)
        sym = gaussx.symbolic_cholesky(op.pattern)
        pattern, index = sym.inverse_plan
        assert pattern.symmetric
        assert pattern.nnz == sym.nnz
        np.testing.assert_array_equal(np.sort(index), np.arange(sym.nnz))
        keys = set(zip(pattern.rows.tolist(), pattern.cols.tolist(), strict=True))
        q = set(zip(op.pattern.rows.tolist(), op.pattern.cols.tolist(), strict=True))
        assert q <= keys


class TestErrors:
    def test_non_square(self):
        p = SparsityPattern(np.array([0]), np.array([1]), (2, 3))
        with pytest.raises(ValueError, match="square"):
            gaussx.symbolic_cholesky(p)

    def test_unknown_options(self):
        p = _path_pattern(3)
        with pytest.raises(ValueError, match="ordering"):
            gaussx.symbolic_cholesky(p, ordering="metis")  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="backend"):
            gaussx.symbolic_cholesky(p, backend="cuda")  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        ("ordering", "backend"), [("amd", "jax"), ("rcm", "cholmod")]
    )
    def test_cholmod_missing(self, monkeypatch, ordering, backend):
        monkeypatch.setitem(sys.modules, "sksparse", None)
        with pytest.raises(ImportError, match="scikit-sparse"):
            gaussx.symbolic_cholesky(
                _path_pattern(4),
                ordering=ordering,
                backend=backend,  # ty: ignore[invalid-argument-type]
            )
