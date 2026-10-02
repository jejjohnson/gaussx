"""Tests for the Takahashi selected inverse (G4)."""

from __future__ import annotations

import numpy as np
import pytest

import gaussx
from gaussx import SparseOperator
from gaussx._sparse._takahashi import takahashi


@pytest.mark.parametrize(
    ("ordering", "banded", "fixed_effect"),
    [
        ("natural", True, False),
        ("natural", False, False),
        ("rcm", True, False),
        ("rcm", False, False),
        ("rcm", False, True),
    ],
)
def test_equals_dense_inverse_on_pattern_of_l(
    grid, symbolic_for, ordering, banded, fixed_effect
):
    op = grid(4, 5, fixed_effect=fixed_effect)
    sym = symbolic_for(op, ordering, banded=banded)
    factor = gaussx.sparse_cholesky(op, sym)
    Z = takahashi(sym, factor.values)
    inverse = np.linalg.inv(np.asarray(op.as_matrix())[np.ix_(sym.perm, sym.perm)])
    np.testing.assert_allclose(Z, inverse[sym.rowidx, sym.colidx], atol=1e-12)


@pytest.mark.parametrize("symmetric", [True, False])
def test_selected_inverse_in_original_order(grid, symmetric):
    op = grid(4, 4, symmetric=symmetric, fixed_effect=True)
    selected = gaussx.sparse_cholesky(op).selected_inverse()
    assert isinstance(selected, SparseOperator)
    assert selected.pattern.symmetric
    inverse = np.linalg.inv(np.asarray(op.as_matrix()))
    rows, cols = selected.pattern.rows, selected.pattern.cols
    np.testing.assert_allclose(selected.values, inverse[rows, cols], atol=1e-12)
    # pattern(Q) ⊆ pattern(L + Lᵀ): every entry the logdet gradient needs.
    stored = set(zip(rows.tolist(), cols.tolist(), strict=True))
    lower = (
        np.maximum(op.pattern.rows, op.pattern.cols),
        np.minimum(op.pattern.rows, op.pattern.cols),
    )
    assert set(zip(lower[0].tolist(), lower[1].tolist(), strict=True)) <= stored


def test_diag_inv(grid):
    op = grid(5, 4)
    np.testing.assert_allclose(
        gaussx.sparse_cholesky(op).diag_inv(),
        np.diag(np.linalg.inv(np.asarray(op.as_matrix()))),
        atol=1e-12,
    )
