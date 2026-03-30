"""Tests for the LowRankUpdate operator and convenience constructors."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import LowRankUpdate, low_rank_plus_diag, svd_low_rank_plus_diag
from gaussx._tags import is_low_rank


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_basic_construction(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3, 2))
    lr = LowRankUpdate(base, U)
    assert lr.in_size() == 3
    assert lr.out_size() == 3
    assert lr.rank == 2


def test_construction_with_d_and_v(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
    U = jr.normal(getkey(), (4, 2))
    d = jr.normal(getkey(), (2,))
    V = jr.normal(getkey(), (4, 2))
    lr = LowRankUpdate(base, U, d, V)
    assert lr.rank == 2


def test_1d_U_promoted_to_2d(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3,))
    lr = LowRankUpdate(base, U)
    assert lr.U.shape == (3, 1)
    assert lr.rank == 1


def test_defaults_d_to_ones(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3, 2))
    lr = LowRankUpdate(base, U)
    assert jnp.allclose(lr.d, jnp.ones(2))


def test_defaults_V_to_U(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3, 2))
    lr = LowRankUpdate(base, U)
    assert jnp.allclose(lr.V, lr.U)


def test_mismatched_dimensions_raises(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (2,)))
    U = jr.normal(getkey(), (3, 1))  # 3 rows but base is 2x2
    with pytest.raises(ValueError, match="rows"):
        LowRankUpdate(base, U)


def test_mismatched_rank_raises(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3, 2))
    d = jr.normal(getkey(), (3,))  # rank 3 but U has 2 cols
    with pytest.raises(ValueError, match="Rank dimensions"):
        LowRankUpdate(base, U, d)


def test_rectangular_base_supported(getkey):
    base = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 3)))
    U = jr.normal(getkey(), (2, 1))
    d = jnp.abs(jr.normal(getkey(), (1,))) + 0.1
    V = jr.normal(getkey(), (3, 1))
    lr = LowRankUpdate(base, U, d, V)
    v = jr.normal(getkey(), (3,))
    assert lr.in_size() == 3
    assert lr.out_size() == 2
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


# ---------------------------------------------------------------------------
# mv correctness — mv matches dense as_matrix
# ---------------------------------------------------------------------------


def test_mv_symmetric_rank1(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
    U = jr.normal(getkey(), (4, 1))
    lr = LowRankUpdate(base, U)
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


def test_mv_with_d_scaling(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (5,)))
    U = jr.normal(getkey(), (5, 2))
    d = jr.normal(getkey(), (2,))
    lr = LowRankUpdate(base, U, d)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


def test_mv_asymmetric(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
    U = jr.normal(getkey(), (4, 2))
    d = jr.normal(getkey(), (2,))
    V = jr.normal(getkey(), (4, 2))
    lr = LowRankUpdate(base, U, d, V)
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


def test_mv_with_dense_base(getkey):
    base = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    U = jr.normal(getkey(), (3, 1))
    lr = LowRankUpdate(base, U)
    v = jr.normal(getkey(), (3,))
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


def test_mv_random(getkey):
    n, k = 10, 3
    base = lx.DiagonalLinearOperator(jnp.abs(jr.normal(getkey(), (n,))) + 0.1)
    U = jr.normal(getkey(), (n, k))
    d = jr.normal(getkey(), (k,))
    lr = LowRankUpdate(base, U, d)
    v = jr.normal(getkey(), (n,))
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


def test_as_matrix_matches_formula(getkey):
    diag = jr.normal(getkey(), (3,))
    base = lx.DiagonalLinearOperator(diag)
    U = jr.normal(getkey(), (3, 2))
    d = jr.normal(getkey(), (2,))
    lr = LowRankUpdate(base, U, d)
    expected = jnp.diag(diag) + U @ jnp.diag(d) @ U.T
    assert tree_allclose(lr.as_matrix(), expected)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def test_transpose(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
    U = jr.normal(getkey(), (4, 2))
    d = jr.normal(getkey(), (2,))
    V = jr.normal(getkey(), (4, 2))
    lr = LowRankUpdate(base, U, d, V)
    assert tree_allclose(lr.T.as_matrix(), lr.as_matrix().T)


def test_transpose_mv(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
    U = jr.normal(getkey(), (4, 2))
    d = jr.normal(getkey(), (2,))
    V = jr.normal(getkey(), (4, 2))
    lr = LowRankUpdate(base, U, d, V)
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(lr.T.mv(v), lr.as_matrix().T @ v)


def test_transpose_swaps_U_and_V(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3, 2))
    V = jr.normal(getkey(), (3, 2))
    d = jr.normal(getkey(), (2,))
    lr = LowRankUpdate(base, U, d, V)
    lr_t = lr.T
    assert jnp.allclose(lr_t.U, V)
    assert jnp.allclose(lr_t.V, U)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_has_low_rank_tag(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3, 1))
    lr = LowRankUpdate(base, U)
    assert is_low_rank(lr) is True


def test_not_diagonal(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3, 1))
    lr = LowRankUpdate(base, U)
    assert lx.is_diagonal(lr) is False


def test_symmetric_tag_inferred_for_default_update(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    base = lx.DiagonalLinearOperator(d)
    U = jr.normal(getkey(), (4, 2))
    lr = LowRankUpdate(base, U)
    assert lx.is_symmetric(lr) is True


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def test_low_rank_plus_diag(getkey):
    diag = jr.normal(getkey(), (4,))
    U = jr.normal(getkey(), (4, 2))
    lr = low_rank_plus_diag(diag, U)
    assert isinstance(lr, LowRankUpdate)
    assert tree_allclose(lr.base.as_matrix(), jnp.diag(diag))
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


def test_low_rank_plus_diag_infers_psd(getkey):
    diag = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    U = jr.normal(getkey(), (4, 2))
    lr = low_rank_plus_diag(diag, U)
    assert lx.is_symmetric(lr) is True
    assert lx.is_positive_semidefinite(lr) is True


def test_svd_low_rank_plus_diag(getkey):
    diag = jr.normal(getkey(), (4,))
    U = jr.normal(getkey(), (4, 2))
    S = jnp.abs(jr.normal(getkey(), (2,)))
    V = jr.normal(getkey(), (4, 2))
    lr = svd_low_rank_plus_diag(diag, U, S, V)
    assert isinstance(lr, LowRankUpdate)
    assert jnp.allclose(lr.d, S)
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


def test_filter_jit_mv(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
    U = jr.normal(getkey(), (4, 2))
    lr = LowRankUpdate(base, U)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return op.mv(v)

    assert tree_allclose(f(lr, v), lr.as_matrix() @ v)


def test_vmap_mv(getkey):
    base = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
    U = jr.normal(getkey(), (3, 1))
    lr = LowRankUpdate(base, U)
    vs = jr.normal(getkey(), (4, 3))
    results = jax.vmap(lr.mv)(vs)
    assert results.shape == (4, 3)
    assert tree_allclose(results[0], lr.as_matrix() @ vs[0])
