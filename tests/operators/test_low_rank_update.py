"""Tests for the LowRankUpdate operator and convenience constructors."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._operators import (
    LowRankUpdate,
    low_rank_plus_diag,
    low_rank_plus_identity,
    svd_low_rank_plus_diag,
)
from gaussx._tags import is_low_rank
from gaussx._testing import tree_allclose


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


def test_low_rank_plus_diag_psd_is_a_claim_not_an_inference(getkey):
    # gh-343: the sign of ``diag`` is never inspected; PSD is the caller's
    # claim via ``psd=True``.
    diag = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    U = jr.normal(getkey(), (4, 2))
    lr = low_rank_plus_diag(diag, U)
    assert lx.is_symmetric(lr) is True
    assert lx.is_positive_semidefinite(lr) is False
    claimed = low_rank_plus_diag(diag, U, psd=True)
    assert lx.is_symmetric(claimed) is True
    assert lx.is_positive_semidefinite(claimed) is True
    assert lx.is_positive_semidefinite(claimed.base) is True


def test_low_rank_plus_identity_static_scale_is_psd(getkey):
    U = jr.normal(getkey(), (4, 2))
    assert lx.is_positive_semidefinite(low_rank_plus_identity(U)) is True
    assert lx.is_positive_semidefinite(low_rank_plus_identity(U, scale=-1.0)) is False
    array_scale = low_rank_plus_identity(U, scale=jnp.array(2.0))
    assert lx.is_positive_semidefinite(array_scale) is False


def test_svd_low_rank_plus_diag(getkey):
    diag = jr.normal(getkey(), (4,))
    U = jr.normal(getkey(), (4, 2))
    S = jnp.abs(jr.normal(getkey(), (2,)))
    V = U
    lr = svd_low_rank_plus_diag(diag, U, S, V)
    assert isinstance(lr, LowRankUpdate)
    assert lr.orthonormal is True
    assert jnp.allclose(lr.d, S)
    assert lx.is_symmetric(lr) is True
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(lr.mv(v), lr.as_matrix() @ v)


def test_svd_low_rank_plus_diag_matches_low_rank_plus_diag(getkey):
    diag = jr.normal(getkey(), (4,))
    U = jr.normal(getkey(), (4, 2))
    S = jnp.abs(jr.normal(getkey(), (2,)))
    V = jr.normal(getkey(), (4, 2))
    lr = svd_low_rank_plus_diag(diag, U, S, V)
    expected = low_rank_plus_diag(diag, U, S, V)
    assert tree_allclose(lr.as_matrix(), expected.as_matrix())
    assert tree_allclose(lr.d, expected.d)


def test_symmetry_requires_identity_not_equality(getkey):
    """Value-equal but distinct factors are not inferred symmetric (gh-343).

    A value check cannot survive tracing, so symmetry comes only from shared
    factors or from the caller's ``symmetric_tag``.
    """
    diag = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    base = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(diag),
        lx.positive_semidefinite_tag,
    )
    U = jr.normal(getkey(), (4, 2))
    V = U.copy()
    default = LowRankUpdate(base, U, orthonormal=False, V=V)
    orthonormal = LowRankUpdate(base, U, V=V, orthonormal=True)

    assert lx.is_symmetric(default) is False
    assert lx.is_symmetric(orthonormal) is False
    claimed = LowRankUpdate(base, U, V=V, tags=lx.symmetric_tag)
    assert lx.is_symmetric(claimed) is True
    assert claimed.symmetric_factors is False
    shared = LowRankUpdate(base, U, V=U, orthonormal=True)
    assert lx.is_symmetric(shared) is True
    assert shared.symmetric_factors is True


def test_svd_style_low_rank_update_supports_solve_and_logdet():
    n, k = 6, 3
    diag = jnp.ones(n) * 2.0
    key = jax.random.PRNGKey(42)
    U, _, _ = jnp.linalg.svd(jax.random.normal(key, (n, k)), full_matrices=False)
    S = jnp.array([3.0, 2.0, 1.0])
    base = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(diag),
        lx.positive_semidefinite_tag,
    )
    operator = LowRankUpdate(base, U, S, orthonormal=True)
    b = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    x = gaussx.solve(operator, b)
    residual = operator.mv(x) - b

    assert tree_allclose(operator.as_matrix(), operator.as_matrix().T)
    assert tree_allclose(operator.T.as_matrix(), operator.as_matrix().T)
    assert jnp.allclose(residual, 0.0, atol=1e-5)
    expected_logdet = jnp.linalg.slogdet(operator.as_matrix())[1]
    assert jnp.allclose(gaussx.logdet(operator), expected_logdet)


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


def test_svd_low_rank_update_deprecation_alias_returns_low_rank_update(getkey):
    """SVDLowRankUpdate alias must continue to return a LowRankUpdate.

    Backwards-compatible shim for downstream code that imports
    ``gaussx.SVDLowRankUpdate``; emits a DeprecationWarning.
    """
    import pytest

    import gaussx

    n, k = 5, 2
    base = lx.DiagonalLinearOperator(jnp.ones(n))
    U = jr.normal(getkey(), (n, k))
    S = jnp.abs(jr.normal(getkey(), (k,))) + 0.1
    V = U  # same object

    with pytest.warns(DeprecationWarning, match="SVDLowRankUpdate"):
        op = gaussx.SVDLowRankUpdate(base, U, S, V)

    assert isinstance(op, LowRankUpdate)
    assert op.orthonormal is True
    # Equivalent to constructing the LowRankUpdate directly.
    direct = LowRankUpdate(base, U, S, V, orthonormal=True)
    assert tree_allclose(op.as_matrix(), direct.as_matrix())


def test_svd_low_rank_update_class_supports_isinstance(getkey):
    """SVDLowRankUpdate is a real class so ``isinstance`` keeps working."""
    import pytest

    import gaussx

    n, k = 4, 2
    base = lx.DiagonalLinearOperator(jnp.ones(n))
    U = jr.normal(getkey(), (n, k))
    S = jnp.abs(jr.normal(getkey(), (k,))) + 0.1

    with pytest.warns(DeprecationWarning):
        op = gaussx.SVDLowRankUpdate(base, U, S)

    # isinstance / issubclass against both the alias and its parent.
    assert isinstance(op, gaussx.SVDLowRankUpdate)
    assert isinstance(op, LowRankUpdate)
    assert issubclass(gaussx.SVDLowRankUpdate, LowRankUpdate)


def test_svd_low_rank_update_optional_S_and_V_defaults(getkey):
    """The deprecation alias keeps the old optional-V (and -S) defaults."""
    import pytest

    import gaussx

    n, k = 4, 2
    base = lx.DiagonalLinearOperator(jnp.ones(n))
    U = jr.normal(getkey(), (n, k))

    # Old signature: ``SVDLowRankUpdate(base, U)`` — V defaults to U,
    # S defaults to ones (parent class behaviour).
    with pytest.warns(DeprecationWarning):
        op = gaussx.SVDLowRankUpdate(base, U)
    assert jnp.allclose(op.V, U)
    assert jnp.allclose(op.d, jnp.ones(k))

    # Old signature with explicit S: ``SVDLowRankUpdate(base, U, S)`` —
    # V should still default to U.
    S = jnp.abs(jr.normal(getkey(), (k,))) + 0.1
    with pytest.warns(DeprecationWarning):
        op = gaussx.SVDLowRankUpdate(base, U, S)
    assert jnp.allclose(op.V, U)
    assert jnp.allclose(op.d, S)


def test_svd_low_rank_plus_diag_value_equality_is_not_symmetry(getkey):
    """svd_low_rank_plus_diag with V = U.copy() is symmetric only if claimed.

    Value-equal but distinct factors lose any value-based inference under
    tracing (gh-343); pass ``V=U`` or ``psd=True`` instead.
    """
    n, k = 5, 2
    diag = jnp.abs(jr.normal(getkey(), (n,))) + 0.1
    U = jr.normal(getkey(), (n, k))
    S = jnp.abs(jr.normal(getkey(), (k,))) + 0.1
    V = U.copy()  # equal-but-distinct array

    op = svd_low_rank_plus_diag(diag, U, S, V)
    assert lx.is_symmetric(op) is False
    assert lx.is_symmetric(svd_low_rank_plus_diag(diag, U, S, U)) is True
    assert lx.is_positive_semidefinite(svd_low_rank_plus_diag(diag, U, S, V, psd=True))


# ---------------------------------------------------------------------------
# gh-343: tags are identical eagerly and under tracing
# ---------------------------------------------------------------------------


def _tag_report(op):
    return (
        lx.is_symmetric(op),
        lx.is_positive_semidefinite(op),
        jax.tree_util.tree_structure(op),
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda diag, U, d: low_rank_plus_diag(diag, U, d),
        lambda diag, U, d: low_rank_plus_diag(diag, U, d, psd=True),
        lambda diag, U, d: low_rank_plus_diag(diag, U, d, U),
        lambda diag, U, d: low_rank_plus_diag(diag, U, d, U.copy()),
        lambda diag, U, d: low_rank_plus_identity(U),
        lambda diag, U, d: LowRankUpdate(
            lx.TaggedLinearOperator(
                lx.DiagonalLinearOperator(diag), lx.positive_semidefinite_tag
            ),
            U,
        ),
    ],
    ids=["diag", "diag-psd", "V-is-U", "V-copy", "identity", "psd-base-unit-d"],
)
def test_tags_and_treedef_match_eager_and_traced(make):
    U = jr.normal(jr.key(0), (4, 2))
    diag, d = jnp.ones(4), jnp.array([1.0, 2.0])
    eager = _tag_report(make(diag, U, d))
    seen = {}

    def traced(diag, U, d):
        seen["report"] = _tag_report(make(diag, U, d))
        return make(diag, U, d)

    op_jit = jax.jit(traced)(diag, U, d)
    assert seen["report"] == eager
    op_filter = eqx.filter_jit(traced)(diag, U, d)
    assert seen["report"] == eager
    assert _tag_report(op_jit) == eager == _tag_report(op_filter)
    # Same treedef, so the two can meet in lax.cond.
    jax.lax.cond(True, lambda: make(diag, U, d), lambda: op_jit)


def test_transpose_keeps_shared_factors_under_jit():
    U = jr.normal(jr.key(0), (4, 2))
    op = LowRankUpdate(lx.DiagonalLinearOperator(jnp.ones(4)), U)

    @eqx.filter_jit
    def transposed_tags(op):
        return op.T.symmetric_factors, lx.is_symmetric(op.T)

    assert transposed_tags(op) == (True, True)


# ---------------------------------------------------------------------------
# Zero weights and rank 0 (gh-307)
# ---------------------------------------------------------------------------


def _zero_weight_operator(d, *, nonsymmetric=False, dtype=jnp.float64):
    U = jr.normal(jr.key(0), (5, 2), dtype=dtype)
    V = jr.normal(jr.key(1), (5, 2), dtype=dtype) if nonsymmetric else None
    base = lx.DiagonalLinearOperator(jnp.ones(5, dtype=dtype))
    return LowRankUpdate(base, U, d, V)


@pytest.mark.parametrize("nonsymmetric", [False, True], ids=["sym", "general"])
@pytest.mark.parametrize(
    "d", [[1.0, 0.0], [0.0, 0.0], [0.7, -0.2]], ids=["one_zero", "all_zero", "signed"]
)
def test_zero_weight_matches_dense(d, nonsymmetric):
    # The capacitance used to be diag(1/d) + V^T L^{-1} U: inf for d_k = 0.
    op = _zero_weight_operator(jnp.array(d), nonsymmetric=nonsymmetric)
    M = op.as_matrix()
    b = jnp.ones(5)
    tol = {"rtol": 1e-12, "atol": 1e-12}
    assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(M)[1], **tol)
    assert jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(M, b), **tol)
    assert jnp.allclose(gaussx.inv(op).as_matrix(), jnp.linalg.inv(M), **tol)


@pytest.mark.parametrize("nonsymmetric", [False, True], ids=["sym", "general"])
def test_zero_weight_gradients_match_dense(nonsymmetric):
    d0 = jnp.array([1.0, 0.0])
    b = jnp.ones(5)
    template = _zero_weight_operator(d0, nonsymmetric=nonsymmetric)

    def structured(d):
        op = eqx.tree_at(lambda o: o.d, template, d)
        return gaussx.solve(op, b).sum() + gaussx.logdet(op)

    def dense(d):
        op = eqx.tree_at(lambda o: o.d, template, d)
        M = op.as_matrix()
        return jnp.linalg.solve(M, b).sum() + jnp.linalg.slogdet(M)[1]

    grad = jax.grad(structured)(d0)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.allclose(grad, jax.grad(dense)(d0), rtol=1e-10, atol=1e-10)
    assert jnp.allclose(eqx.filter_jit(jax.grad(structured))(d0), grad, rtol=1e-12)


def test_zero_weight_float32():
    op = _zero_weight_operator(jnp.array([1.0, 0.0], jnp.float32), dtype=jnp.float32)
    M = op.as_matrix().astype(jnp.float64)
    b = jnp.ones(5, jnp.float32)
    assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(M)[1], rtol=1e-5)
    assert jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(M, b), atol=1e-5)
    assert jnp.allclose(gaussx.inv(op).as_matrix(), jnp.linalg.inv(M), atol=1e-5)


def test_rank_zero_is_the_base():
    base = lx.DiagonalLinearOperator(jnp.arange(1.0, 6.0))
    op = LowRankUpdate(base, jnp.zeros((5, 0)), jnp.zeros(0))
    b = jnp.ones(5)
    assert jnp.allclose(gaussx.solve(op, b), b / jnp.arange(1.0, 6.0))
    assert jnp.allclose(gaussx.logdet(op), jnp.sum(jnp.log(jnp.arange(1.0, 6.0))))
    assert jnp.allclose(gaussx.inv(op).as_matrix(), jnp.diag(1 / jnp.arange(1.0, 6.0)))
