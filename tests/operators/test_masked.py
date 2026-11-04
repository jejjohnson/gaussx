"""Tests for the MaskedOperator."""

from __future__ import annotations

from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._operators import MaskedOperator
from gaussx._testing import (
    default_tolerances,
    random_kronecker_pd,
    random_pd_operator,
    tree_allclose,
)


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

    @pytest.mark.slow
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


# ---------------------------------------------------------------------------
# masked_obs_covariance: R̃ = M R M + (I − M)
# ---------------------------------------------------------------------------


_OBS_MASK = jnp.array([True, False, True, True, False, True])


def _dense_covariance_cases():
    psd = lx.positive_semidefinite_tag
    block = random_pd_operator(jr.key(1), 4)
    other = random_pd_operator(jr.key(2), 2)
    return [
        pytest.param(lambda: random_pd_operator(jr.key(0), 6), id="dense"),
        pytest.param(
            lambda: lx.DiagonalLinearOperator(
                jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.result_type(float))
            ),
            id="diagonal",
        ),
        pytest.param(lambda: random_kronecker_pd(jr.key(3), (2, 3)), id="kronecker"),
        pytest.param(lambda: gaussx.BlockDiag(block, other, tags=psd), id="block_diag"),
    ]


def _row_column_deleted(matrix, mask):
    """Dense reference: R_oo on o, the identity on u, no cross terms."""
    observed = jnp.flatnonzero(mask)
    reference = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    return reference.at[jnp.ix_(observed, observed)].set(
        matrix[jnp.ix_(observed, observed)]
    )


class TestMaskedObsCovariance:
    @pytest.mark.parametrize("build", _dense_covariance_cases())
    def test_matches_the_row_and_column_deleted_reference(self, build):
        R = build()
        masked = gaussx.masked_obs_covariance(R, _OBS_MASK)
        reference = _row_column_deleted(R.as_matrix(), _OBS_MASK)
        rtol, atol = default_tolerances(reference)
        assert jnp.allclose(masked.as_matrix(), reference, rtol=rtol, atol=atol)
        # r_u = 0, as the masked residual has
        residual = jnp.where(_OBS_MASK, jr.normal(jr.key(4), (6,), reference.dtype), 0)
        solved = gaussx.solve(masked, residual)
        expected = jnp.linalg.solve(reference, residual)
        assert jnp.allclose(solved, expected, rtol=100 * rtol, atol=100 * atol)
        assert jnp.allclose(solved[~_OBS_MASK], 0.0, atol=100 * atol)
        logdet = jnp.linalg.slogdet(reference)[1]
        assert jnp.allclose(gaussx.logdet(masked), logdet, rtol=100 * rtol, atol=atol)

    def test_a_float_mask_is_the_same_mask(self):
        R = random_pd_operator(jr.key(0), 6)
        by_bool = gaussx.masked_obs_covariance(R, _OBS_MASK).as_matrix()
        by_float = gaussx.masked_obs_covariance(
            R, _OBS_MASK.astype(jnp.result_type(float))
        ).as_matrix()
        assert jnp.array_equal(by_bool, by_float)

    def test_correlated_r_weights_by_the_observed_block(self):
        """vardax#189: ρ = 0.9 weights y₀ by R₀₀ = 1, not 1 / (1 − ρ²)."""
        R = lx.MatrixLinearOperator(
            jnp.array([[1.0, 0.9], [0.9, 1.0]]), lx.positive_semidefinite_tag
        )
        masked = gaussx.masked_obs_covariance(R, jnp.array([True, False]))
        solved = gaussx.solve(masked, jnp.array([1.0, 0.0]))
        assert jnp.allclose(solved, jnp.array([1.0, 0.0]))

    def test_structure_is_kept(self):
        tagged = lx.TaggedLinearOperator(
            lx.DiagonalLinearOperator(jnp.ones(6)), lx.positive_semidefinite_tag
        )
        assert isinstance(
            gaussx.masked_obs_covariance(tagged, _OBS_MASK),
            lx.DiagonalLinearOperator,
        )
        shaped = lx.DiagonalLinearOperator(2.0 * jnp.ones((2, 3)))
        masked = gaussx.masked_obs_covariance(shaped, jnp.ones((2, 3), dtype=bool))
        assert masked.in_structure().shape == (2, 3)
        identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((6,), jnp.float32))
        assert gaussx.masked_obs_covariance(identity, _OBS_MASK) is identity
        blocks = gaussx.BlockDiag(
            random_pd_operator(jr.key(1), 4), random_pd_operator(jr.key(2), 2)
        )
        assert isinstance(
            gaussx.masked_obs_covariance(blocks, _OBS_MASK), gaussx.BlockDiag
        )

    def test_lazy_path_never_materialises_r(self):
        R = random_kronecker_pd(jr.key(3), (2, 3))
        masked = gaussx.masked_obs_covariance(R, _OBS_MASK)
        assert lx.is_positive_semidefinite(masked)
        reference = _row_column_deleted(R.as_matrix(), _OBS_MASK)
        vector = jr.normal(jr.key(4), (6,), reference.dtype)
        with forbid_as_matrix(gaussx.Kronecker):
            applied = masked.mv(vector)
        rtol, atol = default_tolerances(reference)
        assert jnp.allclose(applied, reference @ vector, rtol=rtol, atol=atol)

    def test_all_zero_and_all_one_masks(self):
        R = random_pd_operator(jr.key(0), 6)
        rtol, atol = default_tolerances(R.as_matrix())
        none = gaussx.masked_obs_covariance(R, jnp.zeros(6, dtype=bool)).as_matrix()
        assert jnp.allclose(none, jnp.eye(6), rtol=rtol, atol=atol)
        every = gaussx.masked_obs_covariance(R, jnp.ones(6, dtype=bool)).as_matrix()
        assert jnp.allclose(every, R.as_matrix(), rtol=rtol, atol=atol)

    def test_non_finite_masked_entries_do_not_leak(self):
        matrix = jnp.array([[1.0, jnp.nan], [jnp.nan, jnp.nan]])
        masked = gaussx.masked_obs_covariance(
            lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag),
            jnp.array([True, False]),
        )
        assert jnp.array_equal(masked.as_matrix(), jnp.eye(2))
        diagonal = gaussx.masked_obs_covariance(
            lx.DiagonalLinearOperator(jnp.array([1.0, jnp.nan])),
            jnp.array([True, False]),
        )
        assert jnp.array_equal(diagonal.as_matrix(), jnp.eye(2))

    def test_wrong_mask_shape_raises(self):
        R = random_pd_operator(jr.key(0), 6)
        with pytest.raises(ValueError, match="mask has shape"):
            gaussx.masked_obs_covariance(R, jnp.ones(5, dtype=bool))

    @pytest.mark.parametrize("build", _dense_covariance_cases())
    def test_traced_mask_under_jit_and_vmap(self, build):
        R = build()
        vector = jnp.ones(6, dtype=R.in_structure().dtype)

        def apply(mask):
            return gaussx.masked_obs_covariance(R, mask).mv(vector)

        expected = apply(_OBS_MASK)
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(jax.jit(apply)(_OBS_MASK), expected, rtol=rtol, atol=atol)
        batched = jax.vmap(apply)(jnp.stack([_OBS_MASK, ~_OBS_MASK]))
        assert jnp.allclose(batched[0], expected, rtol=rtol, atol=atol)
        assert jnp.allclose(batched[1], apply(~_OBS_MASK), rtol=rtol, atol=atol)

    def test_gradient_matches_the_dense_reference(self):
        """∂/∂R of ½ rᵀ R̃⁻¹ r, by way of the operator and of the reference."""
        matrix = random_pd_operator(jr.key(0), 6).as_matrix()
        residual = jnp.where(_OBS_MASK, jnp.ones(6, dtype=matrix.dtype), 0)

        def through_operator(entries):
            operator = lx.MatrixLinearOperator(entries, lx.positive_semidefinite_tag)
            masked = gaussx.masked_obs_covariance(operator, _OBS_MASK)
            return 0.5 * residual @ gaussx.solve(masked, residual)

        def through_reference(entries):
            reference = _row_column_deleted(entries, _OBS_MASK)
            return 0.5 * residual @ jnp.linalg.solve(reference, residual)

        actual = jax.grad(through_operator)(matrix)
        expected = jax.grad(through_reference)(matrix)
        assert jnp.all(jnp.isfinite(actual))
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(actual, expected, rtol=100 * rtol, atol=100 * atol)


def forbid_as_matrix(cls):
    """Fail if an instance of ``cls`` is materialised inside the block."""
    return mock.patch.object(
        cls, "as_matrix", side_effect=AssertionError(f"{cls.__name__}.as_matrix")
    )
