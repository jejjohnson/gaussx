"""Tests for lazy algebra operators: SumOperator, ScaledOperator, ProductOperator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import Kronecker, ProductOperator, ScaledOperator, SumOperator
from gaussx._testing import tree_allclose


# =========================================================================
# SumOperator
# =========================================================================


class TestSumConstruction:
    def test_requires_at_least_two(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        with pytest.raises(ValueError, match="at least two"):
            SumOperator(A)

    def test_rejects_shape_mismatch(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        with pytest.raises(ValueError, match="Shape mismatch"):
            SumOperator(A, B)

    def test_basic_construction(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(A, B)
        assert S.in_size() == 3
        assert S.out_size() == 3


class TestSumMv:
    def test_mv_matches_dense(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(A, B)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(S.mv(v), S.as_matrix() @ v)

    def test_mv_three_operators(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        C = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(A, B, C)
        v = jr.normal(getkey(), (3,))
        expected = A.as_matrix() @ v + B.as_matrix() @ v + C.as_matrix() @ v
        assert tree_allclose(S.mv(v), expected)

    def test_mv_diagonal_factors(self, getkey):
        A = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
        B = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
        S = SumOperator(A, B)
        v = jr.normal(getkey(), (4,))
        assert tree_allclose(S.mv(v), S.as_matrix() @ v)

    def test_mv_mixed_operator_types(self, getkey):
        A = lx.DiagonalLinearOperator(jr.normal(getkey(), (4,)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4)))
        S = SumOperator(A, B)
        v = jr.normal(getkey(), (4,))
        assert tree_allclose(S.mv(v), S.as_matrix() @ v)


class TestSumAsMatrix:
    def test_as_matrix(self, getkey):
        A_mat = jr.normal(getkey(), (3, 3))
        B_mat = jr.normal(getkey(), (3, 3))
        S = SumOperator(lx.MatrixLinearOperator(A_mat), lx.MatrixLinearOperator(B_mat))
        assert tree_allclose(S.as_matrix(), A_mat + B_mat)


class TestSumTranspose:
    def test_transpose(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(A, B)
        assert tree_allclose(S.T.as_matrix(), S.as_matrix().T)

    def test_transpose_mv(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(A, B)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(S.T.mv(v), S.as_matrix().T @ v)


class TestSumTags:
    def test_symmetric_when_all_symmetric(self, getkey):
        A = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        S = SumOperator(A, B)
        assert lx.is_symmetric(S) is True

    def test_not_symmetric_when_factor_not_symmetric(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        S = SumOperator(A, B)
        assert lx.is_symmetric(S) is False

    def test_psd_when_all_psd(self, getkey):
        m1 = jr.normal(getkey(), (3, 3))
        A = lx.MatrixLinearOperator(m1.T @ m1, lx.positive_semidefinite_tag)
        m2 = jr.normal(getkey(), (3, 3))
        B = lx.MatrixLinearOperator(m2.T @ m2, lx.positive_semidefinite_tag)
        S = SumOperator(A, B)
        assert lx.is_positive_semidefinite(S) is True

    def test_diagonal_when_all_diagonal(self, getkey):
        A = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        S = SumOperator(A, B)
        assert lx.is_diagonal(S) is True


class TestSumJAX:
    def test_jit(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(A, B)
        v = jr.normal(getkey(), (3,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(S, v), S.as_matrix() @ v)

    def test_vmap(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(A, B)
        vs = jr.normal(getkey(), (5, 3))
        results = jax.vmap(S.mv)(vs)
        assert results.shape == (5, 3)
        assert tree_allclose(results[0], S.as_matrix() @ vs[0])

    def test_grad(self, getkey):
        A_mat = jr.normal(getkey(), (3, 3))
        B_mat = jr.normal(getkey(), (3, 3))
        v = jr.normal(getkey(), (3,))

        def loss(a_mat):
            A = lx.MatrixLinearOperator(a_mat)
            B = lx.MatrixLinearOperator(B_mat)
            return jnp.sum(SumOperator(A, B).mv(v) ** 2)

        g = jax.grad(loss)(A_mat)
        assert g.shape == (3, 3)
        assert jnp.all(jnp.isfinite(g))


# =========================================================================
# ScaledOperator
# =========================================================================


class TestScaledConstruction:
    def test_basic_construction(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = ScaledOperator(A, 2.5)
        assert S.in_size() == 3
        assert S.out_size() == 3


class TestScaledMv:
    def test_mv_matches_dense(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = ScaledOperator(A, 3.0)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(S.mv(v), S.as_matrix() @ v)

    def test_mv_matches_manual(self, getkey):
        A_mat = jr.normal(getkey(), (4, 4))
        c = 2.5
        S = ScaledOperator(lx.MatrixLinearOperator(A_mat), c)
        v = jr.normal(getkey(), (4,))
        assert tree_allclose(S.mv(v), c * A_mat @ v)

    def test_mv_negative_scalar(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = ScaledOperator(A, -1.0)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(S.mv(v), -A.mv(v))

    def test_mv_zero_scalar(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = ScaledOperator(A, 0.0)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(S.mv(v), jnp.zeros(3))


class TestScaledAsMatrix:
    def test_as_matrix(self, getkey):
        A_mat = jr.normal(getkey(), (3, 3))
        S = ScaledOperator(lx.MatrixLinearOperator(A_mat), 2.0)
        assert tree_allclose(S.as_matrix(), 2.0 * A_mat)


class TestScaledTranspose:
    def test_transpose(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = ScaledOperator(A, 3.0)
        assert tree_allclose(S.T.as_matrix(), S.as_matrix().T)

    def test_transpose_mv(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = ScaledOperator(A, 3.0)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(S.T.mv(v), S.as_matrix().T @ v)


class TestScaledTags:
    def test_symmetric_when_base_symmetric(self, getkey):
        A = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        S = ScaledOperator(A, 2.0)
        assert lx.is_symmetric(S) is True

    def test_not_symmetric_when_base_not_symmetric(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = ScaledOperator(A, 2.0)
        assert lx.is_symmetric(S) is False

    def test_diagonal_when_base_diagonal(self, getkey):
        A = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        S = ScaledOperator(A, 2.0)
        assert lx.is_diagonal(S) is True


class TestScaledJAX:
    def test_jit(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = ScaledOperator(A, 2.0)
        v = jr.normal(getkey(), (3,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(S, v), S.as_matrix() @ v)

    def test_grad(self, getkey):
        A_mat = jr.normal(getkey(), (3, 3))
        v = jr.normal(getkey(), (3,))

        def loss(c):
            return jnp.sum(ScaledOperator(lx.MatrixLinearOperator(A_mat), c).mv(v) ** 2)

        g = jax.grad(loss)(2.0)
        assert jnp.isfinite(g)


# =========================================================================
# ProductOperator
# =========================================================================


class TestProductConstruction:
    def test_rejects_dimension_mismatch(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 4)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        with pytest.raises(ValueError, match="Inner dimension mismatch"):
            ProductOperator(A, B)

    def test_basic_construction(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 4)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 5)))
        P = ProductOperator(A, B)
        assert P.in_size() == 5
        assert P.out_size() == 3


class TestProductMv:
    def test_mv_matches_dense(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 4)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 5)))
        P = ProductOperator(A, B)
        v = jr.normal(getkey(), (5,))
        assert tree_allclose(P.mv(v), P.as_matrix() @ v)

    def test_mv_square(self, getkey):
        A_mat = jr.normal(getkey(), (3, 3))
        B_mat = jr.normal(getkey(), (3, 3))
        P = ProductOperator(
            lx.MatrixLinearOperator(A_mat), lx.MatrixLinearOperator(B_mat)
        )
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(P.mv(v), (A_mat @ B_mat) @ v)

    def test_mv_with_diagonal(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        P = ProductOperator(A, B)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(P.mv(v), P.as_matrix() @ v)


class TestProductAsMatrix:
    def test_as_matrix(self, getkey):
        A_mat = jr.normal(getkey(), (3, 4))
        B_mat = jr.normal(getkey(), (4, 5))
        P = ProductOperator(
            lx.MatrixLinearOperator(A_mat), lx.MatrixLinearOperator(B_mat)
        )
        assert tree_allclose(P.as_matrix(), A_mat @ B_mat)


class TestProductTranspose:
    def test_transpose(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 4)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 5)))
        P = ProductOperator(A, B)
        assert tree_allclose(P.T.as_matrix(), P.as_matrix().T)

    def test_transpose_mv(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 4)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 5)))
        P = ProductOperator(A, B)
        v = jr.normal(getkey(), (3,))
        assert tree_allclose(P.T.mv(v), P.as_matrix().T @ v)


class TestProductTags:
    def test_not_symmetric_by_default(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        P = ProductOperator(A, B)
        assert lx.is_symmetric(P) is False

    def test_not_diagonal(self, getkey):
        A = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        B = lx.DiagonalLinearOperator(jr.normal(getkey(), (3,)))
        P = ProductOperator(A, B)
        assert lx.is_diagonal(P) is False


class TestProductJAX:
    def test_jit(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 4)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (4, 5)))
        P = ProductOperator(A, B)
        v = jr.normal(getkey(), (5,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(P, v), P.as_matrix() @ v)

    def test_vmap(self, getkey):
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        P = ProductOperator(A, B)
        vs = jr.normal(getkey(), (5, 3))
        results = jax.vmap(P.mv)(vs)
        assert results.shape == (5, 3)
        assert tree_allclose(results[0], P.as_matrix() @ vs[0])

    def test_grad(self, getkey):
        A_mat = jr.normal(getkey(), (3, 3))
        B_mat = jr.normal(getkey(), (3, 3))
        v = jr.normal(getkey(), (3,))

        def loss(a_mat):
            A = lx.MatrixLinearOperator(a_mat)
            B = lx.MatrixLinearOperator(B_mat)
            return jnp.sum(ProductOperator(A, B).mv(v) ** 2)

        g = jax.grad(loss)(A_mat)
        assert g.shape == (3, 3)
        assert jnp.all(jnp.isfinite(g))


# =========================================================================
# Composition tests — operators working together
# =========================================================================


class TestComposition:
    def test_sum_of_scaled(self, getkey):
        """(2A + 3B) v should match dense."""
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(ScaledOperator(A, 2.0), ScaledOperator(B, 3.0))
        v = jr.normal(getkey(), (3,))
        expected = 2.0 * A.as_matrix() @ v + 3.0 * B.as_matrix() @ v
        assert tree_allclose(S.mv(v), expected)

    def test_scaled_product(self, getkey):
        """c(AB) v should match dense."""
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        P = ScaledOperator(ProductOperator(A, B), 2.0)
        v = jr.normal(getkey(), (3,))
        expected = 2.0 * (A.as_matrix() @ B.as_matrix()) @ v
        assert tree_allclose(P.mv(v), expected)

    def test_sum_of_kronecker_products(self, getkey):
        """(A1⊗B1 + A2⊗B2) v should match dense."""
        A1 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B1 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        A2 = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)))
        B2 = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        K1 = Kronecker(A1, B1)
        K2 = Kronecker(A2, B2)
        S = SumOperator(K1, K2)
        v = jr.normal(getkey(), (6,))
        expected = (K1.as_matrix() + K2.as_matrix()) @ v
        assert tree_allclose(S.mv(v), expected)

    def test_product_then_sum(self, getkey):
        """(AB + C) v should match dense."""
        A = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        C = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
        S = SumOperator(ProductOperator(A, B), C)
        v = jr.normal(getkey(), (3,))
        expected = (A.as_matrix() @ B.as_matrix() + C.as_matrix()) @ v
        assert tree_allclose(S.mv(v), expected)
