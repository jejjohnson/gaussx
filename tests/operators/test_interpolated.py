"""Tests for the InterpolatedOperator (SKI / KISS-GP)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import InterpolatedOperator, Toeplitz
from gaussx._testing import tree_allclose


def _make_interp(n, m, p, key):
    """Create random interpolation indices and weights."""
    k1, k2 = jr.split(key)
    indices = jr.randint(k1, (n, p), 0, m)
    values = jr.uniform(k2, (n, p), minval=0.0, maxval=1.0)
    # Normalize rows so they sum to 1 (like proper interpolation weights)
    values = values / jnp.sum(values, axis=-1, keepdims=True)
    return indices, values


def _build_W(indices, values, m):
    """Build dense interpolation matrix W."""
    n = indices.shape[0]
    W = jnp.zeros((n, m), dtype=values.dtype)
    rows = jnp.arange(n)[:, None]
    W = W.at[rows, indices].add(values)
    return W


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, getkey):
        m = 10
        K_uu = lx.MatrixLinearOperator(jr.normal(getkey(), (m, m)))
        indices, values = _make_interp(20, m, 4, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        assert op.in_size() == 20
        assert op.out_size() == 20

    def test_rejects_shape_mismatch(self, getkey):
        K_uu = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        indices = jr.randint(getkey(), (10, 3), 0, 5)
        values = jr.normal(getkey(), (10, 4))  # wrong p
        with pytest.raises(ValueError, match="same shape"):
            InterpolatedOperator(K_uu, indices, values)


# ---------------------------------------------------------------------------
# mv correctness
# ---------------------------------------------------------------------------


class TestMv:
    def test_mv_matches_dense(self, getkey):
        m = 8
        n = 15
        p = 3
        K_uu_mat = jr.normal(getkey(), (m, m))
        K_uu = lx.MatrixLinearOperator(K_uu_mat)
        indices, values = _make_interp(n, m, p, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        v = jr.normal(getkey(), (n,))
        assert tree_allclose(op.mv(v), op.as_matrix() @ v, rtol=1e-5)

    def test_mv_with_toeplitz_base(self, getkey):
        m = 16
        n = 30
        p = 4
        column = jr.normal(getkey(), (m,))
        K_uu = Toeplitz(column)
        indices, values = _make_interp(n, m, p, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        v = jr.normal(getkey(), (n,))
        # Compare against W @ K_uu_dense @ W^T @ v
        W = _build_W(indices, values, m)
        expected = W @ (K_uu.as_matrix() @ (W.T @ v))
        assert tree_allclose(op.mv(v), expected, rtol=1e-4)

    def test_mv_with_diagonal_base(self, getkey):
        m = 6
        n = 10
        p = 2
        diag_vals = jnp.abs(jr.normal(getkey(), (m,))) + 0.1
        K_uu = lx.DiagonalLinearOperator(diag_vals)
        indices, values = _make_interp(n, m, p, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        v = jr.normal(getkey(), (n,))
        assert tree_allclose(op.mv(v), op.as_matrix() @ v, rtol=1e-5)

    def test_mv_single_point_interpolation(self, getkey):
        """p=1: each data point maps to exactly one inducing point."""
        m = 5
        n = 8
        K_uu_mat = jr.normal(getkey(), (m, m))
        K_uu = lx.MatrixLinearOperator(K_uu_mat)
        indices = jr.randint(getkey(), (n, 1), 0, m)
        values = jnp.ones((n, 1))
        op = InterpolatedOperator(K_uu, indices, values)
        v = jr.normal(getkey(), (n,))
        assert tree_allclose(op.mv(v), op.as_matrix() @ v, rtol=1e-5)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


class TestAsMatrix:
    def test_as_matrix_matches_manual(self, getkey):
        m = 6
        n = 10
        p = 3
        K_uu_mat = jr.normal(getkey(), (m, m))
        K_uu = lx.MatrixLinearOperator(K_uu_mat)
        indices, values = _make_interp(n, m, p, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        W = _build_W(indices, values, m)
        expected = W @ K_uu_mat @ W.T
        assert tree_allclose(op.as_matrix(), expected, rtol=1e-5)

    def test_symmetric_when_base_symmetric(self, getkey):
        m = 6
        n = 10
        p = 2
        mat = jr.normal(getkey(), (m, m))
        K_uu_mat = mat + mat.T  # symmetric
        K_uu = lx.MatrixLinearOperator(K_uu_mat, lx.symmetric_tag)
        indices, values = _make_interp(n, m, p, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        M = op.as_matrix()
        assert tree_allclose(M, M.T, rtol=1e-5)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_transpose(self, getkey):
        m = 6
        n = 10
        p = 3
        K_uu = lx.MatrixLinearOperator(jr.normal(getkey(), (m, m)))
        indices, values = _make_interp(n, m, p, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        assert tree_allclose(op.T.as_matrix(), op.as_matrix().T, rtol=1e-5)

    def test_transpose_mv(self, getkey):
        m = 6
        n = 10
        p = 3
        K_uu = lx.MatrixLinearOperator(jr.normal(getkey(), (m, m)))
        indices, values = _make_interp(n, m, p, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        v = jr.normal(getkey(), (n,))
        assert tree_allclose(op.T.mv(v), op.as_matrix().T @ v, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_not_symmetric_by_default(self, getkey):
        K_uu = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        indices, values = _make_interp(8, 5, 2, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        assert lx.is_symmetric(op) is False

    def test_symmetric_when_tagged(self, getkey):
        K_uu = lx.MatrixLinearOperator(jr.normal(getkey(), (5, 5)))
        indices, values = _make_interp(8, 5, 2, getkey())
        op = InterpolatedOperator(K_uu, indices, values, tags=lx.symmetric_tag)
        assert lx.is_symmetric(op) is True

    def test_not_diagonal(self, getkey):
        K_uu = lx.DiagonalLinearOperator(jr.normal(getkey(), (5,)))
        indices, values = _make_interp(8, 5, 2, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        assert lx.is_diagonal(op) is False


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


class TestJAX:
    def test_jit(self, getkey):
        m = 6
        n = 10
        K_uu = lx.MatrixLinearOperator(jr.normal(getkey(), (m, m)))
        indices, values = _make_interp(n, m, 3, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        v = jr.normal(getkey(), (n,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(op, v), op.as_matrix() @ v, rtol=1e-5)

    def test_vmap(self, getkey):
        m = 6
        n = 10
        K_uu = lx.MatrixLinearOperator(jr.normal(getkey(), (m, m)))
        indices, values = _make_interp(n, m, 3, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        vs = jr.normal(getkey(), (5, n))
        results = jax.vmap(op.mv)(vs)
        assert results.shape == (5, n)
        assert tree_allclose(results[0], op.as_matrix() @ vs[0], rtol=1e-5)

    def test_grad_through_values(self, getkey):
        m = 6
        n = 10
        K_uu_mat = jr.normal(getkey(), (m, m))
        indices, _ = _make_interp(n, m, 3, getkey())
        v = jr.normal(getkey(), (n,))

        def loss(values):
            K_uu = lx.MatrixLinearOperator(K_uu_mat)
            op = InterpolatedOperator(K_uu, indices, values)
            return jnp.sum(op.mv(v) ** 2)

        values = jr.uniform(getkey(), (n, 3))
        g = jax.grad(loss)(values)
        assert g.shape == (n, 3)
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Integration with Toeplitz
# ---------------------------------------------------------------------------


class TestToeplitzIntegration:
    def test_toeplitz_ski_matches_dense(self, getkey):
        """Full SKI pipeline: Toeplitz base + interpolation."""
        m = 12
        n = 20
        p = 4
        column = jnp.abs(jr.normal(getkey(), (m,)))
        column = column.at[0].set(column[0] + 1.0)  # ensure dominant diagonal
        K_uu = Toeplitz(column)
        indices, values = _make_interp(n, m, p, getkey())
        op = InterpolatedOperator(K_uu, indices, values)
        v = jr.normal(getkey(), (n,))
        assert tree_allclose(op.mv(v), op.as_matrix() @ v, rtol=1e-4)
