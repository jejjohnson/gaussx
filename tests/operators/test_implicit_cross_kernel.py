"""Tests for the ImplicitCrossKernelOperator (rectangular cross-kernel)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import ImplicitCrossKernelOperator, implicit_cross_kernel
from gaussx._testing import tree_allclose


def _rbf(x, y):
    """Simple RBF kernel with unit hyperparameters."""
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2))


def _build_dense(kernel_fn, X1, X2):
    return jax.vmap(lambda x_i: jax.vmap(lambda x_j: kernel_fn(x_i, x_j))(X2))(X1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, getkey):
        X = jr.normal(getkey(), (20, 3))
        Z = jr.normal(getkey(), (8, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z)
        assert op.in_size() == 8
        assert op.out_size() == 20

    def test_custom_batch_size(self, getkey):
        X = jr.normal(getkey(), (20, 3))
        Z = jr.normal(getkey(), (8, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=5)
        assert op.batch_size == 5

    def test_convenience_function(self, getkey):
        X = jr.normal(getkey(), (15, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = implicit_cross_kernel(_rbf, X, Z, batch_size=4)
        assert isinstance(op, ImplicitCrossKernelOperator)
        assert op.in_size() == 6
        assert op.out_size() == 15


# ---------------------------------------------------------------------------
# mv correctness
# ---------------------------------------------------------------------------


class TestMv:
    def test_mv_matches_dense(self, getkey):
        X = jr.normal(getkey(), (12, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=4)
        v = jr.normal(getkey(), (6,))
        K_dense = _build_dense(_rbf, X, Z)
        assert tree_allclose(op.mv(v), K_dense @ v, rtol=1e-5)

    def test_mv_batch_size_1(self, getkey):
        """Extreme batch_size=1 should still be correct."""
        X = jr.normal(getkey(), (8, 2))
        Z = jr.normal(getkey(), (5, 2))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=1)
        v = jr.normal(getkey(), (5,))
        K_dense = _build_dense(_rbf, X, Z)
        assert tree_allclose(op.mv(v), K_dense @ v, rtol=1e-5)

    def test_mv_batch_size_equals_n(self, getkey):
        """batch_size = N: single batch, no padding needed."""
        N, M = 10, 6
        X = jr.normal(getkey(), (N, 3))
        Z = jr.normal(getkey(), (M, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=N)
        v = jr.normal(getkey(), (M,))
        K_dense = _build_dense(_rbf, X, Z)
        assert tree_allclose(op.mv(v), K_dense @ v, rtol=1e-5)

    def test_mv_batch_size_larger_than_n(self, getkey):
        """batch_size > N: everything in one padded batch."""
        N, M = 5, 4
        X = jr.normal(getkey(), (N, 2))
        Z = jr.normal(getkey(), (M, 2))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=32)
        v = jr.normal(getkey(), (M,))
        K_dense = _build_dense(_rbf, X, Z)
        assert tree_allclose(op.mv(v), K_dense @ v, rtol=1e-5)

    def test_mv_non_divisible_batch_size(self, getkey):
        """N not divisible by batch_size: padding is correctly handled."""
        N, M = 11, 7
        X = jr.normal(getkey(), (N, 3))
        Z = jr.normal(getkey(), (M, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=4)
        v = jr.normal(getkey(), (M,))
        K_dense = _build_dense(_rbf, X, Z)
        assert tree_allclose(op.mv(v), K_dense @ v, rtol=1e-5)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


class TestAsMatrix:
    def test_as_matrix_matches_manual(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z)
        K_dense = _build_dense(_rbf, X, Z)
        assert tree_allclose(op.as_matrix(), K_dense, rtol=1e-5)

    def test_as_matrix_shape(self, getkey):
        N, M = 12, 8
        X = jr.normal(getkey(), (N, 3))
        Z = jr.normal(getkey(), (M, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z)
        assert op.as_matrix().shape == (N, M)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_transpose_matrix(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=4)
        assert tree_allclose(op.T.as_matrix(), op.as_matrix().T, rtol=1e-5)

    def test_transpose_mv(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=4)
        u = jr.normal(getkey(), (10,))
        K_dense = _build_dense(_rbf, X, Z)
        assert tree_allclose(op.T.mv(u), K_dense.T @ u, rtol=1e-5)

    def test_transpose_sizes(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z)
        assert op.T.in_size() == 10
        assert op.T.out_size() == 6


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_not_symmetric_by_default(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z)
        assert lx.is_symmetric(op) is False

    def test_symmetric_when_tagged(self, getkey):
        X = jr.normal(getkey(), (8, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, X, tags=lx.symmetric_tag)
        assert lx.is_symmetric(op) is True

    def test_not_diagonal(self, getkey):
        X = jr.normal(getkey(), (8, 3))
        Z = jr.normal(getkey(), (5, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z)
        assert lx.is_diagonal(op) is False


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


class TestJAX:
    def test_jit(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=4)
        v = jr.normal(getkey(), (6,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(op, v), op.as_matrix() @ v, rtol=1e-5)

    def test_vmap(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=4)
        vs = jr.normal(getkey(), (5, 6))
        results = jax.vmap(op.mv)(vs)
        assert results.shape == (5, 10)
        assert tree_allclose(results[0], op.as_matrix() @ vs[0], rtol=1e-5)

    def test_grad_through_inducing(self, getkey):
        """Gradients flow through X_inducing."""
        X = jr.normal(getkey(), (8, 2))
        v = jr.normal(getkey(), (5,))

        def loss(Z):
            op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=4)
            return jnp.sum(op.mv(v) ** 2)

        Z = jr.normal(getkey(), (5, 2))
        g = jax.grad(loss)(Z)
        assert g.shape == (5, 2)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_data(self, getkey):
        """Gradients flow through X_data."""
        Z = jr.normal(getkey(), (5, 2))
        v = jr.normal(getkey(), (5,))

        def loss(X):
            op = ImplicitCrossKernelOperator(_rbf, X, Z, batch_size=4)
            return jnp.sum(op.mv(v) ** 2)

        X = jr.normal(getkey(), (8, 2))
        g = jax.grad(loss)(X)
        assert g.shape == (8, 2)
        assert jnp.all(jnp.isfinite(g))
