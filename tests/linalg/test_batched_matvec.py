"""Tests for batched kernel matvec utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from gaussx._linalg._batched_matvec import (
    batched_kernel_matvec,
    batched_kernel_rmatvec,
)
from gaussx._testing import tree_allclose


def _rbf(x, y):
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2))


def _build_dense(kernel_fn, X, Z):
    return jax.vmap(lambda x_i: jax.vmap(lambda z_j: kernel_fn(x_i, z_j))(Z))(X)


# ---------------------------------------------------------------------------
# batched_kernel_matvec
# ---------------------------------------------------------------------------


class TestBatchedKernelMatvec:
    def test_matches_dense(self, getkey):
        X = jr.normal(getkey(), (20, 3))
        Z = jr.normal(getkey(), (10, 3))
        v = jr.normal(getkey(), (10,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_matvec(_rbf, X, Z, v, batch_size=8)
        assert tree_allclose(result, K @ v, rtol=1e-5)

    def test_batch_size_1(self, getkey):
        X = jr.normal(getkey(), (8, 2))
        Z = jr.normal(getkey(), (5, 2))
        v = jr.normal(getkey(), (5,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_matvec(_rbf, X, Z, v, batch_size=1)
        assert tree_allclose(result, K @ v, rtol=1e-5)

    def test_batch_size_equals_n(self, getkey):
        N = 10
        X = jr.normal(getkey(), (N, 3))
        Z = jr.normal(getkey(), (6, 3))
        v = jr.normal(getkey(), (6,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_matvec(_rbf, X, Z, v, batch_size=N)
        assert tree_allclose(result, K @ v, rtol=1e-5)

    def test_batch_size_larger_than_n(self, getkey):
        X = jr.normal(getkey(), (5, 2))
        Z = jr.normal(getkey(), (4, 2))
        v = jr.normal(getkey(), (4,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_matvec(_rbf, X, Z, v, batch_size=32)
        assert tree_allclose(result, K @ v, rtol=1e-5)

    def test_non_divisible(self, getkey):
        """N % batch_size != 0."""
        X = jr.normal(getkey(), (11, 3))
        Z = jr.normal(getkey(), (7, 3))
        v = jr.normal(getkey(), (7,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_matvec(_rbf, X, Z, v, batch_size=4)
        assert tree_allclose(result, K @ v, rtol=1e-5)

    def test_n_equals_1(self, getkey):
        X = jr.normal(getkey(), (1, 3))
        Z = jr.normal(getkey(), (5, 3))
        v = jr.normal(getkey(), (5,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_matvec(_rbf, X, Z, v, batch_size=4)
        assert tree_allclose(result, K @ v, rtol=1e-5)

    def test_output_shape(self, getkey):
        X = jr.normal(getkey(), (15, 3))
        Z = jr.normal(getkey(), (8, 3))
        v = jr.normal(getkey(), (8,))
        result = batched_kernel_matvec(_rbf, X, Z, v, batch_size=4)
        assert result.shape == (15,)

    def test_consistent_batch_sizes(self, getkey):
        """Result should be identical regardless of batch_size."""
        X = jr.normal(getkey(), (12, 3))
        Z = jr.normal(getkey(), (6, 3))
        v = jr.normal(getkey(), (6,))
        r1 = batched_kernel_matvec(_rbf, X, Z, v, batch_size=1)
        r4 = batched_kernel_matvec(_rbf, X, Z, v, batch_size=4)
        r12 = batched_kernel_matvec(_rbf, X, Z, v, batch_size=12)
        r32 = batched_kernel_matvec(_rbf, X, Z, v, batch_size=32)
        assert tree_allclose(r1, r4, rtol=1e-5)
        assert tree_allclose(r1, r12, rtol=1e-5)
        assert tree_allclose(r1, r32, rtol=1e-5)


# ---------------------------------------------------------------------------
# batched_kernel_rmatvec
# ---------------------------------------------------------------------------


class TestBatchedKernelRmatvec:
    def test_matches_dense(self, getkey):
        X = jr.normal(getkey(), (20, 3))
        Z = jr.normal(getkey(), (10, 3))
        u = jr.normal(getkey(), (20,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_rmatvec(_rbf, X, Z, u, batch_size=8)
        assert tree_allclose(result, K.T @ u, rtol=1e-5)

    def test_batch_size_1(self, getkey):
        X = jr.normal(getkey(), (8, 2))
        Z = jr.normal(getkey(), (5, 2))
        u = jr.normal(getkey(), (8,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_rmatvec(_rbf, X, Z, u, batch_size=1)
        assert tree_allclose(result, K.T @ u, rtol=1e-5)

    def test_non_divisible(self, getkey):
        X = jr.normal(getkey(), (11, 3))
        Z = jr.normal(getkey(), (7, 3))
        u = jr.normal(getkey(), (11,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_rmatvec(_rbf, X, Z, u, batch_size=4)
        assert tree_allclose(result, K.T @ u, rtol=1e-5)

    def test_n_equals_1(self, getkey):
        X = jr.normal(getkey(), (1, 3))
        Z = jr.normal(getkey(), (5, 3))
        u = jr.normal(getkey(), (1,))
        K = _build_dense(_rbf, X, Z)
        result = batched_kernel_rmatvec(_rbf, X, Z, u, batch_size=4)
        assert tree_allclose(result, K.T @ u, rtol=1e-5)

    def test_output_shape(self, getkey):
        X = jr.normal(getkey(), (15, 3))
        Z = jr.normal(getkey(), (8, 3))
        u = jr.normal(getkey(), (15,))
        result = batched_kernel_rmatvec(_rbf, X, Z, u, batch_size=4)
        assert result.shape == (8,)

    def test_gradient_flows(self, getkey):
        """Gradient should flow through kernel hyperparameters (via closure)."""
        X = jr.normal(getkey(), (8, 2))
        Z = jr.normal(getkey(), (5, 2))
        u = jr.normal(getkey(), (8,))

        def loss(ls):
            def k(x, y):
                diff = x - y
                return jnp.exp(-0.5 * jnp.sum(diff**2) / ls**2)

            return jnp.sum(batched_kernel_rmatvec(k, X, Z, u, batch_size=4) ** 2)

        g = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(g)
