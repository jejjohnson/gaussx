"""Tests for mixed-precision stable squared distances and RBF kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from gaussx._sugar._mixed_precision import stable_rbf_kernel, stable_squared_distances
from gaussx._testing import tree_allclose


# ---------------------------------------------------------------------------
# stable_squared_distances
# ---------------------------------------------------------------------------


class TestStableSquaredDistances:
    def test_matches_naive_float64(self, getkey):
        """Should match direct ||x - z||^2 in float64."""
        X = jr.normal(getkey(), (10, 5)).astype(jnp.float64)
        Z = jr.normal(getkey(), (8, 5)).astype(jnp.float64)
        expected = jnp.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        result = stable_squared_distances(
            X,
            Z,
            compute_dtype=jnp.float64,
            accumulate_dtype=jnp.float64,
        )
        assert tree_allclose(result, expected, rtol=1e-10)

    def test_non_negative_high_dim(self, getkey):
        """All distances should be >= 0 even in float32 with D=1000."""
        D = 1000
        X = jr.normal(getkey(), (50, D)).astype(jnp.float32)
        Z = jr.normal(getkey(), (50, D)).astype(jnp.float32)
        dist_sq = stable_squared_distances(X, Z)
        assert jnp.all(dist_sq >= 0.0)

    def test_self_distance_zero(self, getkey):
        """Distance of a point to itself should be ~0."""
        X = jr.normal(getkey(), (5, 10)).astype(jnp.float32)
        dist_sq = stable_squared_distances(X, X)
        diag = jnp.diag(dist_sq)
        assert jnp.allclose(diag, 0.0, atol=1e-5)

    def test_output_shape(self, getkey):
        X = jr.normal(getkey(), (7, 3))
        Z = jr.normal(getkey(), (4, 3))
        assert stable_squared_distances(X, Z).shape == (7, 4)

    def test_symmetric(self, getkey):
        """D(X, Z) should equal D(Z, X)^T."""
        X = jr.normal(getkey(), (6, 4))
        Z = jr.normal(getkey(), (8, 4))
        d1 = stable_squared_distances(X, Z)
        d2 = stable_squared_distances(Z, X)
        assert tree_allclose(d1, d2.T, rtol=1e-5)


# ---------------------------------------------------------------------------
# stable_rbf_kernel
# ---------------------------------------------------------------------------


class TestStableRBFKernel:
    def test_matches_naive(self, getkey):
        """Should match naive RBF computation in float64."""
        X = jr.normal(getkey(), (6, 3)).astype(jnp.float64)
        Z = jr.normal(getkey(), (4, 3)).astype(jnp.float64)
        ls = 1.5
        var = 2.0
        dist_sq = jnp.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        expected = var * jnp.exp(-0.5 * dist_sq / ls**2)
        result = stable_rbf_kernel(
            X,
            Z,
            ls,
            var,
            compute_dtype=jnp.float64,
            accumulate_dtype=jnp.float64,
        )
        assert tree_allclose(result, expected, rtol=1e-10)

    def test_output_shape(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        K = stable_rbf_kernel(X, Z, lengthscale=1.0)
        assert K.shape == (10, 6)

    def test_psd_high_dim(self, getkey):
        """RBF kernel eigenvalues should all be >= 0 for D=500."""
        D = 500
        X = jr.normal(getkey(), (30, D)).astype(jnp.float32)
        K = stable_rbf_kernel(X, X, lengthscale=1.0)
        eigvals = jnp.linalg.eigvalsh(K.astype(jnp.float64))
        assert jnp.all(eigvals > -1e-5)

    def test_cholesky_float32_high_dim(self, getkey):
        """Cholesky should succeed in float32 for D=500."""
        D = 500
        X = jr.normal(getkey(), (20, D)).astype(jnp.float32)
        K = stable_rbf_kernel(X, X, lengthscale=1.0)
        K = K + 1e-4 * jnp.eye(20)  # small jitter for numerics
        L = jnp.linalg.cholesky(K)
        assert jnp.all(jnp.isfinite(L))

    def test_grad_lengthscale(self, getkey):
        """Gradient should flow through lengthscale."""
        X = jr.normal(getkey(), (6, 3))
        Z = jr.normal(getkey(), (4, 3))

        def loss(ls):
            K = stable_rbf_kernel(X, Z, lengthscale=ls)
            return jnp.sum(K)

        g = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert g != 0.0
