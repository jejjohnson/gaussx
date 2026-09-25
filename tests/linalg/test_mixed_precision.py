"""Tests for mixed-precision stable squared distances."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from gaussx._linalg._mixed_precision import stable_squared_distances
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
