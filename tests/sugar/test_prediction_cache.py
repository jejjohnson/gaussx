"""Tests for prediction cache."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx import build_prediction_cache, predict_mean, predict_variance


class TestPredictionCache:
    def test_alpha_matches_solve(self, getkey):
        """Cached alpha matches direct solve."""
        N = 10
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        y = jax.random.normal(getkey(), (N,))
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        cache = build_prediction_cache(op, y)
        expected = jnp.linalg.solve(K, y)
        assert jnp.allclose(cache.alpha, expected, atol=1e-5)

    def test_predict_mean(self, getkey):
        """Predictive mean matches K_cross @ K_inv @ y."""
        N, Nt = 10, 3
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        y = jax.random.normal(getkey(), (N,))
        K_cross = jax.random.normal(getkey(), (Nt, N))
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        cache = build_prediction_cache(op, y)
        mu = predict_mean(cache, K_cross)
        expected = K_cross @ jnp.linalg.solve(K, y)
        assert jnp.allclose(mu, expected, atol=1e-5)
        assert mu.shape == (Nt,)

    def test_predict_variance(self, getkey):
        """Predictive variance matches exact computation."""
        N, Nt = 8, 4
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        y = jax.random.normal(getkey(), (N,))
        K_cross = jax.random.normal(getkey(), (Nt, N))
        K_test_diag = jnp.ones(Nt) * 2.0
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        cache = build_prediction_cache(op, y)
        var = predict_variance(cache, K_cross, K_test_diag, op)
        # Expected: k_** - diag(K_cross @ K_inv @ K_cross.T)
        K_inv = jnp.linalg.inv(K)
        expected = K_test_diag - jnp.sum(K_cross @ K_inv * K_cross, axis=1)
        assert jnp.allclose(var, expected, atol=1e-4)
        assert var.shape == (Nt,)

    def test_pytree_compatible(self, getkey):
        """Cache is a valid JAX pytree."""
        N = 5
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        y = jax.random.normal(getkey(), (N,))
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        cache = build_prediction_cache(op, y)
        leaves = jax.tree_util.tree_leaves(cache)
        assert len(leaves) == 1  # just alpha
