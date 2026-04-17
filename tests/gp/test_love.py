"""Tests for LOVE — Lanczos Variance Estimates."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._gp._love import love_cache, love_variance


class TestLOVECache:
    def test_cache_shapes(self, getkey):
        """Cache should have correct shapes."""
        N = 20
        k = 10
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T + 0.1 * jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)

        cache = love_cache(K_op, lanczos_order=k, key=getkey())
        assert cache.Q.shape == (N, k)
        assert cache.inv_eigvals.shape == (k,)

    def test_cache_order_clamped(self, getkey):
        """Lanczos order should be clamped to N."""
        N = 5
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T + 0.1 * jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)

        cache = love_cache(K_op, lanczos_order=100, key=getkey())
        assert cache.Q.shape == (N, N)
        assert cache.inv_eigvals.shape == (N,)


class TestLOVEVariance:
    def test_approximates_true_variance(self, getkey):
        """LOVE variance should approximate k^T K^{-1} k."""
        N = 30
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T + 0.5 * jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        k_star = jax.random.normal(getkey(), (N,))

        cache = love_cache(K_op, lanczos_order=N, key=getkey())
        approx = love_variance(cache, k_star)

        # Dense reference
        K_inv = jnp.linalg.inv(K)
        exact = k_star @ K_inv @ k_star
        assert jnp.allclose(approx, exact, rtol=0.05)

    def test_nonnegative(self, getkey):
        """LOVE variance should be non-negative."""
        N = 15
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T + 0.2 * jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        k_star = jax.random.normal(getkey(), (N,))

        cache = love_cache(K_op, lanczos_order=N, key=getkey())
        v = love_variance(cache, k_star)
        assert v >= -1e-6  # Allow small numerical error

    def test_full_rank_exact(self, getkey):
        """With full Lanczos order, should be exact."""
        N = 8
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T + 0.3 * jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        k_star = jax.random.normal(getkey(), (N,))

        cache = love_cache(K_op, lanczos_order=N, key=getkey())
        approx = love_variance(cache, k_star)

        K_inv = jnp.linalg.inv(K)
        exact = k_star @ K_inv @ k_star
        assert jnp.allclose(approx, exact, atol=1e-4)

    def test_jit(self, getkey):
        """Should be JIT-compatible."""
        N = 10
        K = jnp.eye(N) + 0.1 * jnp.ones((N, N))
        K_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        k_star = jax.random.normal(getkey(), (N,))

        cache = love_cache(K_op, lanczos_order=N, key=getkey())
        v1 = love_variance(cache, k_star)
        v2 = jax.jit(love_variance)(cache, k_star)
        assert jnp.allclose(v1, v2, atol=1e-10)
