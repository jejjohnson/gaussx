"""Tests for LOVE — Lanczos Variance Estimates."""

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._gp._love import (
    love_cache,
    love_residual,
    love_variance,
    love_variance_error_bound,
)
from gaussx._testing import psd_operator, random_pd_matrix


def _rbf_problem():
    """1-D RBF GP, N=80 points on [0, 10], noise 0.01, test point 4.37 (gh-293)."""
    x = jnp.linspace(0.0, 10.0, 80)
    K = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2) + 0.01 * jnp.eye(80)
    k_star = jnp.exp(-0.5 * (x - 4.37) ** 2)
    exact_var = 1.01 - k_star @ jnp.linalg.solve(K, k_star)
    return psd_operator(K), k_star, exact_var


class TestLOVECache:
    @pytest.mark.slow
    def test_cache_shapes(self):
        """Cache should have correct shapes."""
        N, k = 20, 10
        K_op = psd_operator(random_pd_matrix(jr.key(0), N))

        cache = love_cache(K_op, lanczos_order=k, key=jr.key(1))
        assert cache.Q.shape == (N, k)
        assert cache.inv_eigvals.shape == (k,)

    @pytest.mark.slow
    def test_cache_order_clamped(self):
        """Lanczos order should be clamped to N."""
        N = 5
        K_op = psd_operator(random_pd_matrix(jr.key(0), N))

        cache = love_cache(K_op, lanczos_order=100, key=jr.key(1))
        assert cache.Q.shape == (N, N)
        assert cache.inv_eigvals.shape == (N,)


class TestLOVEVariance:
    @pytest.mark.slow
    def test_approximates_true_variance(self):
        """At lanczos_order=N the cache is an exact factorisation of K^{-1}."""
        N = 30
        # jitter 0.5: cond(K) ~ 1e2-1e3 at N=30, so 1e-8 is comfortable in float64.
        K = random_pd_matrix(jr.key(0), N, jitter=0.5)
        k_star = jr.normal(jr.key(1), (N,))

        cache = love_cache(psd_operator(K), lanczos_order=N, key=jr.key(2))
        approx = love_variance(cache, k_star)

        exact = k_star @ jnp.linalg.solve(K, k_star)
        assert jnp.allclose(approx, exact, rtol=1e-8)

    @pytest.mark.slow
    def test_nonnegative(self):
        """LOVE variance should be non-negative."""
        N = 15
        K_op = psd_operator(random_pd_matrix(jr.key(0), N, jitter=0.2))
        k_star = jr.normal(jr.key(1), (N,))

        cache = love_cache(K_op, lanczos_order=N, key=jr.key(2))
        v = love_variance(cache, k_star)
        assert v >= -1e-6  # Allow small numerical error

    @pytest.mark.slow
    def test_full_rank_exact(self):
        """With full Lanczos order, should be exact."""
        N = 8
        K = random_pd_matrix(jr.key(0), N, jitter=0.3)
        k_star = jr.normal(jr.key(1), (N,))

        cache = love_cache(psd_operator(K), lanczos_order=N, key=jr.key(2))
        approx = love_variance(cache, k_star)

        exact = k_star @ jnp.linalg.solve(K, k_star)
        assert jnp.allclose(approx, exact, rtol=1e-8)

    def test_jit(self):
        """Should be JIT-compatible."""
        N = 10
        K = jnp.eye(N) + 0.1 * jnp.ones((N, N))
        K_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        k_star = jr.normal(jr.key(0), (N,))

        cache = love_cache(K_op, lanczos_order=N, key=jr.key(1))
        v1 = love_variance(cache, k_star)
        v2 = jax.jit(love_variance)(cache, k_star)
        assert jnp.allclose(v1, v2, atol=1e-10)


class TestLOVEConvergence:
    """gh-293: lanczos_order < N biases the variance high; love_residual flags it."""

    @pytest.mark.x64_only(reason="rtol=1e-6 against a float64 dense solve")
    def test_converged_order_matches_dense(self):
        K_op, k_star, exact_var = _rbf_problem()
        cache = love_cache(K_op, lanczos_order=20, key=jr.key(0))
        var = 1.01 - love_variance(cache, k_star)
        assert jnp.allclose(var, exact_var, rtol=1e-6)
        # Observed 2e-7 at k=20 (5e-9 at k=25); 1e-3 is the documented threshold.
        assert love_residual(cache, K_op, k_star) < 1e-3

    @pytest.mark.x64_only(reason="one-signed bias checked against float64 solve")
    @pytest.mark.parametrize("order", [5, 10])
    def test_truncated_order_overestimates_and_is_flagged(self, order):
        K_op, k_star, exact_var = _rbf_problem()
        cache = love_cache(K_op, lanczos_order=order, key=jr.key(0))
        var = 1.01 - love_variance(cache, k_star)
        # K^{-1} - Q T^{-1} Q^T is PSD: the predictive variance is an upper bound.
        assert var >= exact_var
        assert var > 1.1 * exact_var
        assert love_residual(cache, K_op, k_star) > 1e-3

    def test_initial_vector(self):
        """A deterministic start vector replaces the key and is reproducible."""
        K_op, k_star, _ = _rbf_problem()
        v0 = jnp.sin(jnp.linspace(0.0, 10.0, 80))
        c1 = love_cache(K_op, lanczos_order=30, initial_vector=v0)
        c2 = love_cache(K_op, lanczos_order=30, key=jr.key(5), initial_vector=v0)
        assert jnp.array_equal(c1.Q, c2.Q)
        assert love_residual(c1, K_op, k_star) < 1e-3


class TestLOVEEdgeCases:
    """Review follow-ups on gh-293."""

    @pytest.mark.parametrize(
        "K_op",
        [
            pytest.param(
                lx.DiagonalLinearOperator(2.0 * jnp.ones(6)), id="diag-repeated"
            ),
            pytest.param(
                lx.DiagonalLinearOperator(jnp.array([3.0, 1.0, 1.0, 2.0, 2.0, 0.5])),
                id="diag",
            ),
        ],
    )
    def test_diagonal_is_exact_at_full_order(self, K_op):
        """A single Krylov vector cannot span repeated eigenvalues; no Lanczos."""
        k_star = jnp.arange(1.0, 7.0)
        cache = love_cache(K_op, lanczos_order=6)
        exact = jnp.sum(k_star**2 / lx.diagonal(K_op))
        assert jnp.all(jnp.isfinite(cache.Q))
        assert jnp.allclose(love_variance(cache, k_star), exact, rtol=1e-6)

    def test_zero_initial_vector_is_replaced(self):
        K_op, k_star, _ = _rbf_problem()
        cache = love_cache(K_op, lanczos_order=30, initial_vector=jnp.zeros(80))
        assert jnp.all(jnp.isfinite(cache.Q))
        assert jnp.isfinite(love_variance(cache, k_star))

    def test_zero_cross_covariance_residual_is_zero(self):
        K_op, _, _ = _rbf_problem()
        cache = love_cache(K_op, lanczos_order=10, key=jr.key(0))
        assert love_residual(cache, K_op, jnp.zeros(80)) == 0.0

    @pytest.mark.x64_only(reason="bound checked against a float64 dense solve")
    @pytest.mark.parametrize("order", [5, 10, 20])
    def test_error_bound_holds(self, order):
        """0 <= true variance error <= ||r||² / λ_min, with λ_min >= σ² = 0.01."""
        K_op, k_star, exact_var = _rbf_problem()
        cache = love_cache(K_op, lanczos_order=order, key=jr.key(0))
        err = (1.01 - love_variance(cache, k_star)) - exact_var
        bound = love_variance_error_bound(cache, K_op, k_star, 0.01)
        assert -1e-12 <= err <= bound * (1 + 1e-8) + 1e-14
        if order == 20:
            assert bound < 1e-6
