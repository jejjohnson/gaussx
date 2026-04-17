"""Tests for safe_cholesky."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx import safe_cholesky


class TestSafeCholesky:
    def test_well_conditioned(self, getkey):
        """Succeeds with no jitter on well-conditioned matrix."""
        N = 10
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        L = safe_cholesky(op)
        assert not jnp.any(jnp.isnan(L))
        assert jnp.allclose(L @ L.T, K, atol=1e-5)

    def test_ill_conditioned(self, getkey):
        """Succeeds on ill-conditioned matrix via jitter."""
        N = 10
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T  # no diagonal boost — may be near-singular
        K = K + 1e-12 * jnp.eye(N)  # barely PD
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        L = safe_cholesky(op, initial_jitter=1e-6)
        assert not jnp.any(jnp.isnan(L))

    def test_lower_triangular(self, getkey):
        """Result is lower triangular."""
        N = 8
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        L = safe_cholesky(op)
        assert jnp.allclose(L, jnp.tril(L))

    def test_jit_compatible(self, getkey):
        """Works under jax.jit."""
        N = 6
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        L = jax.jit(safe_cholesky)(op)
        assert not jnp.any(jnp.isnan(L))
        assert jnp.allclose(L @ L.T, K, atol=1e-5)
