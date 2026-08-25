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

    def test_reverse_mode_differentiable(self):
        """`grad` works through the retry loop (gh-229).

        The adaptive-jitter loop used ``lax.while_loop``, which has no
        reverse-mode rule and made every objective reaching
        ``safe_cholesky`` fail under ``grad``. On the no-retry path the
        gradient must match differentiating a plain Cholesky. Key pinned:
        the test checks a correctness property, any PD matrix would do.
        """

        def loss(K):
            op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
            return jnp.sum(safe_cholesky(op))

        def loss_plain(K):
            return jnp.sum(jnp.linalg.cholesky(K))

        N = 6
        A = jax.random.normal(jax.random.key(0), (N, N))
        K = A @ A.T + jnp.eye(N)

        grad = jax.grad(loss)(K)
        assert jnp.all(jnp.isfinite(grad))
        # jnp.linalg.cholesky symmetrizes its cotangent; the structured
        # route reports the raw one. Both are valid conventions for a
        # symmetric input, so compare after symmetrizing.
        sym_grad = 0.5 * (grad + grad.T)
        assert jnp.allclose(sym_grad, jax.grad(loss_plain)(K), atol=1e-8)

    def test_reverse_mode_differentiable_under_jit(self):
        """`jit(grad(...))` composes — the training-loop configuration."""

        def loss(K):
            op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
            return jnp.sum(safe_cholesky(op))

        N = 6
        A = jax.random.normal(jax.random.key(1), (N, N))
        K = A @ A.T + jnp.eye(N)
        grad = jax.jit(jax.grad(loss))(K)
        assert jnp.all(jnp.isfinite(grad))
