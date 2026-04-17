"""Tests for base_conditional — Gaussian conditional via Schur complement."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsla

from gaussx._gp._base_conditional import base_conditional
from gaussx._testing import tree_allclose


def _make_pd(key, M):
    A = jr.normal(key, (M, M))
    return A @ A.T + 0.1 * jnp.eye(M)


# ---------------------------------------------------------------------------
# Prior conditional (no q_sqrt)
# ---------------------------------------------------------------------------


class TestPriorConditional:
    def test_mean_shape(self, getkey):
        M, N, R = 5, 8, 2
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1
        f = jr.normal(getkey(), (M, R))
        mean, var = base_conditional(K_mm, K_mn, K_nn_diag, f)
        assert mean.shape == (N, R)
        assert var.shape == (N, R)

    def test_mean_matches_dense(self, getkey):
        """Mean should be K_nm K_mm^{-1} f."""
        M, N, R = 5, 8, 1
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        f = jr.normal(getkey(), (M, R))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1

        mean, _ = base_conditional(K_mm, K_mn, K_nn_diag, f)
        expected = K_mn.T @ jnp.linalg.solve(K_mm, f)
        assert tree_allclose(mean, expected, rtol=1e-4)

    def test_var_diagonal_knn(self, getkey):
        """Variance with diagonal K_nn."""
        M, N = 4, 6
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, 1))

        _, var = base_conditional(K_mm, K_mn, K_nn_diag, f)

        # Expected: K_nn_diag - diag(K_nm K_mm^{-1} K_mn)
        A = jnp.linalg.solve(K_mm, K_mn)  # (M, N)
        schur_diag = jnp.sum(K_mn * A, axis=0)
        expected = K_nn_diag - schur_diag
        assert tree_allclose(var[:, 0], expected, rtol=1e-4)

    def test_var_full_knn(self, getkey):
        """Variance with full K_nn."""
        M, N = 4, 6
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn = _make_pd(getkey(), N)
        f = jr.normal(getkey(), (M, 1))

        _, var = base_conditional(K_mm, K_mn, K_nn, f)
        assert var.shape == (N, N, 1)

        A = jnp.linalg.solve(K_mm, K_mn)
        expected = K_nn - K_mn.T @ A
        assert tree_allclose(var[:, :, 0], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Whitened parameterization
# ---------------------------------------------------------------------------


class TestWhitened:
    def test_mean_whitened(self, getkey):
        """Whitened: mean = A^T f where A = L^{-1} K_mn."""
        M, N, R = 5, 8, 1
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1
        f = jr.normal(getkey(), (M, R))

        mean, _ = base_conditional(K_mm, K_mn, K_nn_diag, f, white=True)
        L = jnp.linalg.cholesky(K_mm)
        A = jsla.solve_triangular(L, K_mn, lower=True)
        expected = A.T @ f
        assert tree_allclose(mean, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# With q_sqrt (variational posterior)
# ---------------------------------------------------------------------------


class TestVariational:
    def test_diagonal_q_sqrt(self, getkey):
        M, N, R = 5, 8, 2
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, R))
        q_diag = jnp.abs(jr.normal(getkey(), (M, R))) + 0.1

        mean, var = base_conditional(K_mm, K_mn, K_nn_diag, f, q_sqrt=q_diag)
        assert mean.shape == (N, R)
        assert var.shape == (N, R)

        # Verify variance is adjusted from prior conditional
        _, var_prior = base_conditional(K_mm, K_mn, K_nn_diag, f)
        # Variational variance should differ from prior (unless q_sqrt=0)
        assert not jnp.allclose(var, var_prior)

    def test_full_q_sqrt(self, getkey):
        M, N, R = 4, 6, 2
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, R))

        q_sqrt_list = []
        for _ in range(R):
            L = jnp.tril(jr.normal(getkey(), (M, M)))
            L = L.at[jnp.diag_indices(M)].set(jnp.abs(jnp.diag(L)) + 0.1)
            q_sqrt_list.append(L)
        q_sqrt = jnp.stack(q_sqrt_list, axis=0)  # (R, M, M)

        mean, var = base_conditional(K_mm, K_mn, K_nn_diag, f, q_sqrt=q_sqrt)
        assert mean.shape == (N, R)
        assert var.shape == (N, R)

    def test_full_q_sqrt_full_knn(self, getkey):
        """Full q_sqrt with full K_nn should give (N, N, R) variance."""
        M, N, R = 4, 6, 2
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn = _make_pd(getkey(), N)
        f = jr.normal(getkey(), (M, R))

        q_sqrt_list = []
        for _ in range(R):
            L = jnp.tril(jr.normal(getkey(), (M, M)))
            L = L.at[jnp.diag_indices(M)].set(jnp.abs(jnp.diag(L)) + 0.1)
            q_sqrt_list.append(L)
        q_sqrt = jnp.stack(q_sqrt_list, axis=0)

        mean, var = base_conditional(K_mm, K_mn, K_nn, f, q_sqrt=q_sqrt)
        assert mean.shape == (N, R)
        assert var.shape == (N, N, R)

    def test_variance_formula_diagonal(self, getkey):
        """Verify the variance formula with diagonal q_sqrt."""
        M, N, R = 4, 6, 1
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, R))
        q_diag = jnp.abs(jr.normal(getkey(), (M, R))) + 0.1

        _, var = base_conditional(K_mm, K_mn, K_nn_diag, f, q_sqrt=q_diag)

        Kmm_inv = jnp.linalg.inv(K_mm)
        q_cov = jnp.diag(q_diag[:, 0] ** 2)
        schur = K_mn.T @ Kmm_inv @ K_mn
        var_adj = jnp.diag(K_mn.T @ Kmm_inv @ q_cov @ Kmm_inv @ K_mn)
        expected = K_nn_diag - jnp.diag(schur) + var_adj
        assert tree_allclose(var[:, 0], expected, rtol=1e-4)

    def test_variance_formula_diagonal_nonwhite(self, getkey):
        """Non-whitened q_sqrt should include the prior solve."""
        M, N, R = 4, 6, 1
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, R))
        q_diag = jnp.abs(jr.normal(getkey(), (M, R))) + 0.1

        _, var = base_conditional(K_mm, K_mn, K_nn_diag, f, q_sqrt=q_diag)

        Kmm_inv = jnp.linalg.inv(K_mm)
        q_cov = jnp.diag(q_diag[:, 0] ** 2)
        schur = K_mn.T @ Kmm_inv @ K_mn
        var_adj = jnp.diag(K_mn.T @ Kmm_inv @ q_cov @ Kmm_inv @ K_mn)
        expected = K_nn_diag - jnp.diag(schur) + var_adj
        assert tree_allclose(var[:, 0], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


class TestGradient:
    def test_grad_through_f(self, getkey):
        M, N = 5, 8
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1

        def loss(f):
            mean, var = base_conditional(K_mm, K_mn, K_nn_diag, f)
            return jnp.sum(mean**2) + jnp.sum(var)

        f = jr.normal(getkey(), (M, 1))
        g = jax.grad(loss)(f)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == (M, 1)
