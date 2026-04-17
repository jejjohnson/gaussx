"""Tests for gauss_kl — KL divergence between Gaussians."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from gaussx._gp._gauss_kl import gauss_kl
from gaussx._testing import tree_allclose


def _naive_kl(q_mu, q_cov, p_cov):
    """Naive KL[N(q_mu, q_cov) || N(0, p_cov)] via dense matrices."""
    M = q_mu.shape[0]
    p_cov_inv = jnp.linalg.inv(p_cov)
    logdet_p = jnp.linalg.slogdet(p_cov)[1]
    logdet_q = jnp.linalg.slogdet(q_cov)[1]
    trace_term = jnp.trace(p_cov_inv @ q_cov)
    mahal = q_mu @ p_cov_inv @ q_mu
    return 0.5 * (logdet_p - logdet_q - M + trace_term + mahal)


# ---------------------------------------------------------------------------
# White prior (K = I)
# ---------------------------------------------------------------------------


class TestWhitePrior:
    def test_full_q_sqrt_single_output(self, getkey):
        M, R = 5, 1
        q_mu = jr.normal(getkey(), (M, R))
        L = jr.normal(getkey(), (M, M))
        L = jnp.tril(L)
        L = L.at[jnp.diag_indices(M)].set(jnp.abs(jnp.diag(L)) + 0.1)
        q_sqrt = L[None, :, :]  # (1, M, M)

        result = gauss_kl(q_mu, q_sqrt, K=None)
        q_cov = L @ L.T
        expected = _naive_kl(q_mu[:, 0], q_cov, jnp.eye(M))
        assert tree_allclose(result, expected, rtol=1e-4)

    def test_diagonal_q_sqrt(self, getkey):
        M, R = 6, 1
        q_mu = jr.normal(getkey(), (M, R))
        q_diag = jnp.abs(jr.normal(getkey(), (M, R))) + 0.1
        result = gauss_kl(q_mu, q_diag, K=None)

        q_cov = jnp.diag(q_diag[:, 0] ** 2)
        expected = _naive_kl(q_mu[:, 0], q_cov, jnp.eye(M))
        assert tree_allclose(result, expected, rtol=1e-4)

    def test_zero_mean_full(self, getkey):
        """KL with zero mean should be non-negative."""
        M = 4
        q_mu = jnp.zeros((M, 1))
        L = jnp.eye(M) * 0.8
        q_sqrt = L[None, :, :]
        result = gauss_kl(q_mu, q_sqrt, K=None)
        assert result >= -1e-6

    def test_identity_posterior_gives_zero(self):
        """KL[N(0, I) || N(0, I)] = 0."""
        M = 4
        q_mu = jnp.zeros((M, 1))
        q_sqrt = jnp.eye(M)[None, :, :]  # (1, M, M)
        result = gauss_kl(q_mu, q_sqrt, K=None)
        assert tree_allclose(result, jnp.array(0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Non-white prior (K given)
# ---------------------------------------------------------------------------


class TestNonWhitePrior:
    def test_full_q_sqrt_with_prior(self, getkey):
        M, R = 5, 1
        q_mu = jr.normal(getkey(), (M, R))
        L_q = jnp.tril(jr.normal(getkey(), (M, M)))
        L_q = L_q.at[jnp.diag_indices(M)].set(jnp.abs(jnp.diag(L_q)) + 0.1)
        q_sqrt = L_q[None, :, :]

        # Random PD prior
        A = jr.normal(getkey(), (M, M))
        K = A @ A.T + 0.1 * jnp.eye(M)

        result = gauss_kl(q_mu, q_sqrt, K=K)
        q_cov = L_q @ L_q.T
        expected = _naive_kl(q_mu[:, 0], q_cov, K)
        assert tree_allclose(result, expected, rtol=1e-4)

    def test_diagonal_q_sqrt_with_prior(self, getkey):
        M, R = 6, 1
        q_mu = jr.normal(getkey(), (M, R))
        q_diag = jnp.abs(jr.normal(getkey(), (M, R))) + 0.1

        A = jr.normal(getkey(), (M, M))
        K = A @ A.T + 0.1 * jnp.eye(M)

        result = gauss_kl(q_mu, q_diag, K=K)
        q_cov = jnp.diag(q_diag[:, 0] ** 2)
        expected = _naive_kl(q_mu[:, 0], q_cov, K)
        assert tree_allclose(result, expected, rtol=1e-4)

    def test_non_negative(self, getkey):
        """KL divergence should always be >= 0."""
        M = 5
        q_mu = jr.normal(getkey(), (M, 1))
        L_q = jnp.tril(jr.normal(getkey(), (M, M)))
        L_q = L_q.at[jnp.diag_indices(M)].set(jnp.abs(jnp.diag(L_q)) + 0.1)
        q_sqrt = L_q[None, :, :]
        A = jr.normal(getkey(), (M, M))
        K = A @ A.T + 0.1 * jnp.eye(M)
        result = gauss_kl(q_mu, q_sqrt, K=K)
        assert result >= -1e-5


# ---------------------------------------------------------------------------
# Multiple outputs
# ---------------------------------------------------------------------------


class TestMultipleOutputs:
    def test_multi_output_full(self, getkey):
        """KL summed over R outputs should match sum of per-output KLs."""
        M, R = 4, 3
        q_mu = jr.normal(getkey(), (M, R))

        # Build R different Cholesky factors
        q_sqrt_list = []
        total_expected = 0.0
        for r in range(R):
            L = jnp.tril(jr.normal(getkey(), (M, M)))
            L = L.at[jnp.diag_indices(M)].set(jnp.abs(jnp.diag(L)) + 0.1)
            q_sqrt_list.append(L)
            q_cov_r = L @ L.T
            total_expected += _naive_kl(q_mu[:, r], q_cov_r, jnp.eye(M))

        q_sqrt = jnp.stack(q_sqrt_list, axis=0)  # (R, M, M)
        result = gauss_kl(q_mu, q_sqrt, K=None)
        assert tree_allclose(result, total_expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


class TestGradient:
    def test_grad_flows(self, getkey):
        M = 4
        q_mu = jr.normal(getkey(), (M, 1))

        def loss(q_mu):
            q_sqrt = jnp.eye(M)[None, :, :]
            return gauss_kl(q_mu, q_sqrt, K=None)

        g = jax.grad(loss)(q_mu)
        assert jnp.all(jnp.isfinite(g))
