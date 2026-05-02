"""Tests for 3-parameterization conversions (mean/var, natural, expectation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from gaussx._expfam._parameterizations import (
    expectation_to_meanvar,
    expectation_to_natural,
    meanvar_to_expectation,
    meanvar_to_natural,
    natural_to_expectation,
    natural_to_meanvar,
)
from gaussx._testing import random_pd_matrix, tree_allclose


def _make_meanvar(key, N):
    k1, k2 = jr.split(key)
    mu = jr.normal(k1, (N,))
    S = random_pd_matrix(k2, N)
    S_sqrt = jnp.linalg.cholesky(S)
    return mu, S_sqrt


# -------------------------------------------------------------------
# Roundtrip: meanvar -> natural -> meanvar
# -------------------------------------------------------------------


class TestMeanvarNatural:
    def test_roundtrip(self, getkey):
        N = 4
        mu, S_sqrt = _make_meanvar(getkey(), N)

        eta1, eta2 = meanvar_to_natural(mu, S_sqrt)
        mu_rec, S_sqrt_rec = natural_to_meanvar(eta1, eta2)

        assert tree_allclose(mu_rec, mu, rtol=1e-4)
        Sigma_orig = S_sqrt @ S_sqrt.T
        Sigma_rec = S_sqrt_rec @ S_sqrt_rec.T
        assert tree_allclose(Sigma_rec, Sigma_orig, rtol=1e-4)

    def test_natural_params_correct(self, getkey):
        """eta1 = Sigma^{-1} mu, eta2 = -0.5 Sigma^{-1}."""
        N = 3
        mu, S_sqrt = _make_meanvar(getkey(), N)
        Sigma = S_sqrt @ S_sqrt.T

        eta1, eta2 = meanvar_to_natural(mu, S_sqrt)

        Sigma_inv = jnp.linalg.inv(Sigma)
        assert tree_allclose(eta1, Sigma_inv @ mu, rtol=1e-4)
        assert tree_allclose(eta2, -0.5 * Sigma_inv, rtol=1e-4)

    def test_reverse_roundtrip(self, getkey):
        """natural -> meanvar -> natural."""
        N = 3
        Lambda = random_pd_matrix(getkey(), N)
        mu = jr.normal(getkey(), (N,))
        eta1 = Lambda @ mu
        eta2 = -0.5 * Lambda

        mu_rec, S_sqrt_rec = natural_to_meanvar(eta1, eta2)
        eta1_rec, eta2_rec = meanvar_to_natural(mu_rec, S_sqrt_rec)

        assert tree_allclose(eta1_rec, eta1, rtol=1e-4)
        assert tree_allclose(eta2_rec, eta2, rtol=1e-4)


# -------------------------------------------------------------------
# Roundtrip: meanvar -> expectation -> meanvar
# -------------------------------------------------------------------


class TestMeanvarExpectation:
    def test_roundtrip(self, getkey):
        N = 4
        mu, S_sqrt = _make_meanvar(getkey(), N)

        m1, m2 = meanvar_to_expectation(mu, S_sqrt)
        mu_rec, S_sqrt_rec = expectation_to_meanvar(m1, m2)

        assert tree_allclose(mu_rec, mu, rtol=1e-4)
        Sigma_orig = S_sqrt @ S_sqrt.T
        Sigma_rec = S_sqrt_rec @ S_sqrt_rec.T
        assert tree_allclose(Sigma_rec, Sigma_orig, rtol=1e-4)

    def test_expectation_params_correct(self, getkey):
        """m1 = mu, m2 = mu mu^T + Sigma."""
        N = 3
        mu, S_sqrt = _make_meanvar(getkey(), N)
        Sigma = S_sqrt @ S_sqrt.T

        m1, m2 = meanvar_to_expectation(mu, S_sqrt)

        assert tree_allclose(m1, mu, rtol=1e-5)
        expected_m2 = jnp.outer(mu, mu) + Sigma
        assert tree_allclose(m2, expected_m2, rtol=1e-5)

    def test_reverse_roundtrip(self, getkey):
        """expectation -> meanvar -> expectation."""
        N = 3
        mu, S_sqrt = _make_meanvar(getkey(), N)
        Sigma = S_sqrt @ S_sqrt.T
        m1 = mu
        m2 = jnp.outer(mu, mu) + Sigma

        mu_rec, S_sqrt_rec = expectation_to_meanvar(m1, m2)
        m1_rec, m2_rec = meanvar_to_expectation(mu_rec, S_sqrt_rec)

        assert tree_allclose(m1_rec, m1, rtol=1e-4)
        assert tree_allclose(m2_rec, m2, rtol=1e-4)


# -------------------------------------------------------------------
# Roundtrip: natural -> expectation -> natural
# -------------------------------------------------------------------


class TestNaturalExpectation:
    def test_roundtrip(self, getkey):
        N = 4
        Lambda = random_pd_matrix(getkey(), N)
        mu = jr.normal(getkey(), (N,))
        eta1 = Lambda @ mu
        eta2 = -0.5 * Lambda

        m1, m2 = natural_to_expectation(eta1, eta2)
        eta1_rec, eta2_rec = expectation_to_natural(m1, m2)

        assert tree_allclose(eta1_rec, eta1, rtol=1e-4)
        assert tree_allclose(eta2_rec, eta2, rtol=1e-4)

    def test_reverse_roundtrip(self, getkey):
        """expectation -> natural -> expectation."""
        N = 3
        mu, S_sqrt = _make_meanvar(getkey(), N)
        Sigma = S_sqrt @ S_sqrt.T
        m1 = mu
        m2 = jnp.outer(mu, mu) + Sigma

        eta1, eta2 = expectation_to_natural(m1, m2)
        m1_rec, m2_rec = natural_to_expectation(eta1, eta2)

        assert tree_allclose(m1_rec, m1, rtol=1e-4)
        assert tree_allclose(m2_rec, m2, rtol=1e-4)


# -------------------------------------------------------------------
# Full cycle: meanvar -> natural -> expectation -> meanvar
# -------------------------------------------------------------------


class TestFullCycle:
    def test_three_way_cycle(self, getkey):
        """meanvar -> natural -> expectation -> meanvar is identity."""
        N = 5
        mu, S_sqrt = _make_meanvar(getkey(), N)

        eta1, eta2 = meanvar_to_natural(mu, S_sqrt)
        m1, m2 = natural_to_expectation(eta1, eta2)
        mu_rec, S_sqrt_rec = expectation_to_meanvar(m1, m2)

        assert tree_allclose(mu_rec, mu, rtol=1e-4)
        Sigma_orig = S_sqrt @ S_sqrt.T
        Sigma_rec = S_sqrt_rec @ S_sqrt_rec.T
        assert tree_allclose(Sigma_rec, Sigma_orig, rtol=1e-4)

    def test_reverse_cycle(self, getkey):
        """meanvar -> expectation -> natural -> meanvar is identity."""
        N = 4
        mu, S_sqrt = _make_meanvar(getkey(), N)

        m1, m2 = meanvar_to_expectation(mu, S_sqrt)
        eta1, eta2 = expectation_to_natural(m1, m2)
        mu_rec, S_sqrt_rec = natural_to_meanvar(eta1, eta2)

        assert tree_allclose(mu_rec, mu, rtol=1e-4)
        Sigma_orig = S_sqrt @ S_sqrt.T
        Sigma_rec = S_sqrt_rec @ S_sqrt_rec.T
        assert tree_allclose(Sigma_rec, Sigma_orig, rtol=1e-4)


# -------------------------------------------------------------------
# Gradient
# -------------------------------------------------------------------


class TestBatched:
    """Batched (``*batch``) inputs flatten/vmap correctly through every conversion."""

    def _make_batched(self, key, batch_shape, N):
        keys = jr.split(key, len(batch_shape) + 2)
        mu = jr.normal(keys[0], (*batch_shape, N))

        def _one(k):
            return jnp.linalg.cholesky(random_pd_matrix(k, N))

        flat = jr.split(keys[1], int(jnp.prod(jnp.array(batch_shape))))
        S_flat = jax.vmap(_one)(flat)
        S_sqrt = S_flat.reshape((*batch_shape, N, N))
        return mu, S_sqrt

    def test_meanvar_natural_batched_roundtrip(self, getkey):
        N = 3
        batch_shape = (2, 4)
        mu, S_sqrt = self._make_batched(getkey(), batch_shape, N)

        eta1, eta2 = meanvar_to_natural(mu, S_sqrt)
        assert eta1.shape == (*batch_shape, N)
        assert eta2.shape == (*batch_shape, N, N)

        mu_rec, S_sqrt_rec = natural_to_meanvar(eta1, eta2)
        Sigma_orig = S_sqrt @ jnp.swapaxes(S_sqrt, -1, -2)
        Sigma_rec = S_sqrt_rec @ jnp.swapaxes(S_sqrt_rec, -1, -2)
        assert tree_allclose(mu_rec, mu, rtol=1e-4)
        assert tree_allclose(Sigma_rec, Sigma_orig, rtol=1e-4)

    def test_meanvar_expectation_batched_roundtrip(self, getkey):
        N = 4
        batch_shape = (3,)
        mu, S_sqrt = self._make_batched(getkey(), batch_shape, N)

        m1, m2 = meanvar_to_expectation(mu, S_sqrt)
        assert m1.shape == (*batch_shape, N)
        assert m2.shape == (*batch_shape, N, N)

        mu_rec, S_sqrt_rec = expectation_to_meanvar(m1, m2)
        Sigma_orig = S_sqrt @ jnp.swapaxes(S_sqrt, -1, -2)
        Sigma_rec = S_sqrt_rec @ jnp.swapaxes(S_sqrt_rec, -1, -2)
        assert tree_allclose(mu_rec, mu, rtol=1e-4)
        assert tree_allclose(Sigma_rec, Sigma_orig, rtol=1e-4)

    def test_natural_expectation_batched_roundtrip(self, getkey):
        N = 3
        batch_shape = (2, 2)
        mu, S_sqrt = self._make_batched(getkey(), batch_shape, N)
        eta1, eta2 = meanvar_to_natural(mu, S_sqrt)

        m1, m2 = natural_to_expectation(eta1, eta2)
        assert m1.shape == (*batch_shape, N)
        assert m2.shape == (*batch_shape, N, N)

        eta1_rec, eta2_rec = expectation_to_natural(m1, m2)
        assert tree_allclose(eta1_rec, eta1, rtol=1e-4)
        assert tree_allclose(eta2_rec, eta2, rtol=1e-4)

    def test_batched_matches_loop(self, getkey):
        """Batched output equals loop-over-batch single-instance call."""
        N = 3
        batch_shape = (5,)
        mu, S_sqrt = self._make_batched(getkey(), batch_shape, N)

        eta1_b, eta2_b = meanvar_to_natural(mu, S_sqrt)
        for i in range(batch_shape[0]):
            eta1_i, eta2_i = meanvar_to_natural(mu[i], S_sqrt[i])
            assert tree_allclose(eta1_b[i], eta1_i, rtol=1e-5)
            assert tree_allclose(eta2_b[i], eta2_i, rtol=1e-5)


class TestGradient:
    def test_grad_meanvar_to_natural(self, getkey):
        N = 3
        mu, S_sqrt = _make_meanvar(getkey(), N)

        def loss(mu, S_sqrt):
            eta1, eta2 = meanvar_to_natural(mu, S_sqrt)
            return jnp.sum(eta1**2) + jnp.sum(eta2**2)

        g_mu, g_sqrt = jax.grad(loss, argnums=(0, 1))(mu, S_sqrt)
        assert jnp.all(jnp.isfinite(g_mu))
        assert jnp.all(jnp.isfinite(g_sqrt))

    def test_grad_natural_to_meanvar(self, getkey):
        N = 3
        Lambda = random_pd_matrix(getkey(), N)
        mu = jr.normal(getkey(), (N,))
        eta1 = Lambda @ mu
        eta2 = -0.5 * Lambda

        def loss(eta1, eta2):
            mu_rec, S_sqrt_rec = natural_to_meanvar(eta1, eta2)
            return jnp.sum(mu_rec**2) + jnp.sum(S_sqrt_rec**2)

        g1, g2 = jax.grad(loss, argnums=(0, 1))(eta1, eta2)
        assert jnp.all(jnp.isfinite(g1))
        assert jnp.all(jnp.isfinite(g2))
