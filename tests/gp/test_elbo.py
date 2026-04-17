"""Tests for variational ELBO sugar."""

import jax
import jax.numpy as jnp

from gaussx._gp._elbo import variational_elbo_gaussian, variational_elbo_mc


class TestVariationalElboGaussian:
    def test_basic_value(self):
        """ELBO should be negative (log-prob minus KL)."""
        key = jax.random.key(0)
        N = 10
        y = jax.random.normal(key, (N,))
        f_loc = jnp.zeros(N)
        f_var = jnp.ones(N)
        noise_var = 1.0
        kl = jnp.array(0.5)
        elbo = variational_elbo_gaussian(y, f_loc, f_var, noise_var, kl)
        assert elbo.shape == ()
        assert jnp.isfinite(elbo)

    def test_zero_kl(self):
        """With zero KL, ELBO equals expected log-likelihood."""
        N = 5
        y = jnp.ones(N)
        f_loc = jnp.ones(N)
        f_var = jnp.zeros(N)
        noise_var = 1.0
        kl = jnp.array(0.0)
        elbo = variational_elbo_gaussian(y, f_loc, f_var, noise_var, kl)
        # With y == f_loc and f_var == 0: ELL = -0.5 * N * log(2 pi sigma^2)
        expected = -0.5 * N * jnp.log(2.0 * jnp.pi * noise_var)
        assert jnp.allclose(elbo, expected, atol=1e-6)

    def test_variance_penalty(self):
        """Increasing f_var should decrease the ELBO."""
        N = 5
        y = jnp.ones(N)
        f_loc = jnp.ones(N)
        kl = jnp.array(0.0)
        noise_var = 1.0
        elbo_low = variational_elbo_gaussian(y, f_loc, 0.1 * jnp.ones(N), noise_var, kl)
        elbo_high = variational_elbo_gaussian(
            y, f_loc, 10.0 * jnp.ones(N), noise_var, kl
        )
        assert elbo_low > elbo_high

    def test_kl_penalty(self):
        """Increasing KL should decrease the ELBO."""
        N = 5
        y = jnp.ones(N)
        f_loc = jnp.ones(N)
        f_var = jnp.ones(N)
        noise_var = 1.0
        elbo_low_kl = variational_elbo_gaussian(
            y, f_loc, f_var, noise_var, jnp.array(0.1)
        )
        elbo_high_kl = variational_elbo_gaussian(
            y, f_loc, f_var, noise_var, jnp.array(10.0)
        )
        assert elbo_low_kl > elbo_high_kl

    def test_jit(self):
        """Should be JIT-compatible."""
        N = 5
        y = jnp.ones(N)
        f_loc = jnp.zeros(N)
        f_var = jnp.ones(N)
        noise_var = 1.0
        kl = jnp.array(0.5)
        elbo_eager = variational_elbo_gaussian(y, f_loc, f_var, noise_var, kl)
        elbo_jit = jax.jit(variational_elbo_gaussian)(y, f_loc, f_var, noise_var, kl)
        assert jnp.allclose(elbo_eager, elbo_jit, atol=1e-10)


class TestVariationalElboMC:
    def test_basic_value(self):
        """MC ELBO should return a finite scalar."""
        key = jax.random.key(42)
        N = 5
        S = 100
        f_samples = jax.random.normal(key, (S, N))

        def log_lik(f):
            return -0.5 * jnp.sum(f**2)

        kl = jnp.array(1.0)
        elbo = variational_elbo_mc(log_lik, f_samples, kl)
        assert elbo.shape == ()
        assert jnp.isfinite(elbo)

    def test_zero_kl(self):
        """With zero KL, MC ELBO equals average log-likelihood."""
        key = jax.random.key(0)
        N = 3
        S = 50
        f_samples = jax.random.normal(key, (S, N))

        def log_lik(f):
            return -0.5 * jnp.sum(f**2)

        kl = jnp.array(0.0)
        elbo = variational_elbo_mc(log_lik, f_samples, kl)
        expected = jnp.mean(jax.vmap(log_lik)(f_samples))
        assert jnp.allclose(elbo, expected, atol=1e-6)

    def test_jit(self):
        """Should be JIT-compatible."""
        key = jax.random.key(0)
        f_samples = jax.random.normal(key, (20, 3))
        log_lik = lambda f: -0.5 * jnp.sum(f**2)
        kl = jnp.array(0.5)

        elbo_eager = variational_elbo_mc(log_lik, f_samples, kl)
        elbo_jit = jax.jit(variational_elbo_mc, static_argnums=(0,))(
            log_lik, f_samples, kl
        )
        assert jnp.allclose(elbo_eager, elbo_jit, atol=1e-10)
