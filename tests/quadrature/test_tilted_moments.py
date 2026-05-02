"""Tests for EP tilted moment computation."""

import jax
import jax.numpy as jnp

from gaussx import ep_tilted_moments, site_natural_from_tilted


class TestEPTiltedMoments:
    def test_gaussian_likelihood_exact(self):
        """With Gaussian likelihood, tilted moments match exact posterior."""
        y = jnp.array(2.0)
        R = jnp.array(0.5)

        def log_lik(f):
            return -0.5 * (y - f) ** 2 / R

        cav_mean = jnp.array(1.0)
        cav_var = jnp.array(1.0)
        post_prec = 1.0 / cav_var + 1.0 / R
        post_var = 1.0 / post_prec
        post_mean = post_var * (cav_mean / cav_var + y / R)

        t_mean, t_var = ep_tilted_moments(log_lik, cav_mean, cav_var, order=30)
        assert jnp.allclose(t_mean, post_mean, atol=1e-4)
        assert jnp.allclose(t_var, post_var, atol=1e-4)

    def test_bernoulli_likelihood(self):
        y = 1.0

        def log_lik(f):
            return y * jax.nn.log_sigmoid(f) + (1.0 - y) * jax.nn.log_sigmoid(-f)

        t_mean, t_var = ep_tilted_moments(
            log_lik, jnp.array(0.0), jnp.array(1.0), order=30
        )
        assert t_mean > 0.0
        assert t_var < 1.0
        assert t_var > 0.0

    def test_jit_compatible(self):
        def log_lik(f):
            return -0.5 * (1.0 - f) ** 2

        @jax.jit
        def compute(cav_mean, cav_var):
            return ep_tilted_moments(log_lik, cav_mean, cav_var)

        t_mean, t_var = compute(jnp.array(0.0), jnp.array(1.0))
        assert jnp.isfinite(t_mean)
        assert jnp.isfinite(t_var)

    def test_integration_with_site_naturals(self):
        def log_lik(f):
            return jax.nn.log_sigmoid(f)

        t_mean, t_var = ep_tilted_moments(log_lik, jnp.array(0.0), jnp.array(2.0))
        nat1, nat2 = site_natural_from_tilted(
            t_mean, t_var, jnp.array(0.0), jnp.array(2.0)
        )
        assert nat2 > 0.0
        assert jnp.isfinite(nat1)
