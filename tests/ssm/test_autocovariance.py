"""Tests for SDE autocovariance utility."""

import jax
import jax.numpy as jnp

from gaussx import ConstantSDE, CosineSDE, MaternSDE, sde_autocovariance


class TestSDEAutocovariance:
    def test_zero_lag_equals_variance(self):
        kern = MaternSDE(variance=jnp.array(2.5), lengthscale=jnp.array(1.0), order=1)
        k0 = sde_autocovariance(kern, jnp.array(0.0))
        assert jnp.allclose(k0, 2.5, atol=1e-6)

    def test_symmetry(self):
        kern = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)
        k_pos = sde_autocovariance(kern, jnp.array(0.5))
        k_neg = sde_autocovariance(kern, jnp.array(-0.5))
        assert jnp.allclose(k_pos, k_neg, atol=1e-6)

    def test_matern12_analytical(self):
        sigma2 = jnp.array(2.0)
        ell = jnp.array(1.5)
        kern = MaternSDE(variance=sigma2, lengthscale=ell, order=0)
        taus = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
        k_vals = sde_autocovariance(kern, taus)
        expected = sigma2 * jnp.exp(-jnp.abs(taus) / ell)
        assert jnp.allclose(k_vals, expected, atol=1e-5)

    def test_cosine_kernel(self):
        sigma2 = jnp.array(1.5)
        w = jnp.array(3.0)
        kern = CosineSDE(variance=sigma2, frequency=w)
        taus = jnp.array([0.0, 0.1, 0.5, 1.0])
        k_vals = sde_autocovariance(kern, taus)
        expected = sigma2 * jnp.cos(w * taus)
        assert jnp.allclose(k_vals, expected, atol=1e-5)

    def test_constant_kernel(self):
        sigma2 = jnp.array(3.0)
        kern = ConstantSDE(variance=sigma2)
        taus = jnp.array([0.0, 1.0, 10.0, 100.0])
        k_vals = sde_autocovariance(kern, taus)
        assert jnp.allclose(k_vals, sigma2, atol=1e-5)

    def test_differentiable(self):
        def loss(variance, lengthscale):
            kern = MaternSDE(variance=variance, lengthscale=lengthscale, order=1)
            return sde_autocovariance(kern, jnp.array(0.5))

        grad_fn = jax.grad(loss, argnums=(0, 1))
        g_var, g_ell = grad_fn(jnp.array(1.0), jnp.array(1.0))
        assert jnp.isfinite(g_var)
        assert jnp.isfinite(g_ell)
