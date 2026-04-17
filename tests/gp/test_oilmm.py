"""Tests for OILMM projection."""

import jax
import jax.numpy as jnp

from gaussx import oilmm_back_project, oilmm_project


class TestOILMM:
    def test_identity_roundtrip(self, getkey):
        """W=I gives identity projection."""
        N, P = 10, 3
        Y = jax.random.normal(getkey(), (N, P))
        W = jnp.eye(P)
        noise_var = 0.1

        Y_lat, noise_lat = oilmm_project(Y, W, noise_var)
        assert jnp.allclose(Y_lat, Y, atol=1e-6)
        assert jnp.allclose(noise_lat, noise_var * jnp.ones(P), atol=1e-6)

        y_means, _y_vars = oilmm_back_project(Y_lat, jnp.ones((N, P)), W)
        assert jnp.allclose(y_means, Y_lat, atol=1e-6)

    def test_output_shapes(self, getkey):
        """Shapes are correct for non-square W."""
        N, P, L = 20, 5, 3
        Y = jax.random.normal(getkey(), (N, P))
        # Orthogonal W via QR
        W, _ = jnp.linalg.qr(jax.random.normal(getkey(), (P, L)))
        noise_var = 0.1

        Y_lat, noise_lat = oilmm_project(Y, W, noise_var)
        assert Y_lat.shape == (N, L)
        assert noise_lat.shape == (L,)

        f_vars = jnp.ones((N, L))
        y_means, y_vars = oilmm_back_project(Y_lat, f_vars, W)
        assert y_means.shape == (N, P)
        assert y_vars.shape == (N, P)

    def test_heteroscedastic_noise(self, getkey):
        """Per-output noise variance is handled correctly."""
        N, P, L = 10, 4, 2
        Y = jax.random.normal(getkey(), (N, P))
        W, _ = jnp.linalg.qr(jax.random.normal(getkey(), (P, L)))
        noise_var = jnp.array([0.1, 0.2, 0.3, 0.4])

        _Y_lat, noise_lat = oilmm_project(Y, W, noise_var)
        assert noise_lat.shape == (L,)
        # noise_latent = W^2 @ noise_var
        expected = (W**2).T @ noise_var
        assert jnp.allclose(noise_lat, expected, atol=1e-6)

    def test_jit_compatible(self, getkey):
        """Both functions work under jax.jit."""
        N, P, L = 10, 3, 2
        Y = jax.random.normal(getkey(), (N, P))
        W, _ = jnp.linalg.qr(jax.random.normal(getkey(), (P, L)))

        Y_lat, _noise_lat = jax.jit(oilmm_project)(Y, W, 0.1)
        assert jnp.all(jnp.isfinite(Y_lat))

        f_vars = jnp.ones((N, L))
        y_means, _y_vars = jax.jit(oilmm_back_project)(Y_lat, f_vars, W)
        assert jnp.all(jnp.isfinite(y_means))
