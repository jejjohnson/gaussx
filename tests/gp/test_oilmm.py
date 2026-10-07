"""Tests for OILMM projection."""

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx import oilmm_back_project, oilmm_project
from gaussx._einx import einsum


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

    def test_heteroscedastic_noise_is_diag_of_projected_noise(self):
        """Latent noise is diag(WᵀDW): the OILMM independence approximation."""
        P = 4
        W, _ = jnp.linalg.qr(jr.normal(jr.key(0), (P, 2)))
        D = jnp.array([0.05, 0.1, 0.2, 0.4])
        _, noise_lat = oilmm_project(jnp.ones((3, P)), W, D)
        projected = einsum(einx.multiply("p i, p -> p i", W, D), W, "p i, p j -> i j")
        assert jnp.allclose(noise_lat, jnp.diag(projected), atol=1e-12)
        # The dropped off-diagonal is genuinely non-zero for this D.
        assert jnp.abs(projected[0, 1]) > 1e-3

    def test_isotropic_noise_projects_exactly(self):
        P = 4
        W, _ = jnp.linalg.qr(jr.normal(jr.key(0), (P, 2)))
        _, noise_lat = oilmm_project(jnp.ones((3, P)), W, 0.1)
        projected = 0.1 * einsum(W, W, "p i, p j -> i j")
        assert jnp.allclose(projected, jnp.diag(jnp.diag(projected)), atol=1e-12)
        assert jnp.allclose(noise_lat, 0.1, atol=1e-12)

    def test_check_orthonormal(self):
        P = 4
        W, _ = jnp.linalg.qr(jr.normal(jr.key(0), (P, 2)))
        Y = jnp.ones((3, P))
        oilmm_project(Y, W, 0.1, check_orthonormal=True)  # passes
        with pytest.raises(eqx.EquinoxRuntimeError, match="orthonormal"):
            oilmm_project(Y, 2.0 * W, 0.1, check_orthonormal=True)
        # Under jit the error surfaces through the runtime callback.
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError), match="orthonormal"
        ):
            jax.jit(lambda w: oilmm_project(Y, w, 0.1, check_orthonormal=True))(2.0 * W)

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
