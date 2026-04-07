"""Tests for EmissionModel."""

import equinox as eqx
import jax
import jax.numpy as jnp

from gaussx import EmissionModel


class TestEmissionModel:
    def test_project_mean(self, getkey):
        """project_mean computes H @ mean."""
        M, N = 2, 3
        H = jax.random.normal(getkey(), (M, N))
        mean = jax.random.normal(getkey(), (N,))
        em = EmissionModel(H=H)
        result = em.project_mean(mean)
        assert result.shape == (M,)
        assert jnp.allclose(result, H @ mean, atol=1e-6)

    def test_project_covariance(self, getkey):
        """project_covariance computes H @ cov @ H^T."""
        M, N = 2, 3
        H = jax.random.normal(getkey(), (M, N))
        cov = jnp.eye(N)
        em = EmissionModel(H=H)
        S = em.project_covariance(cov)
        assert S.shape == (M, M)
        assert jnp.allclose(S, H @ cov @ H.T, atol=1e-6)

    def test_project_covariance_with_noise(self, getkey):
        """project_covariance adds noise when provided."""
        M, N = 2, 3
        H = jax.random.normal(getkey(), (M, N))
        cov = jnp.eye(N)
        noise = 0.5 * jnp.eye(M)
        em = EmissionModel(H=H)
        S = em.project_covariance(cov, noise=noise)
        assert jnp.allclose(S, H @ cov @ H.T + noise, atol=1e-6)

    def test_innovation(self, getkey):
        """innovation computes y - H @ x_pred."""
        M, N = 2, 3
        H = jax.random.normal(getkey(), (M, N))
        x_pred = jax.random.normal(getkey(), (N,))
        y = jax.random.normal(getkey(), (M,))
        em = EmissionModel(H=H)
        v = em.innovation(y, x_pred)
        assert v.shape == (M,)
        assert jnp.allclose(v, y - H @ x_pred, atol=1e-6)

    def test_back_project_precision(self, getkey):
        """back_project_precision computes H^T @ R^{-1} @ H."""
        M, N = 2, 3
        H = jax.random.normal(getkey(), (M, N))
        noise_prec = 2.0 * jnp.eye(M)
        em = EmissionModel(H=H)
        result = em.back_project_precision(noise_prec)
        assert result.shape == (N, N)
        assert jnp.allclose(result, H.T @ noise_prec @ H, atol=1e-6)

    def test_back_project_info(self, getkey):
        """back_project_info computes H^T @ R^{-1} @ y."""
        M, N = 2, 3
        H = jax.random.normal(getkey(), (M, N))
        y = jax.random.normal(getkey(), (M,))
        noise_prec = 2.0 * jnp.eye(M)
        em = EmissionModel(H=H)
        result = em.back_project_info(y, noise_prec)
        assert result.shape == (N,)
        assert jnp.allclose(result, H.T @ noise_prec @ y, atol=1e-6)

    def test_jit_compatible(self, getkey):
        """All methods work under jax.jit."""
        M, N = 2, 3
        H = jax.random.normal(getkey(), (M, N))
        em = EmissionModel(H=H)
        mean = jax.random.normal(getkey(), (N,))
        cov = jnp.eye(N)

        result = eqx.filter_jit(em.project_mean)(mean)
        assert jnp.all(jnp.isfinite(result))

        S = eqx.filter_jit(em.project_covariance)(cov)
        assert jnp.all(jnp.isfinite(S))
