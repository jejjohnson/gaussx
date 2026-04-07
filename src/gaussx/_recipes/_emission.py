"""Emission model projection for Kalman filter operations."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class EmissionModel(eqx.Module):
    """Observation (emission) model wrapping a linear observation matrix.

    Provides named methods for common Kalman filter projection operations.

    Attributes:
        H: Observation matrix, shape ``(M, N)``.
    """

    H: jnp.ndarray

    def project_mean(self, mean: jnp.ndarray) -> jnp.ndarray:
        """Project state mean to observation space.

        Args:
            mean: State mean, shape ``(N,)``.

        Returns:
            Projected mean, shape ``(M,)``.
        """
        return self.H @ mean

    def project_covariance(
        self,
        cov: jnp.ndarray,
        noise: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Project state covariance to observation space.

        Args:
            cov: State covariance, shape ``(N, N)``.
            noise: Optional observation noise, shape ``(M, M)``.

        Returns:
            Innovation covariance, shape ``(M, M)``.
        """
        S = self.H @ cov @ self.H.T
        if noise is not None:
            S = S + noise
        return S

    def innovation(
        self,
        y: jnp.ndarray,
        x_pred: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute innovation (measurement residual).

        Args:
            y: Observation, shape ``(M,)``.
            x_pred: Predicted state mean, shape ``(N,)``.

        Returns:
            Innovation vector, shape ``(M,)``.
        """
        return y - self.H @ x_pred

    def back_project_precision(
        self,
        noise_prec: jnp.ndarray,
    ) -> jnp.ndarray:
        """Back-project observation precision to state space.

        Args:
            noise_prec: Observation noise precision, shape ``(M, M)``.

        Returns:
            Information matrix contribution, shape ``(N, N)``.
        """
        return self.H.T @ noise_prec @ self.H

    def back_project_info(
        self,
        y: jnp.ndarray,
        noise_prec: jnp.ndarray,
    ) -> jnp.ndarray:
        """Back-project observation to information vector.

        Args:
            y: Observation, shape ``(M,)``.
            noise_prec: Observation noise precision, shape ``(M, M)``.

        Returns:
            Information vector contribution, shape ``(N,)``.
        """
        return self.H.T @ noise_prec @ y
