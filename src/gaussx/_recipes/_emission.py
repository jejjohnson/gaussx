"""Emission model projection for Kalman filter operations."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float


class EmissionModel(eqx.Module):
    """Observation (emission) model wrapping a linear observation matrix.

    Provides named methods for common Kalman filter projection
    operations with observation matrix H ∈ ℝᴹˣᴺ.

    Attributes:
        H: Observation matrix, shape ``(M, N)``.
    """

    H: Float[Array, "M N"]

    def project_mean(
        self,
        mean: Float[Array, " N"],
    ) -> Float[Array, " M"]:
        """Project state mean to observation space: ŷ = H x.

        Args:
            mean: State mean, shape ``(N,)``.

        Returns:
            Projected mean, shape ``(M,)``.
        """
        return self.H @ mean

    def project_covariance(
        self,
        cov: Float[Array, "N N"],
        noise: Float[Array, "M M"] | None = None,
    ) -> Float[Array, "M M"]:
        """Project state covariance: S = H P Hᵀ [+ R].

        Args:
            cov: State covariance P, shape ``(N, N)``.
            noise: Optional observation noise R, shape ``(M, M)``.

        Returns:
            Innovation covariance S, shape ``(M, M)``.
        """
        S = self.H @ cov @ self.H.T  # (M, M)
        if noise is not None:
            S = S + noise
        return S

    def innovation(
        self,
        y: Float[Array, " M"],
        x_pred: Float[Array, " N"],
    ) -> Float[Array, " M"]:
        """Compute innovation (measurement residual): v = y − H x.

        Args:
            y: Observation, shape ``(M,)``.
            x_pred: Predicted state mean, shape ``(N,)``.

        Returns:
            Innovation vector v, shape ``(M,)``.
        """
        return y - self.H @ x_pred

    def back_project_precision(
        self,
        noise_prec: Float[Array, "M M"],
    ) -> Float[Array, "N N"]:
        """Back-project observation precision: Hᵀ R⁻¹ H.

        Args:
            noise_prec: Observation noise precision R⁻¹, shape ``(M, M)``.

        Returns:
            Information matrix contribution, shape ``(N, N)``.
        """
        return self.H.T @ noise_prec @ self.H

    def back_project_info(
        self,
        y: Float[Array, " M"],
        noise_prec: Float[Array, "M M"],
    ) -> Float[Array, " N"]:
        """Back-project observation to information vector: Hᵀ R⁻¹ y.

        Args:
            y: Observation, shape ``(M,)``.
            noise_prec: Observation noise precision R⁻¹, shape ``(M, M)``.

        Returns:
            Information vector contribution, shape ``(N,)``.
        """
        return self.H.T @ noise_prec @ y
