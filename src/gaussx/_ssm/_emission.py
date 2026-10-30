"""Linear emission model y = H x + ε and its Kalman-filter projections."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from gaussx._einx import einsum


class EmissionModel(eqx.Module):
    """Linear observation (emission) model with a dense observation operator H.

        y = H x + ε,   ε ~ 𝒩(0, Σ_ε)

    with a state x of length D_x and an observation y of length D_y. The
    methods are the projections a Kalman filter needs, into observation
    space (H μ_x, H Σ_x Hᵀ + Σ_ε, the residual) and back into state space in
    information form (Hᵀ Λ_ε H, Hᵀ Λ_ε y), with Λ_ε = Σ_ε⁻¹.

    Attributes:
        H: Observation operator H.

    Examples:
        Observe the first of two state components:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> emission = gaussx.EmissionModel(jnp.array([[1.0, 0.0]]))
        >>> emission.project_mean(jnp.array([2.0, 3.0])).shape
        (1,)
        >>> obs_covariance = emission.project_covariance(
        ...     jnp.eye(2), jnp.array([[0.5]])
        ... )
        >>> bool(jnp.allclose(obs_covariance, jnp.array([[1.5]])))
        True

        Back-projected information only touches the observed component:

        >>> information = emission.back_project_precision(jnp.array([[2.0]]))
        >>> bool(jnp.allclose(information, jnp.array([[2.0, 0.0], [0.0, 0.0]])))
        True
    """

    H: Float[Array, "M N"]

    def project_mean(
        self,
        mean: Float[Array, " N"],
    ) -> Float[Array, " M"]:
        """Project a state mean into observation space: μ_y = H μ_x.

        Args:
            mean: State mean μ_x.

        Returns:
            The observation mean μ_y.
        """
        # μ_y = H μ_x
        return self.H @ mean

    def project_covariance(
        self,
        cov: Float[Array, "N N"],
        noise: Float[Array, "M M"] | None = None,
    ) -> Float[Array, "M M"]:
        """Project a state covariance: Σ_y = H Σ_x Hᵀ + Σ_ε.

        Args:
            cov: State covariance Σ_x.
            noise: Observation-error covariance Σ_ε; ``None`` leaves it
                out and returns H Σ_x Hᵀ.

        Returns:
            The observation covariance Σ_y.
        """
        # H Σ_x
        obs_projection = einsum(self.H, cov, "obs state, state col -> obs col")
        # H Σ_x Hᵀ
        obs_covariance = einsum(
            obs_projection, self.H, "obs col, other col -> obs other"
        )
        if noise is not None:
            # Σ_y = H Σ_x Hᵀ + Σ_ε
            obs_covariance = obs_covariance + noise
        return obs_covariance

    def innovation(
        self,
        y: Float[Array, " M"],
        x_pred: Float[Array, " N"],
    ) -> Float[Array, " M"]:
        """Residual (innovation) of an observation: r = y − H μ_x.

        Args:
            y: Observation y.
            x_pred: Predicted state mean μ_x.

        Returns:
            The residual r.
        """
        # H μ_x
        predicted_obs = self.H @ x_pred
        # r = y − H μ_x
        return y - predicted_obs

    def back_project_precision(
        self,
        noise_prec: Float[Array, "M M"],
    ) -> Float[Array, "N N"]:
        """Back-project the observation precision: Hᵀ Λ_ε H.

        Args:
            noise_prec: Observation-error precision Λ_ε = Σ_ε⁻¹.

        Returns:
            The state-space information matrix Hᵀ Λ_ε H.
        """
        # Hᵀ Λ_ε
        weighted_obs_operator = einsum(
            self.H, noise_prec, "obs state, obs col -> state col"
        )
        # Hᵀ Λ_ε H
        return einsum(weighted_obs_operator, self.H, "state obs, obs col -> state col")

    def back_project_info(
        self,
        y: Float[Array, " M"],
        noise_prec: Float[Array, "M M"],
    ) -> Float[Array, " N"]:
        """Back-project an observation to an information vector: Hᵀ Λ_ε y.

        Args:
            y: Observation y.
            noise_prec: Observation-error precision Λ_ε = Σ_ε⁻¹.

        Returns:
            The state-space information vector Hᵀ Λ_ε y.
        """
        # Hᵀ Λ_ε
        weighted_obs_operator = einsum(
            self.H, noise_prec, "obs state, obs col -> state col"
        )
        # Hᵀ Λ_ε y
        return weighted_obs_operator @ y
