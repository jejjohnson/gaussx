"""OILMM projection for multi-output Gaussian processes."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def oilmm_project(
    Y: Float[Array, "N P"],
    W: Float[Array, "P L"],
    noise_var: Float[Array, " P"] | float,
) -> tuple[Float[Array, "N L"], Float[Array, " L"]]:
    """Project multi-output data to independent latent GPs via OILMM.

    Given an orthogonal mixing matrix W ∈ ℝᴾˣᴸ with WᵀW = Iₗ, projects
    P-output observations to L independent latent channels::

        Y_latent    = Y W              (N, L)
        σ²_latent   = (W ⊙ W)ᵀ σ²     (L,)

    Args:
        Y: Observations, shape ``(N, P)``.
        W: Orthogonal mixing matrix, shape ``(P, L)`` with WᵀW = Iₗ.
        noise_var: Observation noise variance. Scalar for isotropic noise,
            or shape ``(P,)`` for heteroscedastic noise.

    Returns:
        Tuple ``(Y_latent, noise_latent)`` with shapes ``(N, L)``
        and ``(L,)``.
    """
    Y_latent = Y @ W  # (N, L)
    noise_var = jnp.broadcast_to(jnp.asarray(noise_var), (W.shape[0],))  # (P,)
    noise_latent = (W**2).T @ noise_var  # (L,)
    return Y_latent, noise_latent


def oilmm_back_project(
    f_means: Float[Array, "N L"],
    f_vars: Float[Array, "N L"],
    W: Float[Array, "P L"],
) -> tuple[Float[Array, "N P"], Float[Array, "N P"]]:
    """Back-project latent GP predictions to the observation space.

    Reconstructs observation-space predictions via::

        y_means = f_means Wᵀ              (N, P)
        y_vars  = f_vars (W ⊙ W)ᵀ        (N, P)

    Args:
        f_means: Latent predictive means, shape ``(N, L)``.
        f_vars: Latent predictive variances, shape ``(N, L)``.
        W: Orthogonal mixing matrix, shape ``(P, L)`` with WᵀW = Iₗ.

    Returns:
        Tuple ``(y_means, y_vars)`` with shapes ``(N, P)`` and ``(N, P)``.
    """
    y_means = f_means @ W.T  # (N, P)
    y_vars = f_vars @ (W**2).T  # (N, P)
    return y_means, y_vars
