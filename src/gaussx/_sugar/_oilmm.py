"""OILMM projection for multi-output Gaussian processes."""

from __future__ import annotations

import jax.numpy as jnp


def oilmm_project(
    Y: jnp.ndarray,
    W: jnp.ndarray,
    noise_var: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project multi-output data to independent latent GPs via OILMM.

    Given an orthogonal mixing matrix W (P x L, W^T W = I_L), projects
    P-output observations to L independent latent channels.

    Args:
        Y: Observations, shape ``(N, P)``.
        W: Orthogonal mixing matrix, shape ``(P, L)`` with ``W^T W = I_L``.
        noise_var: Observation noise variance. Scalar for isotropic noise,
            or shape ``(P,)`` for heteroscedastic noise.

    Returns:
        Tuple ``(Y_latent, noise_latent)`` where ``Y_latent`` has shape
        ``(N, L)`` and ``noise_latent`` has shape ``(L,)``.
    """
    Y_latent = Y @ W  # (N, L)
    noise_var = jnp.broadcast_to(jnp.asarray(noise_var), (W.shape[0],))
    noise_latent = (W**2).T @ noise_var  # (L,)
    return Y_latent, noise_latent


def oilmm_back_project(
    f_means: jnp.ndarray,
    f_vars: jnp.ndarray,
    W: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Back-project latent GP predictions to the observation space.

    Args:
        f_means: Latent predictive means, shape ``(N, L)``.
        f_vars: Latent predictive variances, shape ``(N, L)``.
        W: Orthogonal mixing matrix, shape ``(P, L)`` with ``W^T W = I_L``.

    Returns:
        Tuple ``(y_means, y_vars)`` where ``y_means`` has shape
        ``(N, P)`` and ``y_vars`` has shape ``(N, P)``.
    """
    y_means = f_means @ W.T  # (N, P)
    y_vars = f_vars @ (W**2).T  # (N, P)
    return y_means, y_vars
