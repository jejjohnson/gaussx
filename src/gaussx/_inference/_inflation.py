"""Ensemble inflation: multiplicative, RTPP and RTPS."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def inflate_multiplicative(
    ensemble: Float[Array, "J N"],
    factor: float | Float[Array, ""],
) -> Float[Array, "J N"]:
    r"""Multiplicative ensemble inflation about the mean.

    Restores ensemble spread lost to sampling error / model collapse by scaling
    perturbations: ``x_j <- x_bar + factor (x_j - x_bar)``.

    Args:
        ensemble: Ensemble of shape ``(J, N)``.
        factor: Inflation factor ``>= 1`` (e.g. ``1.02``-``1.10``).

    Returns:
        Inflated ensemble, shape ``(J, N)``. The mean is unchanged.
    """
    mean = jnp.mean(ensemble, axis=0, keepdims=True)
    return mean + factor * (ensemble - mean)


def inflate_rtpp(
    posterior: Float[Array, "J N"],
    prior: Float[Array, "J N"],
    alpha: float | Float[Array, ""],
) -> Float[Array, "J N"]:
    r"""Relaxation to prior perturbations (RTPP; Zhang et al. 2004).

    Relaxes posterior perturbations toward the prior perturbations while keeping
    the posterior mean: ``x'^a <- (1 - alpha) x'^a + alpha x'^f``, where the
    perturbations are taken about each ensemble's own mean.

    Args:
        posterior: Analysis ensemble, shape ``(J, N)``.
        prior: Forecast ensemble, shape ``(J, N)``.
        alpha: Relaxation weight in ``[0, 1]``.

    Returns:
        Relaxed analysis ensemble, shape ``(J, N)``. The posterior mean is
        preserved.
    """
    post_mean = jnp.mean(posterior, axis=0, keepdims=True)
    post_pert = posterior - post_mean
    prior_pert = prior - jnp.mean(prior, axis=0, keepdims=True)
    return post_mean + (1.0 - alpha) * post_pert + alpha * prior_pert


def inflate_rtps(
    posterior: Float[Array, "J N"],
    prior: Float[Array, "J N"],
    beta: float | Float[Array, ""],
    eps: float = 1e-12,
) -> Float[Array, "J N"]:
    r"""Relaxation to prior spread (RTPS; Whitaker & Hamill 2012).

    Scales each posterior perturbation, per coordinate, so the analysis spread
    relaxes back toward the prior spread:
    ``x'^a <- x'^a [ (1 - beta) + beta sigma^f / sigma^a ]``, with ``sigma`` the
    per-coordinate ensemble standard deviation.

    Args:
        posterior: Analysis ensemble, shape ``(J, N)``.
        prior: Forecast ensemble, shape ``(J, N)``.
        beta: Relaxation weight in ``[0, 1]``.
        eps: Floor on the posterior std to avoid division by zero.

    Returns:
        Spread-restored analysis ensemble, shape ``(J, N)``. The posterior mean
        is preserved.
    """
    post_mean = jnp.mean(posterior, axis=0, keepdims=True)
    post_pert = posterior - post_mean
    sigma_post = jnp.std(posterior, axis=0)
    sigma_prior = jnp.std(prior, axis=0)
    scale = (1.0 - beta) + beta * sigma_prior / (sigma_post + eps)
    return post_mean + post_pert * scale[None, :]
