"""Ensemble covariance and cross-covariance recipes."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._operators._low_rank_update import LowRankUpdate


def ensemble_covariance(
    particles: jnp.ndarray,
) -> LowRankUpdate:
    """Empirical covariance from an ensemble as a low-rank operator.

    Computes ``C = (1/J) sum_j (x_j - x_bar)(x_j - x_bar)^T``
    and returns a ``LowRankUpdate`` of rank ``<= J-1`` rather than
    materializing the full ``(N, N)`` matrix.  Efficient when
    ``J << N``.

    Args:
        particles: Ensemble of shape ``(J, N)``.

    Returns:
        A ``LowRankUpdate`` operator representing the empirical
        covariance, with a zero base and rank ``J``.
    """
    J, N = particles.shape
    mean = jnp.mean(particles, axis=0)
    deviations = particles - mean[None, :]  # (J, N)

    # C = (1/J) dev^T dev  =>  represent as U diag(d) U^T
    # where U = dev^T / sqrt(J),  d = ones(J)
    U = deviations.T / jnp.sqrt(J)  # (N, J)

    base = lx.DiagonalLinearOperator(jnp.zeros(N, dtype=particles.dtype))
    return LowRankUpdate(base, U)


def ensemble_cross_covariance(
    particles_theta: jnp.ndarray,
    particles_G: jnp.ndarray,
) -> jnp.ndarray:
    """Cross-covariance between two ensemble sets.

    Computes ``C^{theta,G} = (1/J) sum_j (theta_j - bar)(G_j - bar)^T``.

    Args:
        particles_theta: First ensemble, shape ``(J, N)``.
        particles_G: Second ensemble, shape ``(J, M)``.

    Returns:
        Cross-covariance array of shape ``(N, M)``.
    """
    J = particles_theta.shape[0]
    dev_theta = particles_theta - jnp.mean(particles_theta, axis=0, keepdims=True)
    dev_G = particles_G - jnp.mean(particles_G, axis=0, keepdims=True)
    return (dev_theta.T @ dev_G) / J
