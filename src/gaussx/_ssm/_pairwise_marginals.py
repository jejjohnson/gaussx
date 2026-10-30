"""Joint Gaussians of consecutive state pairs (x_t, x_t+1) of a Markov chain."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import rearrange


def pairwise_marginals(
    means: Float[Array, "T d"],
    covariances: Float[Array, "T d d"],
    cross_covariances: Float[Array, "Tm1 d d"],
) -> tuple[Float[Array, "Tm1 two_d"], Float[Array, "Tm1 two_d two_d"]]:
    """Joint Gaussian p(x_t, x_t+1) of each pair of consecutive states.

        p(x_t, x_t+1) = 𝒩([μ_x,t; μ_x,t+1], [[Σ_x,t, Σ_x,t+1,tᵀ],
                                            [Σ_x,t+1,t, Σ_x,t+1]])

    with the lag-one cross-covariance Σ_x,t+1,t = Cov[x_t+1, x_t] and
    t = 1, …, T − 1. The moments are usually smoothed ones, conditioned on
    y_1:T; the conditioning is left out of the symbols here.

    Args:
        means: Marginal means μ_x,t, one row per step.
        covariances: Marginal covariances Σ_x,t.
        cross_covariances: Lag-one cross-covariances Σ_x,t+1,t, one per
            consecutive pair.

    Returns:
        Tuple ``(joint_means, joint_covariances)`` holding the stacked mean
        [μ_x,t; μ_x,t+1] and the 2 × 2 block covariance of each pair, of
        length 2 D_x.

    Examples:
        Three steps of a two-dimensional state give two pairs:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> means = jnp.zeros((3, 2))
        >>> covariances = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
        >>> cross = jnp.broadcast_to(jnp.array([[0.5, 0.1], [0.0, 0.5]]), (2, 2, 2))
        >>> joint_means, joint_covs = gaussx.pairwise_marginals(
        ...     means, covariances, cross
        ... )
        >>> joint_means.shape, joint_covs.shape
        ((2, 4), (2, 4, 4))

        The diagonal blocks are the marginals and Σ_x,t+1,t sits below
        the diagonal:

        >>> bool(jnp.allclose(joint_covs[0, :2, :2], covariances[0]))
        True
        >>> bool(jnp.allclose(joint_covs[0, 2:, :2], cross[0]))
        True
    """

    def _single_pair(
        mean: Float[Array, " d"],
        mean_next: Float[Array, " d"],
        covariance: Float[Array, "d d"],
        covariance_next: Float[Array, "d d"],
        cross_covariance: Float[Array, "d d"],
    ) -> tuple[Float[Array, " two_d"], Float[Array, "two_d two_d"]]:
        # [μ_x,t; μ_x,t+1]
        joint_mean = jnp.concatenate([mean, mean_next])
        # Σ_x,t+1,tᵀ = Cov[x_t, x_t+1]
        cross_covariance_transpose = rearrange(
            cross_covariance, "next state -> state next"
        )
        # [[Σ_x,t, Σ_x,t+1,tᵀ], [Σ_x,t+1,t, Σ_x,t+1]]
        joint_covariance = jnp.block(
            [
                [covariance, cross_covariance_transpose],
                [cross_covariance, covariance_next],
            ]
        )
        return joint_mean, joint_covariance

    joint_means, joint_covariances = jax.vmap(_single_pair)(
        means[:-1],
        means[1:],
        covariances[:-1],
        covariances[1:],
        cross_covariances,
    )

    return joint_means, joint_covariances
