"""Pairwise marginals for consecutive SSM states."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pairwise_marginals(
    means: jnp.ndarray,
    covariances: jnp.ndarray,
    cross_covariances: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Joint p(x_k, x_{k+1}) for each consecutive pair.

    For each pair ``(k, k+1)``, the joint distribution is::

        p(x_k, x_{k+1}) = N([mu_k; mu_{k+1}],
                             [[P_k,      C_k^T],
                              [C_k,      P_{k+1}]])

    where ``C_k = Cov[x_{k+1}, x_k]`` is the pairwise cross-covariance.

    Args:
        means: Smoothed means, shape ``(T, d)``.
        covariances: Smoothed covariances, shape ``(T, d, d)``.
        cross_covariances: Pairwise cross-covariances
            ``Cov[x_{k+1}, x_k]``, shape ``(T-1, d, d)``.

    Returns:
        Tuple ``(joint_means, joint_covariances)`` where:

        - ``joint_means``: shape ``(T-1, 2*d)``
        - ``joint_covariances``: shape ``(T-1, 2*d, 2*d)``
    """

    def _single_pair(
        m_k: jnp.ndarray,
        m_kp1: jnp.ndarray,
        P_k: jnp.ndarray,
        P_kp1: jnp.ndarray,
        C_k: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        joint_mean = jnp.concatenate([m_k, m_kp1])
        joint_cov = jnp.block(
            [
                [P_k, C_k.T],
                [C_k, P_kp1],
            ]
        )
        return joint_mean, joint_cov

    joint_means, joint_covariances = jax.vmap(_single_pair)(
        means[:-1],
        means[1:],
        covariances[:-1],
        covariances[1:],
        cross_covariances,
    )

    return joint_means, joint_covariances
