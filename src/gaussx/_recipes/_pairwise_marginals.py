"""Pairwise marginals for consecutive SSM states."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pairwise_marginals(
    means: jnp.ndarray,
    covariances: jnp.ndarray,
    transitions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Joint p(x_k, x_{k+1}) for each consecutive pair.

    For each pair ``(k, k+1)``, the joint distribution is::

        p(x_k, x_{k+1}) = N([mu_k; mu_{k+1}],
                             [[P_k,      P_k A_k^T],
                              [A_k P_k,  P_{k+1}  ]])

    where the cross-covariance is ``Cov[x_k, x_{k+1}] = P_k @ A_k^T``.

    Args:
        means: Smoothed means, shape ``(T, d)``.
        covariances: Smoothed covariances, shape ``(T, d, d)``.
        transitions: State transition matrices, shape ``(T-1, d, d)``.

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
        A_k: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        joint_mean = jnp.concatenate([m_k, m_kp1])
        cross_cov = P_k @ A_k.T
        joint_cov = jnp.block(
            [
                [P_k, cross_cov],
                [cross_cov.T, P_kp1],
            ]
        )
        return joint_mean, joint_cov

    joint_means, joint_covariances = jax.vmap(_single_pair)(
        means[:-1],
        means[1:],
        covariances[:-1],
        covariances[1:],
        transitions,
    )

    return joint_means, joint_covariances
