"""Joseph-form covariance update for Kalman filters."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def joseph_update(
    P_pred: Float[Array, "N N"],
    K: Float[Array, "N M"],
    H: Float[Array, "M N"],
    R: Float[Array, "M M"],
) -> Float[Array, "N N"]:
    r"""Numerically stable Joseph-form covariance update.

    Computes the updated covariance after a Kalman measurement update::

        P_update = (I - K H) P_pred (I - K H)^T + K R K^T

    This form is more numerically stable than the simplified
    ``P = P_pred - K S K^T`` or ``P = (I - K H) P_pred`` because it
    guarantees symmetry and is more robust when the Kalman gain ``K``
    is approximate or the system is poorly conditioned.

    Args:
        P_pred: Predicted covariance, shape ``(N, N)``.
        K: Kalman gain, shape ``(N, M)``.
        H: Observation model, shape ``(M, N)``.
        R: Observation noise covariance, shape ``(M, M)``.

    Returns:
        Updated covariance, shape ``(N, N)``.
    """
    N = P_pred.shape[0]
    I_KH = jnp.eye(N, dtype=P_pred.dtype) - K @ H  # (N, N)
    P_update = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    return (P_update + P_update.T) / 2
