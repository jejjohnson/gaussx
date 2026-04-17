"""Discrete Algebraic Riccati Equation (DARE) solver."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class DAREResult(eqx.Module):
    """Result of DARE solver.

    Attributes:
        P_inf: Steady-state covariance, shape ``(D, D)``.
        K_inf: Steady-state Kalman gain, shape ``(D, M)``.
        converged: Scalar boolean indicating convergence.
    """

    P_inf: Float[Array, "D D"]
    K_inf: Float[Array, "D M"]
    converged: Bool[Array, ""]


def dare(
    A: Float[Array, "D D"],
    H: Float[Array, "M D"],
    Q: Float[Array, "D D"],
    R: Float[Array, "M M"],
    *,
    P_init: Float[Array, "D D"] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> DAREResult:
    """Discrete Algebraic Riccati Equation solver.

    Iterates the Kalman predict-update equations until convergence::

        Predict:  P⁻ = A P Aᵀ + Q
        Update:   S = H P⁻ Hᵀ + R
                  K = P⁻ Hᵀ S⁻¹
                  P = (I - KH) P⁻

    Convergence is declared when ``max|P_new - P_old| < tol``.

    Args:
        A: Transition matrix, shape ``(D, D)``.
        H: Observation matrix, shape ``(M, D)``.
        Q: Process noise covariance, shape ``(D, D)``.
        R: Observation noise covariance, shape ``(M, M)``.
        P_init: Initial covariance guess, shape ``(D, D)``. Defaults to ``Q``.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the element-wise max absolute change.

    Returns:
        A :class:`DAREResult` containing the steady-state covariance,
        Kalman gain, and convergence flag.
    """
    if P_init is None:
        P_init = Q

    D = A.shape[0]
    I_D = jnp.eye(D)

    def _step(
        P: Float[Array, "D D"],
    ) -> tuple[Float[Array, "D D"], Float[Array, "D M"]]:
        """One predict-update step. Returns ``(P_new, K)``."""
        P_pred = A @ P @ A.T + Q
        S = H @ P_pred @ H.T + R
        # K = P_pred @ H.T @ S⁻¹, computed via solve for numerical stability.
        K = jnp.linalg.solve(S, H @ P_pred).T
        P_new = (I_D - K @ H) @ P_pred
        return P_new, K

    def _cond(
        state: tuple[Float[Array, "D D"], int, Bool[Array, ""]],
    ) -> Bool[Array, ""]:
        _, i, converged = state
        return (i < max_iter) & (~converged)

    def _body(
        state: tuple[Float[Array, "D D"], int, Bool[Array, ""]],
    ) -> tuple[Float[Array, "D D"], int, Bool[Array, ""]]:
        P_old, i, _ = state
        P_new, _ = _step(P_old)
        converged = jnp.max(jnp.abs(P_new - P_old)) < tol
        return P_new, i + 1, converged

    init_state = (P_init, 0, jnp.array(False))
    P_inf, _, converged = jax.lax.while_loop(_cond, _body, init_state)

    # Compute the final gain from the converged covariance.
    _, K_inf = _step(P_inf)

    return DAREResult(P_inf=P_inf, K_inf=K_inf, converged=converged)
