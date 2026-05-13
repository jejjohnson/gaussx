"""Discrete Algebraic Riccati Equation (DARE) solver."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._linalg._linalg import sandwich, solve_matrix
from gaussx._ssm._utils import (
    _as_operator,
    _innovation_covariance,
    _left_matmul,
    _materialise,
)
from gaussx._strategies._base import AbstractSolverStrategy


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
    A: Float[Array, "D D"] | lx.AbstractLinearOperator,
    H: Float[Array, "M D"] | lx.AbstractLinearOperator,
    Q: Float[Array, "D D"] | lx.AbstractLinearOperator,
    R: Float[Array, "M M"] | lx.AbstractLinearOperator,
    *,
    P_init: Float[Array, "D D"] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    solver: AbstractSolverStrategy | None = None,
    woodbury_innovation: bool = False,
) -> DAREResult:
    """Discrete Algebraic Riccati Equation solver.

    Iterates the Kalman predict-update equations until convergence::

        Predict:  P⁻ = A P Aᵀ + Q
        Update:   S = H P⁻ Hᵀ + R
                  K = P⁻ Hᵀ S⁻¹
                  P = (I - KH) P⁻

    Convergence is declared when ``max|P_new - P_old| < tol``.

    Args:
        A: Transition matrix or operator, shape ``(D, D)``.
        H: Observation matrix or operator, shape ``(M, D)``.
        Q: Process noise covariance or operator, shape ``(D, D)``.
        R: Observation noise covariance or operator, shape ``(M, M)``.
        P_init: Initial covariance guess, shape ``(D, D)``. Defaults to ``Q``.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the element-wise max absolute change.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.
        woodbury_innovation: When ``True``, build ``S = H P⁻ Hᵀ + R``
            as a :class:`gaussx.LowRankUpdate` so structured ``R`` uses
            Woodbury solves.

    Returns:
        A :class:`DAREResult` containing the steady-state covariance,
        Kalman gain, and convergence flag.
    """
    A_op = _as_operator(A)
    H_op = _as_operator(H)
    Q_dense = _materialise(Q)
    # Keep ``R`` lazy when the Woodbury innovation path will consume the
    # operator directly — avoids an O(M²) allocation for large structured
    # noise (e.g. ``DiagonalLinearOperator`` with large ``M``).
    R_for_innovation = (
        R
        if woodbury_innovation and isinstance(R, lx.AbstractLinearOperator)
        else _materialise(R)
    )

    if P_init is None:
        P_init = Q_dense

    def _step(
        P: Float[Array, "D D"],
    ) -> tuple[Float[Array, "D D"], Float[Array, "D M"]]:
        """One predict-update step. Returns ``(P_new, K)``."""
        P_op = lx.MatrixLinearOperator(P, lx.positive_semidefinite_tag)
        P_pred = sandwich(A_op, P_op).as_matrix() + Q_dense
        # K = P_pred @ H.T @ S⁻¹, computed via a single factorization
        # on the matrix RHS for numerical stability and efficiency.
        S_op = _innovation_covariance(
            H_op, P_pred, R_for_innovation, woodbury=woodbury_innovation
        )
        HP_pred = _left_matmul(H_op, P_pred)
        K = solve_matrix(S_op, HP_pred, solver=solver).T
        P_new = P_pred - K @ HP_pred
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
