"""Discrete Algebraic Riccati Equation (DARE) solver."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum, rearrange
from gaussx._linalg._linalg import solve_matrix
from gaussx._linalg._symmetrize import symmetrize
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
    r"""Steady-state Kalman filter covariance and gain (the filtering DARE).

    Solves for the fixed point of the Kalman predict-update recursion,

        Predict:  P⁻ = A P Aᵀ + Q
        Update:   S = H P⁻ Hᵀ + R
                  K = P⁻ Hᵀ S⁻¹
                  P = (I - KH) P⁻

    with the structure-preserving doubling algorithm (SDA; Chu, Fan & Lin,
    2005) applied to the predicted-covariance Riccati equation
    ``P⁻ = A P⁻ Aᵀ − A P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹ H P⁻ Aᵀ + Q``. SDA converges
    quadratically -- after ``k`` doublings the error is
    ``O(rho^(2^k))`` for contraction rate ``rho`` -- so slowly mixing
    dynamics (``A`` near the unit circle) need a dozen or so doublings
    where iterating the recursion itself needs thousands (gh-294). Each
    doubling costs a few ``D × D`` solves.

    Convergence is declared when the doubling iterate changes by at most
    ``tol`` relative to its largest entry; by quadratic convergence the
    returned solution is then accurate to about ``tol²``. Check
    ``converged`` when calling ``dare`` directly;
    `infinite_horizon_filter` and `infinite_horizon_smoother` raise on a
    non-converged result.

    Requires ``R`` invertible and the usual stabilisability /
    detectability conditions for a unique stabilising solution.

    Args:
        A: Transition matrix or operator, shape ``(D, D)``.
        H: Observation matrix or operator, shape ``(M, D)``.
        Q: Process noise covariance or operator, shape ``(D, D)``.
        R: Observation noise covariance or operator, shape ``(M, M)``.
        P_init: Deprecated and ignored: the doubling algorithm needs no
            initial guess.
        max_iter: Maximum number of doubling steps.
        tol: Convergence tolerance on the relative change of the iterate.
        solver: Optional solver strategy for structured linear algebra
            (``R`` solves and the final gain). When ``None``, falls back to
            structural dispatch.
        woodbury_innovation: When ``True``, build ``S = H P⁻ Hᵀ + R``
            as a `gaussx.LowRankUpdate` so structured ``R`` uses
            Woodbury solves for the final gain.

    Returns:
        A `DAREResult` containing the steady-state *filtered* covariance,
        Kalman gain, and convergence flag.
    """
    if P_init is not None:
        warn_deprecated(
            "dare(P_init=...) is deprecated and ignored: the doubling "
            "algorithm needs no initial guess. It will be removed in gaussx 0.7.0."
        )
    A_op = _as_operator(A)
    H_op = _as_operator(H)
    A_dense = A_op.as_matrix()
    H_dense = H_op.as_matrix()
    Q_dense = _materialise(Q)
    # Keep ``R`` lazy when the Woodbury innovation path will consume the
    # operator directly — avoids an O(M²) allocation for large structured
    # noise (e.g. ``DiagonalLinearOperator`` with large ``M``).
    R_for_innovation = (
        R
        if woodbury_innovation and isinstance(R, lx.AbstractLinearOperator)
        else _materialise(R)
    )

    # SDA on the control-form DARE with A_c = Aᵀ, B_c = Hᵀ: the iterate
    # ``H_k`` converges to the predicted steady-state covariance P⁻.
    dtype = jnp.result_type(A_dense, H_dense, Q_dense)
    eye = jnp.eye(A_dense.shape[0], dtype=dtype)
    Rinv_H = solve_matrix(_as_operator(R), H_dense, solver=solver)
    G_0 = symmetrize(einsum(H_dense, Rinv_H, "m i, m j -> i j"))
    A_0 = rearrange(A_dense, "i j -> j i")

    def _cond(state):
        *_, i, converged = state
        return (i < max_iter) & (~converged)

    def _body(state):
        A_k, G_k, H_k, i, _ = state
        W = eye + G_k @ H_k
        Winv_A = jnp.linalg.solve(W, A_k)
        Winv_G = jnp.linalg.solve(W, G_k)
        A_next = A_k @ Winv_A
        G_next = symmetrize(G_k + A_k @ Winv_G @ rearrange(A_k, "i j -> j i"))
        H_next = symmetrize(H_k + rearrange(A_k, "i j -> j i") @ H_k @ Winv_A)
        change = jnp.max(jnp.abs(H_next - H_k))
        converged = change <= tol * jnp.max(jnp.abs(H_next))
        return A_next, G_next, H_next, i + 1, converged

    init_state = (A_0, G_0, Q_dense.astype(dtype), 0, jnp.array(False))
    *_, P_pred, _, converged = jax.lax.while_loop(_cond, _body, init_state)

    # Filtered covariance and gain from the steady-state prediction.
    S_op = _innovation_covariance(
        H_op, P_pred, R_for_innovation, woodbury=woodbury_innovation
    )
    HP_pred = _left_matmul(H_op, P_pred)
    K_inf = solve_matrix(S_op, HP_pred, solver=solver).T
    P_inf = symmetrize(P_pred - K_inf @ HP_pred)

    return DAREResult(P_inf=P_inf, K_inf=K_inf, converged=converged)
