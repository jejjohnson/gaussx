"""Conditional interpolation between time points for state-space models."""

from __future__ import annotations

import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._linalg import solve_columns
from gaussx._primitives._inv import inv
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_solve


def conditional_interpolate(
    A_fwd: Float[Array, "d d"],
    Q_fwd: Float[Array, "d d"],
    A_bwd: Float[Array, "d d"],
    Q_bwd: Float[Array, "d d"],
    mu_prev: Float[Array, " d"],
    P_prev: Float[Array, "d d"],
    mu_next: Float[Array, " d"],
    P_next: Float[Array, "d d"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
    r"""Interpolated marginal at time ``t`` given posteriors at ``t^-`` and ``t^+``.

    For an SDE-discretized state-space model:

        x_t     | x_{t^-} \sim N(A_{fwd} x_{t^-}, Q_{fwd})
        x_{t^+} | x_t     \sim N(A_{bwd} x_t,     Q_{bwd})

    computes ``p(x_t | x_{t^-}, x_{t^+})`` using information fusion
    of the forward and backward predictions:

        \Lambda_{fwd} = (A_{fwd} P_{prev} A_{fwd}^T + Q_{fwd})^{-1}
        \Lambda_{bwd} = A_{bwd}^T (P_{next} + Q_{bwd})^{-1} A_{bwd}
        \eta_{1,fwd} = \Lambda_{fwd} m_{fwd}
        \eta_{1,bwd} = A_{bwd}^T (P_{next} + Q_{bwd})^{-1} \mu_{next}
        \Lambda = \Lambda_{fwd} + \Lambda_{bwd}
        P = \Lambda^{-1}
        \mu = P (\eta_{1,fwd} + \eta_{1,bwd})

    Args:
        A_fwd: Forward transition from ``t^-`` to ``t``, shape ``(d, d)``.
        Q_fwd: Forward process noise, shape ``(d, d)``.
        A_bwd: Backward transition from ``t`` to ``t^+``, shape ``(d, d)``.
        Q_bwd: Backward process noise, shape ``(d, d)``.
        mu_prev: Marginal mean at ``t^-``, shape ``(d,)``.
        P_prev: Marginal covariance at ``t^-``, shape ``(d, d)``.
        mu_next: Marginal mean at ``t^+``, shape ``(d,)``.
        P_next: Marginal covariance at ``t^+``, shape ``(d, d)``.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(mean, cov)`` — interpolated marginal at ``t``.
    """
    # Forward prediction to t
    m_fwd = A_fwd @ mu_prev
    P_fwd = A_fwd @ P_prev @ A_fwd.T + Q_fwd

    # Forward information
    P_fwd_op = lx.MatrixLinearOperator(P_fwd, lx.positive_semidefinite_tag)
    Lambda_fwd = inv(P_fwd_op).as_matrix()
    eta1_fwd = dispatch_solve(P_fwd_op, m_fwd, solver)

    # Backward information from t+
    S_bwd = P_next + Q_bwd
    S_bwd_op = lx.MatrixLinearOperator(S_bwd, lx.positive_semidefinite_tag)
    Lambda_bwd = A_bwd.T @ solve_columns(S_bwd_op, A_bwd, solver=solver)
    eta1_bwd = A_bwd.T @ dispatch_solve(S_bwd_op, mu_next, solver)

    # Fuse forward and backward
    Lambda = Lambda_fwd + Lambda_bwd
    Lambda_op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
    P = inv(Lambda_op).as_matrix()
    m = P @ (eta1_fwd + eta1_bwd)

    return m, P
