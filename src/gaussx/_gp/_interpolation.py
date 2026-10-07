"""Conditional interpolation between time points for state-space models."""

from __future__ import annotations

import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._linalg._linalg import solve_columns, solve_rows
from gaussx._linalg._symmetrize import symmetrize
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
    r"""Two-filter fusion at ``t`` from a filtered state and a backward message.

    For an SDE-discretised state-space model with no observation at ``t``,

    $$
    x_t \mid x_{t^-} \sim \mathcal{N}(A_{fwd} x_{t^-}, Q_{fwd}), \qquad
    x_{t^+} \mid x_t \sim \mathcal{N}(A_{bwd} x_t, Q_{bwd}),
    $$

    combines two **conditionally independent** sources of information about
    ``x_t`` by adding precisions (the two-filter smoother):

    - ``(mu_prev, P_prev)``: the **forward-filtered** state at ``t^-``,
      ``p(x_{t^-} \mid y_{\le t^-}) = N(mu_prev, P_prev)``;
    - ``(mu_next, P_next)``: the **backward-information message** at
      ``t^+``, i.e. the likelihood of the data at and after ``t^+`` written
      as ``p(y_{\ge t^+} \mid x_{t^+}) \propto N(mu_next; x_{t^+}, P_next)``
      (no prior on ``x_{t^+}``).

    With those inputs it returns ``p(x_t \mid y_{\le t^-}, y_{\ge t^+})``
    exactly:

    $$
    \begin{aligned}
    m_{fwd} &= A_{fwd}\,\mu_{prev}, \quad
    P_{fwd} = A_{fwd} P_{prev} A_{fwd}^\top + Q_{fwd}, \\
    \Lambda_{bwd} &= A_{bwd}^\top (P_{next} + Q_{bwd})^{-1} A_{bwd}, \quad
    \eta_{bwd} = A_{bwd}^\top (P_{next} + Q_{bwd})^{-1} \mu_{next}, \\
    P &= (P_{fwd}^{-1} + \Lambda_{bwd})^{-1}, \quad
    \mu = P\,(P_{fwd}^{-1} m_{fwd} + \eta_{bwd}).
    \end{aligned}
    $$

    Warning:
        Do **not** pass smoothed marginals (e.g. from `rts_smoother`) for
        both ends: each already contains all the data, so the data is
        counted twice and the result is over-confident (about 20% too small
        a covariance on a 3-state chain, gh-288). To interpolate from a
        filtered state at ``t^-`` and a smoothed state at ``t^+``, use
        `rts_interpolate`.

    Args:
        A_fwd: Forward transition from ``t^-`` to ``t``, shape ``(d, d)``.
        Q_fwd: Forward process noise, shape ``(d, d)``.
        A_bwd: Transition from ``t`` to ``t^+``, shape ``(d, d)``.
        Q_bwd: Process noise from ``t`` to ``t^+``, shape ``(d, d)``.
        mu_prev: Forward-filtered mean at ``t^-``, shape ``(d,)``.
        P_prev: Forward-filtered covariance at ``t^-``, shape ``(d, d)``.
        mu_next: Mean of the backward-information message at ``t^+``,
            shape ``(d,)``.
        P_next: Covariance of the backward-information message at ``t^+``,
            shape ``(d, d)``.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(mean, cov)`` of ``x_t`` given the data on both sides.
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


def rts_interpolate(
    A_fwd: Float[Array, "d d"],
    Q_fwd: Float[Array, "d d"],
    A_bwd: Float[Array, "d d"],
    Q_bwd: Float[Array, "d d"],
    filtered_mean_prev: Float[Array, " d"],
    filtered_cov_prev: Float[Array, "d d"],
    smoothed_mean_next: Float[Array, " d"],
    smoothed_cov_next: Float[Array, "d d"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
    r"""Smoothed marginal at an unobserved ``t`` from filtered and smoothed states.

    For the model of `conditional_interpolate` (no observation at ``t``,
    ``t^- < t < t^+``), takes the **forward-filtered** state at ``t^-`` and
    the **smoothed** state at ``t^+`` — what `kalman_filter` and
    `rts_smoother` return — and performs one Rauch-Tung-Striebel backward
    step through ``t``:

    $$
    \begin{aligned}
    m_t^- &= A_{fwd} m_f, \quad
    P_t^- = A_{fwd} P_f A_{fwd}^\top + Q_{fwd}, \\
    m_+ &= A_{bwd} m_t^-, \quad
    P_+ = A_{bwd} P_t^- A_{bwd}^\top + Q_{bwd}, \\
    G &= P_t^- A_{bwd}^\top P_+^{-1}, \\
    m_t &= m_t^- + G (m_s - m_+), \quad
    P_t = P_t^- + G (P_s - P_+) G^\top .
    \end{aligned}
    $$

    This is exact: given ``x_{t^+}``, ``x_t`` is independent of the data at
    and after ``t^+``, so
    ``p(x_t | y) = \int p(x_t | x_{t^+}, y_{\le t^-}) p(x_{t^+} | y) dx_{t^+}``.

    Args:
        A_fwd: Forward transition from ``t^-`` to ``t``, shape ``(d, d)``.
        Q_fwd: Forward process noise, shape ``(d, d)``.
        A_bwd: Transition from ``t`` to ``t^+``, shape ``(d, d)``.
        Q_bwd: Process noise from ``t`` to ``t^+``, shape ``(d, d)``.
        filtered_mean_prev: Filtered mean ``m_f`` at ``t^-``, shape ``(d,)``.
        filtered_cov_prev: Filtered covariance ``P_f`` at ``t^-``,
            shape ``(d, d)``.
        smoothed_mean_next: Smoothed mean ``m_s`` at ``t^+``, shape ``(d,)``.
        smoothed_cov_next: Smoothed covariance ``P_s`` at ``t^+``,
            shape ``(d, d)``.
        solver: Optional solver strategy for the ``P_+^{-1}`` solve.

    Returns:
        Tuple ``(mean, cov)`` — the smoothed marginal of ``x_t``.
    """
    m_pred = A_fwd @ filtered_mean_prev
    P_pred = symmetrize(
        einsum(A_fwd, filtered_cov_prev, A_fwd, "i j, j k, l k -> i l") + Q_fwd
    )
    m_plus = A_bwd @ m_pred
    P_plus = symmetrize(einsum(A_bwd, P_pred, A_bwd, "i j, j k, l k -> i l") + Q_bwd)
    P_plus_op = lx.MatrixLinearOperator(P_plus, lx.positive_semidefinite_tag)
    # G = P_pred A_bwd^T P_plus^{-1}, row by row (P_plus is symmetric).
    G = solve_rows(P_plus_op, einsum(P_pred, A_bwd, "i j, k j -> i k"), solver=solver)
    m = m_pred + G @ (smoothed_mean_next - m_plus)
    P = symmetrize(
        P_pred + einsum(G, smoothed_cov_next - P_plus, G, "i j, j k, l k -> i l")
    )
    return m, P
