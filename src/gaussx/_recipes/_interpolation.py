"""Conditional interpolation between time points for state-space models."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def conditional_interpolate(
    A_fwd: Float[Array, "d d"],
    Q_fwd: Float[Array, "d d"],
    A_bwd: Float[Array, "d d"],
    Q_bwd: Float[Array, "d d"],
    mu_prev: Float[Array, " d"],
    P_prev: Float[Array, "d d"],
    mu_next: Float[Array, " d"],
    P_next: Float[Array, "d d"],
) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
    r"""Interpolated marginal at time ``t`` given posteriors at ``t^-`` and ``t^+``.

    For an SDE-discretized state-space model::

        x_t     | x_{t^-} \sim N(A_{fwd} x_{t^-}, Q_{fwd})
        x_{t^+} | x_t     \sim N(A_{bwd} x_t,     Q_{bwd})

    computes ``p(x_t | x_{t^-}, x_{t^+})`` using information fusion
    of the forward and backward predictions::

        \Lambda_{fwd} = (A_{fwd} P_{prev} A_{fwd}^T + Q_{fwd})^{-1}
        \Lambda_{bwd} = A_{bwd}^T (P_{next} + Q_{bwd})^{-1} A_{bwd}
        \Lambda = \Lambda_{fwd} + \Lambda_{bwd}
        P = \Lambda^{-1}
        \mu = P (\Lambda_{fwd} m_{fwd} + \Lambda_{bwd,1})

    Args:
        A_fwd: Forward transition from ``t^-`` to ``t``, shape ``(d, d)``.
        Q_fwd: Forward process noise, shape ``(d, d)``.
        A_bwd: Backward transition from ``t`` to ``t^+``, shape ``(d, d)``.
        Q_bwd: Backward process noise, shape ``(d, d)``.
        mu_prev: Marginal mean at ``t^-``, shape ``(d,)``.
        P_prev: Marginal covariance at ``t^-``, shape ``(d, d)``.
        mu_next: Marginal mean at ``t^+``, shape ``(d,)``.
        P_next: Marginal covariance at ``t^+``, shape ``(d, d)``.

    Returns:
        Tuple ``(mean, cov)`` — interpolated marginal at ``t``.
    """
    # Forward prediction to t
    m_fwd = A_fwd @ mu_prev
    P_fwd = A_fwd @ P_prev @ A_fwd.T + Q_fwd

    # Forward information
    Lambda_fwd = jnp.linalg.inv(P_fwd)
    eta1_fwd = jnp.linalg.solve(P_fwd, m_fwd)

    # Backward information from t+
    S_bwd = P_next + Q_bwd
    Lambda_bwd = A_bwd.T @ jnp.linalg.solve(S_bwd, A_bwd)
    eta1_bwd = A_bwd.T @ jnp.linalg.solve(S_bwd, mu_next)

    # Fuse forward and backward
    Lambda = Lambda_fwd + Lambda_bwd
    P = jnp.linalg.inv(Lambda)
    m = P @ (eta1_fwd + eta1_bwd)

    return m, P
