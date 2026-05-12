"""Infinite-horizon Kalman filter and smoother using steady-state gains."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from einops import repeat
from jaxtyping import Array, Float

from gaussx._linalg._linalg import sandwich, solve_rows
from gaussx._linalg._lyapunov import discrete_lyapunov_solve
from gaussx._primitives._cholesky import cholesky
from gaussx._ssm._dare import DAREResult, dare
from gaussx._ssm._utils import (
    _as_operator,
    _materialise,
    _matvec,
    _right_matmul_transpose,
)
from gaussx._strategies._base import AbstractSolverStrategy


class InfiniteHorizonState(eqx.Module):
    """Output of ``infinite_horizon_filter``.

    Attributes:
        filtered_means: Filtered state estimates, shape ``(T, N)``.
        filtered_covs: Filtered covariances (constant), shape ``(T, N, N)``.
        predicted_means: Predicted state estimates, shape ``(T, N)``.
        predicted_covs: Predicted covariances (constant), shape ``(T, N, N)``.
        log_likelihood: Total log-likelihood (scalar).
    """

    filtered_means: Float[Array, "T N"]
    filtered_covs: Float[Array, "T N N"]
    predicted_means: Float[Array, "T N"]
    predicted_covs: Float[Array, "T N N"]
    log_likelihood: Float[Array, ""]


def infinite_horizon_filter(
    transition: Float[Array, "N N"] | lx.AbstractLinearOperator,
    obs_model: Float[Array, "M N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "N N"] | lx.AbstractLinearOperator,
    obs_noise: Float[Array, "M M"] | lx.AbstractLinearOperator,
    observations: Float[Array, "T M"],
    init_mean: Float[Array, " N"] | None = None,
    *,
    dare_result: DAREResult | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    solver: AbstractSolverStrategy | None = None,
) -> InfiniteHorizonState:
    """Infinite-horizon Kalman filter with fixed steady-state gain.

    Uses the DARE solution for a constant Kalman gain K∞, avoiding
    per-step Riccati updates.  For dense matrices, the per-step cost is
    O(N² + MN + M²) instead of O(N³) for the standard Kalman filter::

        Predict:  x⁻ₜ = A xₜ₋₁
        Update:   vₜ  = yₜ − H x⁻ₜ
                  xₜ  = x⁻ₜ + K∞ vₜ

    All four operator/array arguments accept either a raw JAX array or
    a :class:`lineax.AbstractLinearOperator`. Operator inputs preserve
    their structural matvec inside the per-step scan; the sandwiches
    materialise once outside the scan.

    Args:
        transition: State transition matrix or operator, shape ``(N, N)``.
        obs_model: Observation matrix or operator, shape ``(M, N)``.
        process_noise: Process noise covariance or operator, shape ``(N, N)``.
        obs_noise: Observation noise covariance or operator, shape ``(M, M)``.
        observations: Observed data y, shape ``(T, M)``.
        init_mean: Initial state mean, shape ``(N,)``. Defaults to zeros.
        dare_result: Precomputed DARE result. If ``None``, calls
            ``dare()`` internally.
        max_iter: Maximum DARE iterations (used only if ``dare_result``
            is ``None``).
        tol: DARE convergence tolerance (used only if ``dare_result``
            is ``None``).
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        An ``InfiniteHorizonState`` with filtered/predicted means,
        covariances, and total log-likelihood.
    """
    if dare_result is None:
        dare_result = dare(
            transition,
            obs_model,
            process_noise,
            obs_noise,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
        )

    A_op = _as_operator(transition)
    H_op = _as_operator(obs_model)
    Q_dense = _materialise(process_noise)
    R_dense = _materialise(obs_noise)

    P_inf = dare_result.P_inf  # (N, N)
    K_inf = dare_result.K_inf  # (N, M)
    T = observations.shape[0]
    M = observations.shape[-1]
    N = A_op.out_size()

    # Precompute steady-state quantities
    P_inf_op = lx.MatrixLinearOperator(P_inf, lx.positive_semidefinite_tag)
    P_pred_inf = sandwich(A_op, P_inf_op).as_matrix() + Q_dense  # (N, N)
    P_pred_inf_op = lx.MatrixLinearOperator(P_pred_inf, lx.positive_semidefinite_tag)
    S_inf = sandwich(H_op, P_pred_inf_op).as_matrix() + R_dense  # (M, M)
    L_S = cholesky(  # (M, M)
        lx.MatrixLinearOperator(S_inf, lx.positive_semidefinite_tag)
    ).as_matrix()
    from gaussx._primitives._logdet import cholesky_logdet

    ld_inf = cholesky_logdet(L_S)  # scalar
    log_2pi = jnp.log(2.0 * jnp.pi)

    # Steady-state filtered covariance: P_filt = (I − K∞ H) P⁻pred
    HP_pred_inf = _right_matmul_transpose(P_pred_inf, H_op).T
    P_filt_inf = P_pred_inf - K_inf @ HP_pred_inf  # (N, N)

    def step(carry, y_t):
        x_filt, ll = carry

        x_pred = _matvec(transition, x_filt)  # (N,)
        v = y_t - _matvec(obs_model, x_pred)  # (M,)  innovation
        x_filt_new = x_pred + K_inf @ v  # (N,)

        # Log-likelihood increment (reuse precomputed Cholesky L_S)
        Sinv_v = jax.scipy.linalg.cho_solve((L_S, True), v)  # (M,)
        ll_inc = -0.5 * (v @ Sinv_v + ld_inf + M * log_2pi)

        return (x_filt_new, ll + ll_inc), (x_filt_new, x_pred)

    if init_mean is None:
        init_mean = jnp.zeros(N)
    init_carry = (init_mean, jnp.array(0.0))
    (_, total_ll), (f_means, p_means) = jax.lax.scan(
        step,
        init_carry,
        observations,
    )

    # Broadcast constant covariances to (T, N, N)
    f_covs = repeat(P_filt_inf, "n1 n2 -> T n1 n2", T=T)
    p_covs = repeat(P_pred_inf, "n1 n2 -> T n1 n2", T=T)

    return InfiniteHorizonState(
        filtered_means=f_means,
        filtered_covs=f_covs,
        predicted_means=p_means,
        predicted_covs=p_covs,
        log_likelihood=total_ll,
    )


def infinite_horizon_smoother(
    filter_state: InfiniteHorizonState,
    transition: Float[Array, "N N"] | lx.AbstractLinearOperator,
    dare_result: DAREResult,
    process_noise: Float[Array, "N N"] | lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Infinite-horizon RTS smoother with fixed steady-state gain.

    Precomputes the steady-state smoother gain G∞ = P∞ Aᵀ P⁻pred⁻¹,
    then runs a backward scan with fixed G∞.  The steady-state smoothed
    covariance is the solution of the discrete Lyapunov equation::

        P_smooth = P∞ + G∞ (P_smooth − P⁻pred) G∞ᵀ

    Args:
        filter_state: Output of ``infinite_horizon_filter``.
        transition: State transition matrix or operator, shape ``(N, N)``.
        dare_result: DARE result used in the filter.
        process_noise: Process noise covariance or operator, shape ``(N, N)``.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` with shapes
        ``(T, N)`` and ``(T, N, N)``.
    """
    A_op = _as_operator(transition)
    Q_dense = _materialise(process_noise)
    P_inf = dare_result.P_inf  # (N, N)
    P_inf_op = lx.MatrixLinearOperator(P_inf, lx.positive_semidefinite_tag)
    P_pred_inf = sandwich(A_op, P_inf_op).as_matrix() + Q_dense  # (N, N)

    # Steady-state smoother gain: G∞ = P∞ Aᵀ P⁻pred⁻¹
    P_pred_inf_op = lx.MatrixLinearOperator(P_pred_inf, lx.positive_semidefinite_tag)
    G_inf = solve_rows(
        P_pred_inf_op,
        _right_matmul_transpose(P_inf, A_op),
        solver=solver,
    )  # (N, N)

    # Solve discrete Lyapunov equation:
    # P_smooth = P∞ + G∞ (P_smooth − P⁻pred) G∞ᵀ
    # ⟺ P_smooth − G∞ P_smooth G∞ᵀ = P∞ − G∞ P⁻pred G∞ᵀ
    # Routed through :func:`discrete_lyapunov_solve` which uses a
    # per-factor eigendecomposition of ``G∞`` instead of materializing
    # the ``(N², N²)`` Kronecker matrix ``I − G∞ ⊗ G∞``.
    rhs = P_inf - G_inf @ P_pred_inf @ G_inf.T  # (N, N)
    P_smooth_inf = discrete_lyapunov_solve(G_inf, rhs)
    P_smooth_inf = 0.5 * (P_smooth_inf + P_smooth_inf.T)  # enforce symmetry

    T = filter_state.filtered_means.shape[0]

    def step(carry, inputs):
        x_smooth = carry
        x_filt, x_pred = inputs
        x_smooth_new = x_filt + G_inf @ (x_smooth - x_pred)  # (N,)
        return x_smooth_new, x_smooth_new

    init = filter_state.filtered_means[T - 1]
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
    )

    _, s_means_rev = jax.lax.scan(step, init, inputs)

    s_means = jnp.concatenate(
        [s_means_rev[::-1], filter_state.filtered_means[T - 1 :]],
        axis=0,
    )  # (T, N)
    s_covs = repeat(P_smooth_inf, "n1 n2 -> T n1 n2", T=T)  # (T, N, N)

    return s_means, s_covs
