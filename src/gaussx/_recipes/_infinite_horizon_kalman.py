"""Infinite-horizon Kalman filter and smoother using steady-state gains."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from gaussx._recipes._dare import DAREResult, dare


class InfiniteHorizonState(eqx.Module):
    """Output of ``infinite_horizon_filter``.

    Attributes:
        filtered_means: Shape ``(T, N)`` -- filtered state estimates.
        filtered_covs: Shape ``(T, N, N)`` -- filtered covariances (constant).
        predicted_means: Shape ``(T, N)`` -- predicted state estimates.
        predicted_covs: Shape ``(T, N, N)`` -- predicted covariances (constant).
        log_likelihood: Scalar -- total log-likelihood.
    """

    filtered_means: jnp.ndarray
    filtered_covs: jnp.ndarray
    predicted_means: jnp.ndarray
    predicted_covs: jnp.ndarray
    log_likelihood: jnp.ndarray


def infinite_horizon_filter(
    transition: jnp.ndarray,
    obs_model: jnp.ndarray,
    process_noise: jnp.ndarray,
    obs_noise: jnp.ndarray,
    observations: jnp.ndarray,
    *,
    dare_result: DAREResult | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> InfiniteHorizonState:
    """Infinite-horizon Kalman filter with fixed steady-state gain.

    Uses the DARE solution for a constant Kalman gain, avoiding per-step
    Riccati updates. Much cheaper per step than the standard Kalman filter.

    Args:
        transition: State transition matrix A, shape ``(N, N)``.
        obs_model: Observation matrix H, shape ``(M, N)``.
        process_noise: Process noise covariance Q, shape ``(N, N)``.
        obs_noise: Observation noise covariance R, shape ``(M, M)``.
        observations: Observed data y, shape ``(T, M)``.
        dare_result: Precomputed DARE result. If ``None``, calls
            ``dare()`` internally.
        max_iter: Maximum DARE iterations (used only if ``dare_result``
            is ``None``).
        tol: DARE convergence tolerance (used only if ``dare_result``
            is ``None``).

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
        )

    P_inf = dare_result.P_inf
    K_inf = dare_result.K_inf
    T = observations.shape[0]
    M = obs_model.shape[0]
    N = transition.shape[0]

    # Precompute steady-state quantities
    P_pred_inf = transition @ P_inf @ transition.T + process_noise
    S_inf = obs_model @ P_pred_inf @ obs_model.T + obs_noise
    L_S = jnp.linalg.cholesky(S_inf)
    ld_inf = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)))
    log_2pi = jnp.log(2.0 * jnp.pi)

    # Steady-state filtered covariance
    I_N = jnp.eye(N)
    P_filt_inf = (I_N - K_inf @ obs_model) @ P_pred_inf

    def step(carry, y_t):
        x_filt, ll = carry

        # Predict
        x_pred = transition @ x_filt

        # Update with fixed gain
        v = y_t - obs_model @ x_pred
        x_filt_new = x_pred + K_inf @ v

        # Log-likelihood increment
        Sinv_v = jnp.linalg.solve(S_inf, v)
        ll_inc = -0.5 * (v @ Sinv_v + ld_inf + M * log_2pi)

        return (x_filt_new, ll + ll_inc), (x_filt_new, x_pred)

    init_mean = jnp.zeros(N)
    init_carry = (init_mean, jnp.array(0.0))
    (_, total_ll), (f_means, p_means) = jax.lax.scan(
        step,
        init_carry,
        observations,
    )

    # Tile constant covariances to (T, N, N)
    f_covs = jnp.broadcast_to(P_filt_inf[None], (T, N, N))
    p_covs = jnp.broadcast_to(P_pred_inf[None], (T, N, N))

    return InfiniteHorizonState(
        filtered_means=f_means,
        filtered_covs=f_covs,
        predicted_means=p_means,
        predicted_covs=p_covs,
        log_likelihood=total_ll,
    )


def infinite_horizon_smoother(
    filter_state: InfiniteHorizonState,
    transition: jnp.ndarray,
    dare_result: DAREResult,
    process_noise: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Infinite-horizon RTS smoother with fixed steady-state gain.

    Args:
        filter_state: Output of ``infinite_horizon_filter``.
        transition: State transition matrix A, shape ``(N, N)``.
        dare_result: DARE result used in the filter.
        process_noise: Process noise covariance Q, shape ``(N, N)``.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` with shapes
        ``(T, N)`` and ``(T, N, N)``.
    """
    P_inf = dare_result.P_inf
    P_pred_inf = transition @ P_inf @ transition.T + process_noise

    # Steady-state smoother gain: G = P_inf @ A^T @ P_pred^{-1}
    N = transition.shape[0]
    G_inf = jnp.linalg.solve(P_pred_inf.T, (P_inf @ transition.T).T).T

    # Steady-state smoothed covariance
    P_smooth_inf = P_inf + G_inf @ (P_inf - P_pred_inf) @ G_inf.T

    T = filter_state.filtered_means.shape[0]

    def step(carry, inputs):
        x_smooth = carry
        x_filt, x_pred = inputs
        x_smooth_new = x_filt + G_inf @ (x_smooth - x_pred)
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
    )
    s_covs = jnp.broadcast_to(P_smooth_inf[None], (T, N, N))

    return s_means, s_covs
