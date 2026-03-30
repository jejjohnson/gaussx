"""Kalman filter, RTS smoother, and Kalman gain recipes."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._primitives._solve import solve


class FilterState(eqx.Module):
    """Output of ``kalman_filter``.

    Attributes:
        filtered_means: Shape ``(T, N)`` — filtered state estimates.
        filtered_covs: Shape ``(T, N, N)`` — filtered covariances.
        predicted_means: Shape ``(T, N)`` — predicted state estimates.
        predicted_covs: Shape ``(T, N, N)`` — predicted covariances.
        log_likelihood: Scalar — total log-likelihood.
    """

    filtered_means: jnp.ndarray
    filtered_covs: jnp.ndarray
    predicted_means: jnp.ndarray
    predicted_covs: jnp.ndarray
    log_likelihood: jnp.ndarray


def kalman_filter(
    transition: jnp.ndarray,
    obs_model: jnp.ndarray,
    process_noise: jnp.ndarray,
    obs_noise: jnp.ndarray,
    observations: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
) -> FilterState:
    """Kalman filter forward pass via ``jax.lax.scan``.

    Implements the standard predict-update cycle for a linear-Gaussian
    state-space model::

        x_t = A @ x_{t-1} + q_t,   q_t ~ N(0, Q)
        y_t = H @ x_t + r_t,        r_t ~ N(0, R)

    Args:
        transition: State transition matrix A, shape ``(N, N)``.
        obs_model: Observation matrix H, shape ``(M, N)``.
        process_noise: Process noise covariance Q, shape ``(N, N)``.
        obs_noise: Observation noise covariance R, shape ``(M, M)``.
        observations: Observed data y, shape ``(T, M)``.
        init_mean: Initial state mean x0, shape ``(N,)``.
        init_cov: Initial state covariance P0, shape ``(N, N)``.

    Returns:
        A ``FilterState`` with filtered/predicted means, covariances,
        and total log-likelihood.
    """
    M = obs_model.shape[0]
    log_2pi = jnp.log(2.0 * jnp.pi)

    def step(carry, y_t):
        x_filt, P_filt, ll = carry

        # --- Predict ---
        x_pred = transition @ x_filt
        P_pred = transition @ P_filt @ transition.T + process_noise

        # --- Update ---
        v = y_t - obs_model @ x_pred  # innovation
        S = obs_model @ P_pred @ obs_model.T + obs_noise  # innovation cov
        S_op = lx.MatrixLinearOperator(S)

        # Kalman gain: K = P_pred @ H^T @ S^{-1}
        PHt = P_pred @ obs_model.T  # (N, M)
        K = jax.vmap(lambda row: solve(S_op, row))(PHt)  # (N, M)

        x_filt_new = x_pred + K @ v
        P_filt_new = P_pred - K @ S @ K.T

        # Log-likelihood increment
        Sinv_v = solve(S_op, v)
        _, ld = jnp.linalg.slogdet(S)
        ll_inc = -0.5 * (v @ Sinv_v + ld + M * log_2pi)

        carry_new = (x_filt_new, P_filt_new, ll + ll_inc)
        outputs = (x_filt_new, P_filt_new, x_pred, P_pred)
        return carry_new, outputs

    init_carry = (init_mean, init_cov, jnp.array(0.0))
    final_carry, (f_means, f_covs, p_means, p_covs) = jax.lax.scan(
        step, init_carry, observations
    )

    return FilterState(
        filtered_means=f_means,
        filtered_covs=f_covs,
        predicted_means=p_means,
        predicted_covs=p_covs,
        log_likelihood=final_carry[2],
    )


def rts_smoother(
    filter_state: FilterState,
    transition: jnp.ndarray,
    process_noise: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rauch-Tung-Striebel backward smoother.

    Args:
        filter_state: Output of ``kalman_filter``.
        transition: State transition matrix A, shape ``(N, N)``.
        process_noise: Process noise covariance Q, shape ``(N, N)``.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` with shapes
        ``(T, N)`` and ``(T, N, N)``.
    """

    def step(carry, inputs):
        x_smooth, P_smooth = carry
        x_filt, P_filt, x_pred, P_pred = inputs

        # Smoother gain: G = P_filt @ A^T @ P_pred^{-1}
        P_pred_op = lx.MatrixLinearOperator(P_pred)
        At = transition.T
        G = P_filt @ At  # (N, N)
        G = jax.vmap(lambda row: solve(P_pred_op, row))(G)  # (N, N)

        x_smooth_new = x_filt + G @ (x_smooth - x_pred)
        P_smooth_new = P_filt + G @ (P_smooth - P_pred) @ G.T

        return (x_smooth_new, P_smooth_new), (x_smooth_new, P_smooth_new)

    T = filter_state.filtered_means.shape[0]
    init_carry = (
        filter_state.filtered_means[T - 1],
        filter_state.filtered_covs[T - 1],
    )

    # Reverse the sequences for backward pass (exclude last time step)
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.filtered_covs[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
        filter_state.predicted_covs[1:][::-1],
    )

    _, (s_means_rev, s_covs_rev) = jax.lax.scan(step, init_carry, inputs)

    # Reverse back and prepend last filtered state
    s_means = jnp.concatenate(
        [s_means_rev[::-1], filter_state.filtered_means[T - 1 :]], axis=0
    )
    s_covs = jnp.concatenate(
        [s_covs_rev[::-1], filter_state.filtered_covs[T - 1 :]], axis=0
    )

    return s_means, s_covs


def kalman_gain(
    P: lx.AbstractLinearOperator,
    H: lx.AbstractLinearOperator,
    R: lx.AbstractLinearOperator,
) -> jnp.ndarray:
    """Compute Kalman gain ``K = P @ H^T @ (H @ P @ H^T + R)^{-1}``.

    Args:
        P: Prior covariance operator, shape ``(N, N)``.
        H: Observation model operator, shape ``(M, N)``.
        R: Observation noise operator, shape ``(M, M)``.

    Returns:
        Kalman gain matrix of shape ``(N, M)``.
    """
    P_mat = P.as_matrix()
    H_mat = H.as_matrix()
    R_mat = R.as_matrix()

    S = H_mat @ P_mat @ H_mat.T + R_mat  # (M, M)
    S_op = lx.MatrixLinearOperator(S)

    # K = P H^T S^{-1}  =>  solve row by row
    PHt = P_mat @ H_mat.T  # (N, M)
    return jax.vmap(lambda row: solve(S_op, row))(PHt)  # (N, M)
