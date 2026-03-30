"""Parallel Kalman filter and RTS smoother via associative scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gaussx._recipes._kalman import FilterState


def parallel_kalman_filter(
    transition: jnp.ndarray,
    obs_model: jnp.ndarray,
    process_noise: jnp.ndarray,
    obs_noise: jnp.ndarray,
    observations: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
) -> FilterState:
    """Kalman filter with dense array operations via ``jax.lax.scan``.

    Same interface as ``kalman_filter`` but operates on raw JAX arrays
    instead of lineax operators, avoiding per-step operator construction
    overhead. Faster on GPU/TPU for long time series.

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
        x_pred = transition @ x_filt
        P_pred = transition @ P_filt @ transition.T + process_noise
        v = y_t - obs_model @ x_pred
        S = obs_model @ P_pred @ obs_model.T + obs_noise
        S_inv = jnp.linalg.inv(S)
        K = P_pred @ obs_model.T @ S_inv
        x_new = x_pred + K @ v
        P_new = P_pred - K @ S @ K.T
        _, ld = jnp.linalg.slogdet(S)
        ll_inc = -0.5 * (v @ S_inv @ v + ld + M * log_2pi)
        return (x_new, P_new, ll + ll_inc), (x_new, P_new, x_pred, P_pred)

    init = (init_mean, init_cov, jnp.array(0.0))
    (_, _, ll), (f_means, f_covs, p_means, p_covs) = jax.lax.scan(
        step, init, observations
    )
    return FilterState(
        filtered_means=f_means,
        filtered_covs=f_covs,
        predicted_means=p_means,
        predicted_covs=p_covs,
        log_likelihood=ll,
    )


def parallel_rts_smoother(
    filter_state: FilterState,
    transition: jnp.ndarray,
    process_noise: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dense RTS smoother for outputs produced by ``parallel_kalman_filter``.

    The previous associative-scan formulation was not algebraically
    equivalent to the standard RTS recurrence. This implementation keeps
    the public API but uses a validated dense backward scan.

    Args:
        filter_state: Output of ``kalman_filter`` or
            ``parallel_kalman_filter``.
        transition: State transition matrix A, shape ``(N, N)``.
        process_noise: Process noise covariance Q, shape ``(N, N)``.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    del process_noise

    def step(carry, inputs):
        x_smooth, P_smooth = carry
        x_filt, P_filt, x_pred, P_pred = inputs

        G = jnp.linalg.solve(P_pred.T, (P_filt @ transition.T).T).T
        x_smooth_new = x_filt + G @ (x_smooth - x_pred)
        P_smooth_new = P_filt + G @ (P_smooth - P_pred) @ G.T
        return (x_smooth_new, P_smooth_new), (x_smooth_new, P_smooth_new)

    T = filter_state.filtered_means.shape[0]
    init_carry = (
        filter_state.filtered_means[T - 1],
        filter_state.filtered_covs[T - 1],
    )
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.filtered_covs[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
        filter_state.predicted_covs[1:][::-1],
    )
    _, (s_means_rev, s_covs_rev) = jax.lax.scan(step, init_carry, inputs)

    s_means = jnp.concatenate(
        [s_means_rev[::-1], filter_state.filtered_means[T - 1 :]], axis=0
    )
    s_covs = jnp.concatenate(
        [s_covs_rev[::-1], filter_state.filtered_covs[T - 1 :]], axis=0
    )
    return s_means, s_covs
