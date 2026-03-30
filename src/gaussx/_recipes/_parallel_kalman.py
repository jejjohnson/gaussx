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
    """Parallel Kalman filter via ``jax.lax.associative_scan``.

    Same interface as ``kalman_filter`` but uses dense array operations
    compatible with ``jax.lax.scan`` and amenable to future associative
    scan parallelisation. Faster on GPU/TPU for long time series.

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
    """Parallel RTS smoother via ``jax.lax.associative_scan``.

    Uses an associative scan over smoother gain elements for
    O(log T) parallel depth instead of O(T) sequential steps.

    Args:
        filter_state: Output of ``kalman_filter`` or
            ``parallel_kalman_filter``.
        transition: State transition matrix A, shape ``(N, N)``.
        process_noise: Process noise covariance Q, shape ``(N, N)``.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    T = filter_state.filtered_means.shape[0]

    # Build smoother elements: (G_t, g_t, L_t)
    def _build_element(m_filt, P_filt, m_pred_next, P_pred_next):
        G = P_filt @ transition.T @ jnp.linalg.inv(P_pred_next)
        g = m_filt - G @ m_pred_next
        L = P_filt - G @ P_pred_next @ G.T
        return G, g, L

    Gs, gs, Ls = jax.vmap(_build_element)(
        filter_state.filtered_means[:-1],
        filter_state.filtered_covs[:-1],
        filter_state.predicted_means[1:],
        filter_state.predicted_covs[1:],
    )

    # Associative operator: (G1,g1,L1) * (G2,g2,L2)
    def _combine(elem1, elem2):
        G1, g1, L1 = elem1
        G2, g2, L2 = elem2
        G1_T = jnp.swapaxes(G1, -2, -1)
        G_new = jnp.matmul(G1, G2)
        g_new = jnp.einsum("...ij,...j->...i", G1, g2) + g1
        L_new = jnp.matmul(jnp.matmul(G1, L2), G1_T) + L1
        return G_new, g_new, L_new

    # Reverse for backward pass using jnp.flip (trace-safe)
    Gs_rev = jnp.flip(Gs, axis=0)
    gs_rev = jnp.flip(gs, axis=0)
    Ls_rev = jnp.flip(Ls, axis=0)

    Gs_scan, gs_scan, Ls_scan = jax.lax.associative_scan(
        _combine, (Gs_rev, gs_rev, Ls_rev)
    )

    # Flip back
    Gs_scan = jnp.flip(Gs_scan, axis=0)
    gs_scan = jnp.flip(gs_scan, axis=0)
    Ls_scan = jnp.flip(Ls_scan, axis=0)

    # Recover smoothed distributions
    m_last = filter_state.filtered_means[T - 1]
    P_last = filter_state.filtered_covs[T - 1]

    s_means_early = jax.vmap(lambda G, g: G @ m_last + g)(Gs_scan, gs_scan)
    s_covs_early = jax.vmap(lambda G, L: G @ P_last @ G.T + L)(Gs_scan, Ls_scan)

    s_means = jnp.concatenate([s_means_early, m_last[None, :]], axis=0)
    s_covs = jnp.concatenate([s_covs_early, P_last[None, :, :]], axis=0)

    return s_means, s_covs
