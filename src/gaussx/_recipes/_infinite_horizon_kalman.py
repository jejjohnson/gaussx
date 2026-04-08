"""Infinite-horizon Kalman filter and smoother using steady-state gains."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jaxtyping import Array, Float

from gaussx._recipes._dare import DAREResult, dare


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
    transition: Float[Array, "N N"],
    obs_model: Float[Array, "M N"],
    process_noise: Float[Array, "N N"],
    obs_noise: Float[Array, "M M"],
    observations: Float[Array, "T M"],
    init_mean: Float[Array, " N"] | None = None,
    *,
    dare_result: DAREResult | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> InfiniteHorizonState:
    """Infinite-horizon Kalman filter with fixed steady-state gain.

    Uses the DARE solution for a constant Kalman gain K∞, avoiding
    per-step Riccati updates.  For dense matrices, the per-step cost is
    O(N² + MN + M²) instead of O(N³) for the standard Kalman filter::

        Predict:  x⁻ₜ = A xₜ₋₁
        Update:   vₜ  = yₜ − H x⁻ₜ
                  xₜ  = x⁻ₜ + K∞ vₜ

    Args:
        transition: State transition matrix A, shape ``(N, N)``.
        obs_model: Observation matrix H, shape ``(M, N)``.
        process_noise: Process noise covariance Q, shape ``(N, N)``.
        obs_noise: Observation noise covariance R, shape ``(M, M)``.
        observations: Observed data y, shape ``(T, M)``.
        init_mean: Initial state mean, shape ``(N,)``. Defaults to zeros.
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

    P_inf = dare_result.P_inf  # (N, N)
    K_inf = dare_result.K_inf  # (N, M)
    T = observations.shape[0]
    M = obs_model.shape[0]
    N = transition.shape[0]

    # Precompute steady-state quantities
    P_pred_inf = transition @ P_inf @ transition.T + process_noise  # (N, N)
    S_inf = obs_model @ P_pred_inf @ obs_model.T + obs_noise  # (M, M)
    L_S = jnp.linalg.cholesky(S_inf)  # (M, M)
    from gaussx._primitives._logdet import cholesky_logdet

    ld_inf = cholesky_logdet(L_S)  # scalar
    log_2pi = jnp.log(2.0 * jnp.pi)

    # Steady-state filtered covariance: P_filt = (I − K∞ H) P⁻pred
    I_N = jnp.eye(N)
    P_filt_inf = (I_N - K_inf @ obs_model) @ P_pred_inf  # (N, N)

    def step(carry, y_t):
        x_filt, ll = carry

        x_pred = transition @ x_filt  # (N,)
        v = y_t - obs_model @ x_pred  # (M,)  innovation
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
    transition: Float[Array, "N N"],
    dare_result: DAREResult,
    process_noise: Float[Array, "N N"],
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Infinite-horizon RTS smoother with fixed steady-state gain.

    Precomputes the steady-state smoother gain G∞ = P∞ Aᵀ P⁻pred⁻¹,
    then runs a backward scan with fixed G∞.  The steady-state smoothed
    covariance is the solution of the discrete Lyapunov equation::

        P_smooth = P∞ + G∞ (P_smooth − P⁻pred) G∞ᵀ

    Args:
        filter_state: Output of ``infinite_horizon_filter``.
        transition: State transition matrix A, shape ``(N, N)``.
        dare_result: DARE result used in the filter.
        process_noise: Process noise covariance Q, shape ``(N, N)``.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` with shapes
        ``(T, N)`` and ``(T, N, N)``.
    """
    P_inf = dare_result.P_inf  # (N, N)
    P_pred_inf = transition @ P_inf @ transition.T + process_noise  # (N, N)

    # Steady-state smoother gain: G∞ = P∞ Aᵀ P⁻pred⁻¹
    N = transition.shape[0]
    G_inf = jnp.linalg.solve(P_pred_inf.T, (P_inf @ transition.T).T).T  # (N, N)

    # Solve discrete Lyapunov equation:
    # P_smooth = P∞ + G∞ (P_smooth − P⁻pred) G∞ᵀ
    # ⟺ (I − G∞ ⊗ G∞) vec(P_smooth) = vec(P∞ − G∞ P⁻pred G∞ᵀ)
    rhs = P_inf - G_inf @ P_pred_inf @ G_inf.T  # (N, N)
    identity = jnp.eye(N * N, dtype=P_inf.dtype)  # (N², N²)
    kron_term = jnp.kron(G_inf, G_inf)  # (N², N²)
    P_smooth_inf = rearrange(
        jnp.linalg.solve(identity - kron_term, rearrange(rhs, "n1 n2 -> (n1 n2)")),
        "(n1 n2) -> n1 n2",
        n1=N,
        n2=N,
    )  # (N, N)
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
