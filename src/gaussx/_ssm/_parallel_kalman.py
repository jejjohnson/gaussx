"""Parallel Kalman filter and RTS smoother via associative scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._linalg._linalg import solve_matrix, solve_rows
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._logdet import cholesky_logdet
from gaussx._ssm._kalman import FilterState
from gaussx._ssm._utils import _materialise, _normalise_tv_inputs
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


def parallel_kalman_filter(
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    obs_model: Float[Array, "*T M N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    obs_noise: Float[Array, "*T M M"] | lx.AbstractLinearOperator,
    observations: Float[Array, "T M"],
    init_mean: Float[Array, " N"],
    init_cov: Float[Array, "N N"],
    *,
    mask: Bool[Array, " T"] | None = None,
    solver: AbstractSolverStrategy | None = None,
) -> FilterState:
    """Kalman filter with dense array operations via ``jax.lax.scan``.

    Same generalised contract as :func:`gaussx.kalman_filter`:

    - Time-invariant inputs (single ``(N, N)`` etc.) auto-broadcast along ``T``.
    - Time-varying inputs accepted as ``(T, …)`` arrays.
    - Operator-typed inputs accepted in the time-invariant signature only.
    - Optional ``mask`` skips the update step (predict only) per timestep.

    Args:
        transition: State transition matrix or operator.
        obs_model: Observation matrix or operator.
        process_noise: Process noise covariance or operator.
        obs_noise: Observation noise covariance or operator.
        observations: Observed data, shape ``(T, M)``.
        init_mean: Initial state mean, shape ``(N,)``.
        init_cov: Initial state covariance, shape ``(N, N)``.
        mask: Optional ``(T,)`` boolean mask. ``False`` runs predict-only
            and contributes 0 to the log-likelihood. Defaults to all-True.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, factors ``S`` once via Cholesky and reuses
            for the gain, log-det, and quadratic term.

    Returns:
        A ``FilterState`` with filtered/predicted means, covariances,
        and total log-likelihood.
    """
    M = observations.shape[-1]
    T = observations.shape[0]
    log_2pi = jnp.log(2.0 * jnp.pi)

    A_seq, H_seq, Q_seq, R_seq, mask_seq, _ = _normalise_tv_inputs(
        transition, obs_model, process_noise, obs_noise, T=T, mask=mask
    )

    A_op = transition if isinstance(transition, lx.AbstractLinearOperator) else None
    H_op = obs_model if isinstance(obs_model, lx.AbstractLinearOperator) else None

    def step(carry, inputs):
        x_filt, P_filt, ll = carry
        A_t, H_t, Q_t, R_t, y_t, mask_t = inputs

        x_pred = A_op.mv(x_filt) if A_op is not None else A_t @ x_filt
        P_pred = A_t @ P_filt @ A_t.T + Q_t

        v = y_t - (H_op.mv(x_pred) if H_op is not None else H_t @ x_pred)
        S = H_t @ P_pred @ H_t.T + R_t
        S_op = lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)

        # In the default (no custom solver) path, factor S exactly once
        # via Cholesky and reuse it for the gain, log-det, and the
        # quadratic term — avoiding three independent factorizations.
        if solver is None:
            L_S = cholesky(S_op).as_matrix()
            K = jax.scipy.linalg.cho_solve((L_S, True), H_t @ P_pred).T
            Sinv_v = jax.scipy.linalg.cho_solve((L_S, True), v)
            ld = cholesky_logdet(L_S)
        else:
            K = solve_matrix(S_op, H_t @ P_pred, solver=solver).T
            Sinv_v = dispatch_solve(S_op, v, solver)
            ld = dispatch_logdet(S_op, solver)

        x_updated = x_pred + K @ v
        P_updated = P_pred - K @ S @ K.T
        ll_inc = -0.5 * (v @ Sinv_v + ld + M * log_2pi)

        x_new = jnp.where(mask_t, x_updated, x_pred)
        P_new = jnp.where(mask_t, P_updated, P_pred)
        ll_new = ll + jnp.where(mask_t, ll_inc, jnp.array(0.0))

        return (x_new, P_new, ll_new), (x_new, P_new, x_pred, P_pred)

    init = (init_mean, init_cov, jnp.array(0.0))
    (_, _, ll), (f_means, f_covs, p_means, p_covs) = jax.lax.scan(
        step, init, (A_seq, H_seq, Q_seq, R_seq, observations, mask_seq)
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
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Dense RTS smoother for outputs produced by ``parallel_kalman_filter``.

    The previous associative-scan formulation was not algebraically
    equivalent to the standard RTS recurrence. This implementation keeps
    the public API but uses a validated dense backward scan. Accepts the
    same time-invariant / time-varying / operator forms for
    ``transition`` as :func:`parallel_kalman_filter`.

    Args:
        filter_state: Output of :func:`kalman_filter` or
            :func:`parallel_kalman_filter`.
        transition: State transition matrix or operator.
        process_noise: Process noise covariance or operator. (Not used
            by the RTS recurrence — kept for API symmetry.)
        solver: Optional solver strategy.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    del process_noise

    T = filter_state.filtered_means.shape[0]

    A_dense = _materialise(transition)
    A_op = transition if isinstance(transition, lx.AbstractLinearOperator) else None
    if A_dense.ndim == 2:
        A_seq = jnp.broadcast_to(A_dense, (T, *A_dense.shape))
    elif A_dense.ndim == 3:
        if A_op is not None:
            raise TypeError(
                "Operator-typed transition cannot have a leading time axis."
            )
        A_seq = A_dense
    else:
        raise ValueError(f"transition must have ndim 2 or 3, got {A_dense.ndim}.")

    def step(carry, inputs):
        x_smooth, P_smooth = carry
        x_filt, P_filt, x_pred, P_pred, A_next = inputs

        P_pred_op = lx.MatrixLinearOperator(P_pred, lx.positive_semidefinite_tag)
        G = solve_rows(P_pred_op, P_filt @ A_next.T, solver=solver)
        x_smooth_new = x_filt + G @ (x_smooth - x_pred)
        P_smooth_new = P_filt + G @ (P_smooth - P_pred) @ G.T
        return (x_smooth_new, P_smooth_new), (x_smooth_new, P_smooth_new)

    init_carry = (
        filter_state.filtered_means[T - 1],
        filter_state.filtered_covs[T - 1],
    )
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.filtered_covs[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
        filter_state.predicted_covs[1:][::-1],
        A_seq[1:][::-1],
    )
    _, (s_means_rev, s_covs_rev) = jax.lax.scan(step, init_carry, inputs)

    s_means = jnp.concatenate(
        [s_means_rev[::-1], filter_state.filtered_means[T - 1 :]], axis=0
    )
    s_covs = jnp.concatenate(
        [s_covs_rev[::-1], filter_state.filtered_covs[T - 1 :]], axis=0
    )
    return s_means, s_covs
