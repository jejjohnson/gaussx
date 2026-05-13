"""Kalman filter, RTS smoother, and Kalman gain recipes."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._linalg._linalg import sandwich, solve_rows
from gaussx._ssm._utils import (
    _innovation_covariance,
    _left_matmul,
    _materialise,
    _normalise_tv_inputs,
    _right_matmul_transpose,
)
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


class FilterState(eqx.Module):
    """Output of ``kalman_filter``.

    Attributes:
        filtered_means: Shape ``(T, N)`` — filtered state estimates.
        filtered_covs: Shape ``(T, N, N)`` — filtered covariances.
        predicted_means: Shape ``(T, N)`` — predicted state estimates.
        predicted_covs: Shape ``(T, N, N)`` — predicted covariances.
        log_likelihood: Scalar — total log-likelihood.
    """

    filtered_means: Float[Array, "T N"]
    filtered_covs: Float[Array, "T N N"]
    predicted_means: Float[Array, "T N"]
    predicted_covs: Float[Array, "T N N"]
    log_likelihood: Float[Array, ""]


def kalman_filter(
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
    woodbury_innovation: bool = False,
) -> FilterState:
    r"""Kalman filter forward pass via ``jax.lax.scan``.

    Implements the predict-update cycle for a (possibly time-varying)
    linear-Gaussian state-space model::

        x_t = A_t @ x_{t-1} + q_t,   q_t ~ N(0, Q_t)
        y_t = H_t @ x_t + r_t,        r_t ~ N(0, R_t)

    **Time-invariant inputs** (single ``(N, N)`` / ``(M, N)`` etc.) are
    automatically broadcast along the time axis. **Time-varying inputs**
    are passed as ``(T, …)`` stacks (e.g. from
    :meth:`~gaussx.SDEKernel.discretise_sequence`).

    **Operator inputs** (lineax ``BlockDiag`` / ``Kronecker`` /
    ``DiagonalLinearOperator`` / ``MaskedOperator`` / etc.) are accepted
    in the **time-invariant** signature only. The structural matvec
    (``A @ x``, ``H @ x``) runs through the operator's ``mv``;
    operator-typed ``Q`` / ``R`` are materialised to dense arrays once
    outside the scan (the per-step sandwiches ``A P A^T`` / ``H P H^T``
    themselves run inside the scan because they depend on the evolving
    ``P_filt``).

    Args:
        transition: State transition matrix ``A``. Shape ``(N, N)``,
            ``(T, N, N)``, or :class:`lineax.AbstractLinearOperator`.
        obs_model: Observation matrix ``H``. Shape ``(M, N)``,
            ``(T, M, N)``, or operator.
        process_noise: Process noise covariance ``Q``. Shape ``(N, N)``,
            ``(T, N, N)``, or operator.
        obs_noise: Observation noise covariance ``R``. Shape ``(M, M)``,
            ``(T, M, M)``, or operator.
        observations: Observed data, shape ``(T, M)``.
        init_mean: Initial state mean, shape ``(N,)``.
        init_cov: Initial state covariance, shape ``(N, N)``.
        mask: Optional per-step boolean mask, shape ``(T,)``. ``True``
            (or ``1``) runs the full predict + update step; ``False``
            (or ``0``) runs the predict step only and skips the
            log-likelihood contribution. Defaults to all-True. Useful
            for prediction on merged train/test grids.
        solver: Optional solver strategy. When ``None``, uses
            structural dispatch.
        woodbury_innovation: When ``True``, build the innovation
            covariance ``S = H P Hᵀ + R`` as a
            :class:`gaussx.LowRankUpdate` so structured ``R`` can use
            Woodbury solves/log-determinants. Defaults to ``False`` to
            preserve the dense innovation path.

    Raises:
        TypeError: If operator-typed inputs are mixed with 3D ``(T, …)``
            arrays. Operator inputs must come from the time-invariant
            signature (per-step structured stacks are not supported;
            pass dense ``(T, …)`` arrays for the time-varying path).

    Returns:
        A ``FilterState`` with filtered/predicted means, covariances,
        and total log-likelihood.
    """
    M = observations.shape[-1]
    T = observations.shape[0]
    log_2pi = jnp.log(2.0 * jnp.pi)

    # Closure-friendly matvec: when an operator is supplied, prefer its
    # structural ``mv`` over the dense ``A @ x``. Otherwise the
    # broadcast 3D array contains ``A_seq[t]`` for each step.
    A_op = transition if isinstance(transition, lx.AbstractLinearOperator) else None
    H_op = obs_model if isinstance(obs_model, lx.AbstractLinearOperator) else None
    R_op = obs_noise if isinstance(obs_noise, lx.AbstractLinearOperator) else None
    A_seq, H_seq, Q_seq, R_seq, mask_seq, _ = _normalise_tv_inputs(
        transition,
        obs_model,
        process_noise,
        obs_noise,
        T=T,
        mask=mask,
        materialise_transition=A_op is None,
        materialise_obs=H_op is None,
        # Skip the O(T M²) dense broadcast of structured R when the
        # Woodbury path consumes the operator directly.
        materialise_obs_noise=not (woodbury_innovation and R_op is not None),
    )

    def step(carry, inputs):
        x_filt, P_filt, ll = carry
        A_t, H_t, Q_t, R_t, y_t, mask_t = inputs

        # --- Predict ---
        # Structural matvec when an operator was supplied; dense matmul otherwise.
        x_pred = A_op.mv(x_filt) if A_op is not None else A_t @ x_filt
        if A_op is not None:
            P_filt_op = lx.MatrixLinearOperator(P_filt, lx.positive_semidefinite_tag)
            P_pred = sandwich(A_op, P_filt_op).as_matrix() + Q_t
        else:
            P_pred = A_t @ P_filt @ A_t.T + Q_t

        # --- Update (gated by mask via lax.cond so the predict-only
        #             branch evaluates neither the update arithmetic
        #             nor produces gradients for the dropped path). ---
        def _do_update(_):
            v = y_t - (H_op.mv(x_pred) if H_op is not None else H_t @ x_pred)
            # Resolve ``R`` for innovation: operator path uses the
            # closed-over ``R_op`` (kept structural for Woodbury); array
            # path falls back to the per-step ``R_t``.
            R_innov = R_op if R_op is not None else R_t
            # Resolve ``H`` similarly so the operator preserves structure
            # in both the Woodbury and the structural-sandwich paths.
            H_innov = H_op if H_op is not None else H_t
            S_op = _innovation_covariance(
                H_innov, P_pred, R_innov, woodbury=woodbury_innovation
            )

            PHt = (
                _right_matmul_transpose(P_pred, H_op)
                if H_op is not None
                else P_pred @ H_t.T
            )  # (N, M)
            K = solve_rows(S_op, PHt, solver=solver)  # (N, M)

            x_upd = x_pred + K @ v
            if woodbury_innovation:
                # Avoid materialising S for the covariance update. Use
                # the operator path when available (H_t is a (0, 0)
                # placeholder under operator mode).
                HP_pred = (
                    _left_matmul(H_op, P_pred) if H_op is not None else H_t @ P_pred
                )
                P_upd = P_pred - K @ HP_pred
            else:
                P_upd = P_pred - K @ S_op.as_matrix() @ K.T

            Sinv_v = dispatch_solve(S_op, v, solver)
            ld = dispatch_logdet(S_op, solver)
            ll_inc = -0.5 * (v @ Sinv_v + ld + M * log_2pi)
            return x_upd, P_upd, ll_inc

        def _skip_update(_):
            return x_pred, P_pred, jnp.array(0.0)

        x_filt_new, P_filt_new, ll_inc = jax.lax.cond(
            mask_t, _do_update, _skip_update, operand=None
        )
        ll_new = ll + ll_inc

        carry_new = (x_filt_new, P_filt_new, ll_new)
        outputs = (x_filt_new, P_filt_new, x_pred, P_pred)
        return carry_new, outputs

    init_carry = (init_mean, init_cov, jnp.array(0.0))
    final_carry, (f_means, f_covs, p_means, p_covs) = jax.lax.scan(
        step, init_carry, (A_seq, H_seq, Q_seq, R_seq, observations, mask_seq)
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
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Rauch-Tung-Striebel backward smoother.

    Accepts the same time-invariant / time-varying / operator forms for
    ``transition`` and ``process_noise`` as :func:`kalman_filter`. When
    a step was masked off in the filter (``mask[t] == 0``), the
    smoother formula degenerates harmlessly because filtered ==
    predicted at that step.

    Args:
        filter_state: Output of :func:`kalman_filter`.
        transition: State transition matrix or operator.
        process_noise: Process noise covariance or operator. (Not
            currently used by the standard RTS recurrence — kept for
            API symmetry with :func:`kalman_filter`.)
        solver: Optional solver strategy.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    del process_noise  # not used in the standard RTS recurrence

    T = filter_state.filtered_means.shape[0]

    # Materialise once outside the scan for the sandwich; matvec stays
    # structural via the operator's mv.
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

        # Smoother gain: G = P_filt A_{t+1}^T P_pred_{t+1}^{-1}
        P_pred_op = lx.MatrixLinearOperator(P_pred, lx.positive_semidefinite_tag)
        G = P_filt @ A_next.T  # (N, N)
        G = solve_rows(P_pred_op, G, solver=solver)  # (N, N)

        x_smooth_new = x_filt + G @ (x_smooth - x_pred)
        P_smooth_new = P_filt + G @ (P_smooth - P_pred) @ G.T

        return (x_smooth_new, P_smooth_new), (x_smooth_new, P_smooth_new)

    init_carry = (
        filter_state.filtered_means[T - 1],
        filter_state.filtered_covs[T - 1],
    )

    # Reverse the sequences for backward pass (exclude last time step).
    # ``A_next[t]`` is the transition that maps step ``t`` to step ``t+1``,
    # i.e. ``A_seq[t+1]``.
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.filtered_covs[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
        filter_state.predicted_covs[1:][::-1],
        A_seq[1:][::-1],
    )

    _, (s_means_rev, s_covs_rev) = jax.lax.scan(step, init_carry, inputs)

    # Reverse back and prepend last filtered state.
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
    *,
    solver: AbstractSolverStrategy | None = None,
    woodbury_innovation: bool = False,
) -> Float[Array, "N M"]:
    """Compute Kalman gain ``K = P @ H^T @ (H @ P @ H^T + R)^{-1}``.

    Args:
        P: Prior covariance operator, shape ``(N, N)``.
        H: Observation model operator, shape ``(M, N)``.
        R: Observation noise operator, shape ``(M, M)``.
        solver: Optional solver strategy. When ``None``, uses
            structural dispatch.
        woodbury_innovation: When ``True``, route the innovation
            covariance through :class:`gaussx.LowRankUpdate`.

    Returns:
        Kalman gain matrix of shape ``(N, M)``.
    """
    P_mat = _materialise(P)
    H_mat = _materialise(H)

    S_op = _innovation_covariance(H, P, R, woodbury=woodbury_innovation)

    # K = P Hᵀ S⁻¹
    PHt = P_mat @ H_mat.T  # (N, M)
    return solve_rows(S_op, PHt, solver=solver)  # (N, M)
