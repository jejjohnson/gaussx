"""Parallel Kalman filter and RTS smoother via associative scan.

Implements the Särkkä–García-Fernández (IEEE TAC 2021) parallel
formulation of the linear-Gaussian Kalman filter and Rauch–Tung–Striebel
smoother. The forward (filtering) and backward (smoothing) recurrences
are recast as inclusive associative scans of per-step elements, which
:func:`jax.lax.associative_scan` evaluates with ``O(log T)`` depth on
parallel hardware (GPU / TPU). On sequential hardware (CPU) the total
work is strictly larger than :func:`gaussx.kalman_filter`'s ``O(T)``
``lax.scan``; the win is on accelerators with large ``T``.

The element math is the covariance-form combinators from §III.A /
§III.B of the paper. The covariance form is known to lose conditioning
on long sequences and in float32 — a square-root variant is tracked in
#165.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._ssm._kalman import FilterState
from gaussx._ssm._utils import _materialise, _normalise_tv_inputs
from gaussx._strategies._base import AbstractSolverStrategy


# ----------------------------------------------------------------
# Filter element builders
# ----------------------------------------------------------------


def _first_filter_element_active(F, H, Q, R, y, m0, P0):
    """t=0 element absorbing the initial prior (predict + update)."""
    N = F.shape[0]
    m_pred = F @ m0
    P_pred = F @ P0 @ F.T + Q
    S = H @ P_pred @ H.T + R
    # K = P_pred H^T S^{-1}
    K = jnp.linalg.solve(S, H @ P_pred).T
    A = jnp.zeros((N, N), dtype=F.dtype)
    b = m_pred + K @ (y - H @ m_pred)
    C = P_pred - K @ H @ P_pred
    eta = jnp.zeros(N, dtype=F.dtype)
    J = jnp.zeros((N, N), dtype=F.dtype)
    return A, b, C, eta, J


def _first_filter_element_masked(F, Q, m0, P0):
    """t=0 predict-only element (mask=False at index 0)."""
    N = F.shape[0]
    A = jnp.zeros((N, N), dtype=F.dtype)
    b = F @ m0
    C = F @ P0 @ F.T + Q
    eta = jnp.zeros(N, dtype=F.dtype)
    J = jnp.zeros((N, N), dtype=F.dtype)
    return A, b, C, eta, J


def _generic_filter_element_active(F, H, Q, R, y):
    """Generic t>=1 element: predict from x_{t-1} fixed, then update with y."""
    N = F.shape[0]
    S = H @ Q @ H.T + R
    # K = Q H^T S^{-1}
    K = jnp.linalg.solve(S, H @ Q).T
    A = (jnp.eye(N, dtype=F.dtype) - K @ H) @ F
    b = K @ y
    C = Q - K @ H @ Q
    HF = H @ F
    Sinv_y = jnp.linalg.solve(S, y)
    eta = HF.T @ Sinv_y
    Sinv_HF = jnp.linalg.solve(S, HF)
    J = HF.T @ Sinv_HF
    return A, b, C, eta, J


def _generic_filter_element_masked(F, Q):
    """Predict-only element for masked steps."""
    N = F.shape[0]
    A = F
    b = jnp.zeros(N, dtype=F.dtype)
    C = Q
    eta = jnp.zeros(N, dtype=F.dtype)
    J = jnp.zeros((N, N), dtype=F.dtype)
    return A, b, C, eta, J


# ----------------------------------------------------------------
# Associative combinators
# ----------------------------------------------------------------


def _filter_combine(elem1, elem2):
    """Combine two filtering elements (Särkkä 2021, §III.A).

    ``elem1`` is the earlier-time block, ``elem2`` the later-time block.
    """
    A1, b1, C1, eta1, J1 = elem1
    A2, b2, C2, eta2, J2 = elem2
    N = A1.shape[0]
    I = jnp.eye(N, dtype=A1.dtype)

    # temp1 = A2 @ (I + C1 J2)^{-1}
    I_C1J2 = I + C1 @ J2
    temp1 = jnp.linalg.solve(I_C1J2.T, A2.T).T

    A = temp1 @ A1
    b = temp1 @ (b1 + C1 @ eta2) + b2
    C = temp1 @ C1 @ A2.T + C2

    # temp2 = A1.T @ (I + J2 C1)^{-1}
    I_J2C1 = I + J2 @ C1
    temp2 = jnp.linalg.solve(I_J2C1.T, A1).T

    eta = temp2 @ (eta2 - J2 @ b1) + eta1
    J = temp2 @ J2 @ A1 + J1

    return A, b, C, eta, J


def _smoother_combine(elem1, elem2):
    """Combine two smoothing elements (Särkkä 2021, §III.B).

    ``elem1`` is the earlier-time block, ``elem2`` the later-time block;
    the recurrence is m_smooth_t = E_t m_smooth_{t+1} + g_t.
    """
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2
    E = E1 @ E2
    g = E1 @ g2 + g1
    L = E1 @ L2 @ E1.T + L1
    return E, g, L


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------


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
    """Parallel Kalman filter via :func:`jax.lax.associative_scan`.

    Numerically equivalent to :func:`gaussx.kalman_filter` but with
    ``O(log T)`` parallel depth on accelerators. Same generalised
    contract (TI / TV / operator-typed inputs, optional mask, scalar
    log-likelihood).

    Args:
        transition: State transition matrix or operator.
        obs_model: Observation matrix or operator.
        process_noise: Process noise covariance or operator.
        obs_noise: Observation noise covariance or operator.
        observations: Observed data, shape ``(T, M)``.
        init_mean: Initial state mean, shape ``(N,)``.
        init_cov: Initial state covariance, shape ``(N, N)``.
        mask: Optional ``(T,)`` boolean mask; ``False`` runs predict-only
            and contributes 0 to the log-likelihood. Defaults to all-True.
        solver: Accepted for API symmetry with :func:`kalman_filter` but
            not currently threaded through the per-element solves; the
            covariance-form combinator uses unstructured dense solves.
            See #165 for the square-root variant tracking the structured
            path.

    Returns:
        :class:`FilterState` with filtered / predicted means and covs
        and the total log-likelihood.
    """
    del solver  # not currently threaded through; see docstring + #165

    M = observations.shape[-1]
    T = observations.shape[0]
    log_2pi = jnp.log(2.0 * jnp.pi)

    A_seq, H_seq, Q_seq, R_seq, mask_seq, _ = _normalise_tv_inputs(
        transition, obs_model, process_noise, obs_noise, T=T, mask=mask
    )

    # ----- Build per-step elements -----
    def _build_generic(F, H, Q, R, y, m):
        return jax.lax.cond(
            m,
            lambda: _generic_filter_element_active(F, H, Q, R, y),
            lambda: _generic_filter_element_masked(F, Q),
        )

    elems = jax.vmap(_build_generic)(
        A_seq, H_seq, Q_seq, R_seq, observations, mask_seq
    )

    # Patch element 0 to absorb the initial prior.
    first = jax.lax.cond(
        mask_seq[0],
        lambda: _first_filter_element_active(
            A_seq[0], H_seq[0], Q_seq[0], R_seq[0], observations[0],
            init_mean, init_cov,
        ),
        lambda: _first_filter_element_masked(
            A_seq[0], Q_seq[0], init_mean, init_cov,
        ),
    )
    elems = tuple(arr.at[0].set(val) for arr, val in zip(elems, first))

    # ----- Associative scan -----
    _A_out, b_out, C_out, _eta_out, _J_out = jax.lax.associative_scan(
        _filter_combine, elems
    )
    filtered_means = b_out
    filtered_covs = C_out

    # ----- Reconstruct predicted means / covs from filtered + transition -----
    prev_means = jnp.concatenate([init_mean[None], filtered_means[:-1]], axis=0)
    prev_covs = jnp.concatenate([init_cov[None], filtered_covs[:-1]], axis=0)

    def _predict_step(F, m, P, Q):
        return F @ m, F @ P @ F.T + Q

    predicted_means, predicted_covs = jax.vmap(_predict_step)(
        A_seq, prev_means, prev_covs, Q_seq
    )

    # ----- Log-likelihood from innovations (mask-gated) -----
    def _ll_contrib(y, m_pred, P_pred, H, R, m):
        v = y - H @ m_pred
        S = H @ P_pred @ H.T + R
        L = jnp.linalg.cholesky(S)
        Sinv_v = jax.scipy.linalg.cho_solve((L, True), v)
        quad = v @ Sinv_v
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        contrib = -0.5 * (quad + logdet + M * log_2pi)
        return jnp.where(m, contrib, jnp.zeros_like(contrib))

    ll_contribs = jax.vmap(_ll_contrib)(
        observations, predicted_means, predicted_covs, H_seq, R_seq, mask_seq
    )
    log_likelihood = jnp.sum(ll_contribs)

    return FilterState(
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        log_likelihood=log_likelihood,
    )


def parallel_rts_smoother(
    filter_state: FilterState,
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Parallel RTS smoother via reverse :func:`jax.lax.associative_scan`.

    Pairs with :func:`parallel_kalman_filter`. Numerically equivalent to
    :func:`gaussx.rts_smoother` with ``O(log T)`` parallel depth.

    Args:
        filter_state: Output of :func:`parallel_kalman_filter` or
            :func:`gaussx.kalman_filter`.
        transition: State transition matrix or operator.
        process_noise: Unused — kept for API symmetry with the sequential
            smoother.
        solver: Accepted for API symmetry; not currently threaded
            through. See #165.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    del process_noise, solver

    f_means = filter_state.filtered_means
    f_covs = filter_state.filtered_covs
    p_means = filter_state.predicted_means
    p_covs = filter_state.predicted_covs
    T = f_means.shape[0]
    N = f_means.shape[-1]

    # Normalise transition to a (T, N, N) stack — operators TI-only.
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

    # ----- Build per-step smoother elements -----
    # Inner elements (t = 0..T-2): standard smoother gain w.r.t. the
    # predicted state at t+1 reached through A_seq[t+1].
    def _build_inner(f_mean, f_cov, p_mean_next, p_cov_next, A_next):
        # G = f_cov @ A_next.T @ inv(p_cov_next)   (p_cov is symmetric)
        rhs = f_cov @ A_next.T  # (N, N)
        G = jnp.linalg.solve(p_cov_next, rhs.T).T
        E = G
        g = f_mean - G @ p_mean_next
        L = f_cov - G @ p_cov_next @ G.T
        return E, g, L

    inner_E, inner_g, inner_L = jax.vmap(_build_inner)(
        f_means[:-1], f_covs[:-1], p_means[1:], p_covs[1:], A_seq[1:]
    )
    last_E = jnp.zeros((1, N, N), dtype=f_means.dtype)
    last_g = f_means[-1:]
    last_L = f_covs[-1:]

    E = jnp.concatenate([inner_E, last_E], axis=0)
    g = jnp.concatenate([inner_g, last_g], axis=0)
    L = jnp.concatenate([inner_L, last_L], axis=0)

    # ----- Reverse associative scan -----
    _E_out, smoothed_means, smoothed_covs = jax.lax.associative_scan(
        _smoother_combine, (E, g, L), reverse=True
    )
    return smoothed_means, smoothed_covs
