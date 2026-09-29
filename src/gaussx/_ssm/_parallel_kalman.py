"""Parallel Kalman filter and RTS smoother via associative scan.

Implements the Särkkä-García-Fernández (IEEE TAC 2021) parallel
formulation of the linear-Gaussian Kalman filter and Rauch-Tung-Striebel
smoother. The forward (filtering) and backward (smoothing) recurrences
are recast as inclusive associative scans of per-step elements, which
`jax.lax.associative_scan` evaluates with ``O(log T)`` depth on
parallel hardware (GPU / TPU). On sequential hardware (CPU) the total
work is strictly larger than `gaussx.kalman_filter`'s ``O(T)``
``lax.scan``; the win is on accelerators with large ``T``.

The element math is the covariance-form combinators from §III.A / §III.B
of the paper. ``psd_project=True`` projects the returned covariances onto
the PSD cone -- a safety net for ill-conditioned float32 chains, not a
square-root filter: the scan itself still runs in covariance form. A
factor-propagating combinator is tracked in #454.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._linalg._symmetrize import symmetrize as _sym
from gaussx._primitives._logdet import cholesky_logdet
from gaussx._ssm._kalman import FilterState, kalman_filter
from gaussx._ssm._utils import (
    _masked_obs_inputs,
    _materialise,
    _normalise_tv_inputs,
    _warn_unused_process_noise,
)
from gaussx._strategies._base import AbstractSolverStrategy


# ----------------------------------------------------------------
# Filter element builders
# ----------------------------------------------------------------


def _first_filter_element_active(F, H, Q, R, y, m0, P0):
    """t=0 element absorbing the initial prior (predict + update)."""
    N = F.shape[0]
    m_pred = F @ m0
    P_pred = _sym(F @ P0 @ F.T + Q)
    HPpred = H @ P_pred  # (M, N)
    S = _sym(HPpred @ H.T + R)
    L_S = jnp.linalg.cholesky(S)
    K = jax.scipy.linalg.cho_solve((L_S, True), HPpred).T  # (N, M)
    A = jnp.zeros((N, N), dtype=F.dtype)
    b = m_pred + K @ (y - H @ m_pred)
    C = _sym(P_pred - K @ HPpred)
    eta = jnp.zeros(N, dtype=F.dtype)
    J = jnp.zeros((N, N), dtype=F.dtype)
    return A, b, C, eta, J


def _first_filter_element_masked(F, Q, m0, P0):
    """t=0 predict-only element (mask=False at index 0)."""
    N = F.shape[0]
    A = jnp.zeros((N, N), dtype=F.dtype)
    b = F @ m0
    C = _sym(F @ P0 @ F.T + Q)
    eta = jnp.zeros(N, dtype=F.dtype)
    J = jnp.zeros((N, N), dtype=F.dtype)
    return A, b, C, eta, J


def _generic_filter_element_active(F, H, Q, R, y):
    """Generic t>=1 element: predict from x_{t-1} fixed, then update with y.

    ``S = H Q H^T + R`` is factored once; the gain, innovation solve,
    and information solve all reuse the Cholesky factor.
    """
    N = F.shape[0]
    HQ = H @ Q  # (M, N)
    S = _sym(HQ @ H.T + R)
    L_S = jnp.linalg.cholesky(S)
    # K = Q H^T S^{-1} = (S^{-1} (H Q))^T
    K = jax.scipy.linalg.cho_solve((L_S, True), HQ).T  # (N, M)
    A = (jnp.eye(N, dtype=F.dtype) - K @ H) @ F
    b = K @ y
    C = _sym(Q - K @ HQ)
    HF = H @ F  # (M, N)
    Sinv_y = jax.scipy.linalg.cho_solve((L_S, True), y)  # (M,)
    Sinv_HF = jax.scipy.linalg.cho_solve((L_S, True), HF)  # (M, N)
    eta = HF.T @ Sinv_y
    J = _sym(HF.T @ Sinv_HF)
    return A, b, C, eta, J


# ----------------------------------------------------------------
# Associative combinators
# ----------------------------------------------------------------


def _bmv(
    M: Float[Array, "... N N"],
    v: Float[Array, "... N"],
) -> Float[Array, "... N"]:
    """Batched matrix-vector product over the trailing axes.

    Implemented as a broadcasting matmul rather than an einx contraction:
    ``associative_scan`` passes zero-length leading chunks, whose axis
    sizes einx cannot resolve.
    """
    return (M @ v[..., None])[..., 0]


def _filter_combine(elem1, elem2):
    """Combine two filtering elements (Särkkä 2021, §III.A).

    ``elem1`` is the earlier-time block, ``elem2`` the later-time block.
    Operates on the trailing two axes; ``lax.associative_scan`` passes
    batched chunks with a leading scan axis, so all matrix transposes
    use ``swapaxes(-1, -2)`` rather than ``.T`` and matrix-vector
    products use the explicit ``_bmv`` helper (Python ``@`` on
    ``(..., N, N)`` and ``(..., N)`` does not broadcast a matvec).
    """
    A1, b1, C1, eta1, J1 = elem1
    A2, b2, C2, eta2, J2 = elem2
    N = A1.shape[-1]
    eye = jnp.eye(N, dtype=A1.dtype)
    A2_T = jnp.swapaxes(A2, -1, -2)

    # temp1 = A2 @ (I + C1 J2)^{-1}
    #       = swapaxes(solve((I + C1 J2)^T, A2^T), -1, -2)
    I_C1J2 = eye + C1 @ J2
    temp1 = jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(I_C1J2, -1, -2), A2_T), -1, -2)

    A = temp1 @ A1
    b = _bmv(temp1, b1 + _bmv(C1, eta2)) + b2
    C = _sym(temp1 @ C1 @ A2_T + C2)

    # temp2 = A1^T @ (I + J2 C1)^{-1}
    I_J2C1 = eye + J2 @ C1
    temp2 = jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(I_J2C1, -1, -2), A1), -1, -2)

    eta = _bmv(temp2, eta2 - _bmv(J2, b1)) + eta1
    J = _sym(temp2 @ J2 @ A1 + J1)

    return A, b, C, eta, J


def _smoother_combine(elem1, elem2):
    """Combine two smoothing elements (Särkkä 2021, §III.B).

    Encodes ``m_smooth_t = E_t m_smooth_{t+1} + g_t``. Used inside
    ``lax.associative_scan(..., reverse=True)``, which reverses the
    sequence, runs a forward scan, then reverses the result — so under
    the forward-scan call ``combine(left, right)`` the left arg is the
    accumulated *later-time* chain and the right arg is the next
    *earlier-time* element. The combined result represents the
    earlier→later span starting at the right element's time.
    """
    E_later, g_later, L_later = elem1
    E_earlier, g_earlier, L_earlier = elem2
    E = E_earlier @ E_later
    g = _bmv(E_earlier, g_later) + g_earlier
    L = _sym(E_earlier @ L_later @ jnp.swapaxes(E_earlier, -1, -2) + L_earlier)
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
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    solver: AbstractSolverStrategy | None = None,
    woodbury_innovation: bool = False,
    form: str = "covariance",
    psd_project: bool = False,
) -> FilterState:
    """Parallel Kalman filter via `jax.lax.associative_scan`.

    Matches `gaussx.kalman_filter` to floating-point round-off for the
    default ``solver=None`` (``solver`` is not threaded through), with
    ``O(log T)`` parallel depth on accelerators. Same predict-first time
    convention and generalised
    contract (TI / TV / operator-typed inputs, optional mask, scalar
    log-likelihood). Empty observation windows (``T == 0``) return a
    zero-length `FilterState` with ``log_likelihood == 0``.

    Args:
        transition: State transition matrix or operator.
        obs_model: Observation matrix or operator.
        process_noise: Process noise covariance or operator.
        obs_noise: Observation noise covariance or operator.
        observations: Observed data, shape ``(T, M)``.
        init_mean: Mean of the prior on x₀, shape ``(N,)``.
        init_cov: Covariance of the prior on x₀, shape ``(N, N)``.
            The filter predicts before each update, so
            ``observations[0]`` is scored against ``A₀ x₀`` -- it
            observes x₁, not x₀. To observe the prior directly at step 0,
            pass a time-varying transition with ``A₀ = I`` and
            ``Q₀ = 0``.
        mask: Optional observation mask, dispatched on rank exactly as
            in `gaussx.kalman_filter`. Shape ``(T,)`` gates whole
            steps (``False`` runs predict-only and contributes 0 to the
            log-likelihood); shape ``(T, M)`` gates individual channels
            and yields the exact marginal log-likelihood over the
            observed entries. Defaults to all-True. A ``(T, M)`` mask is
            not supported with ``psd_project=True``.
        solver: Used only with ``woodbury_innovation=True``, which
            delegates to `gaussx.kalman_filter`. The associative-scan
            combinators use dense solves, so passing ``solver`` without
            ``woodbury_innovation`` is deprecated (it warns, and will raise
            in 0.5.0).
        woodbury_innovation: When ``True``, delegates to
            `gaussx.kalman_filter` with the same flag so structured
            ``R`` uses the Woodbury innovation path.
        form: ``"covariance"``. The former ``"sqrt"`` is a deprecated
            spelling of ``psd_project=True`` (it warns, and will be removed
            in 0.5.0): it never was a square-root filter.
        psd_project: Project each returned covariance onto the PSD cone
            (eigenvalue clip) and keep lower-triangular factors of the
            projections. The associative scan still runs in covariance
            form, so this has the covariance form's conditioning; it only
            guarantees PSD outputs, which float32 chains with very small
            observation noise can otherwise lose (the covariance form can
            return an indefinite covariance and a NaN log-likelihood
            there). Gradients are those of the unprojected path. gaussx has
            no square-root (PSD-by-construction) filter yet, sequential or
            parallel; see #454.

    Raises:
        ValueError: If ``form`` is not ``"covariance"`` or ``"sqrt"``.

    Returns:
        `FilterState` with filtered / predicted means and covs
        and the total log-likelihood.
    """
    if solver is not None and not woodbury_innovation:
        warnings.warn(
            "parallel_kalman_filter(solver=...) has no effect unless "
            "woodbury_innovation=True: the associative-scan combinators use "
            "dense solves. Passing it otherwise is deprecated and will raise "
            "in 0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    if form == "sqrt":
        warnings.warn(
            'form="sqrt" is deprecated: it is a PSD projection of the '
            "covariance-form combinator, not a square-root filter. Pass "
            'psd_project=True instead; form="sqrt" will be removed in 0.5.0.',
            DeprecationWarning,
            stacklevel=2,
        )
        psd_project = True
    elif form != "covariance":
        raise ValueError("form must be 'covariance' or 'sqrt'.")
    if psd_project:
        from gaussx._ssm._parallel_kalman_sqrt import parallel_kalman_filter_sqrt

        return parallel_kalman_filter_sqrt(
            transition,
            obs_model,
            process_noise,
            obs_noise,
            observations,
            init_mean,
            init_cov,
            mask=mask,
            solver=solver,
        )

    if woodbury_innovation:
        return kalman_filter(
            transition,
            obs_model,
            process_noise,
            obs_noise,
            observations,
            init_mean,
            init_cov,
            mask=mask,
            solver=solver,
            woodbury_innovation=True,
        )

    M_obs = observations.shape[-1]
    T = observations.shape[0]
    N = init_mean.shape[0]

    # Empty observation window: match kalman_filter's empty-scan output.
    if T == 0:
        return FilterState(
            filtered_means=jnp.zeros((0, N), dtype=init_mean.dtype),
            filtered_covs=jnp.zeros((0, N, N), dtype=init_cov.dtype),
            predicted_means=jnp.zeros((0, N), dtype=init_mean.dtype),
            predicted_covs=jnp.zeros((0, N, N), dtype=init_cov.dtype),
            log_likelihood=jnp.zeros((), dtype=init_mean.dtype),
        )

    A_seq, H_seq, Q_seq, R_seq, mask_seq, _ = _normalise_tv_inputs(
        transition, obs_model, process_noise, obs_noise, T=T, mask=mask, M=M_obs
    )
    # Work per-channel throughout: a ``(T,)`` gate is the special case
    # where every channel of a step shares one flag, so broadcasting it
    # reproduces the whole-step path exactly.
    mask_ch = (
        mask_seq
        if mask_seq.ndim == 2
        else jnp.broadcast_to(mask_seq[:, None], (T, M_obs))
    )
    step_active = jnp.any(mask_ch, axis=-1)

    # Build per-step elements. ``vmap`` of ``lax.cond`` evaluates both
    # branches and selects, so we instead substitute mask-aware safe
    # inputs (zeroed H rows, unit R block, zeroed y) into a single
    # active path. For a fully-masked step those substitutions collapse
    # the active builder to (F, 0, Q, 0, 0) — exactly the predict-only
    # element — and the Cholesky operates on the well-conditioned
    # identity, so even garbage in masked H / R / y can't NaN the
    # gradient.
    def _build_step(F, H, Q, R, y, m):
        H_eff, R_eff, y_eff, _ = _masked_obs_inputs(H, R, y, m)
        return _generic_filter_element_active(F, H_eff, Q, R_eff, y_eff)

    elems = jax.vmap(_build_step)(A_seq, H_seq, Q_seq, R_seq, observations, mask_ch)

    # Patch element 0 to absorb the initial prior. Outer ``lax.cond``
    # genuinely skips the inactive branch (no ``vmap`` wrapping here);
    # a partially-observed step 0 takes the active branch on the
    # substituted inputs.
    H_first, R_first, y_first, _ = _masked_obs_inputs(
        H_seq[0], R_seq[0], observations[0], mask_ch[0]
    )
    first = jax.lax.cond(
        step_active[0],
        lambda: _first_filter_element_active(
            A_seq[0],
            H_first,
            Q_seq[0],
            R_first,
            y_first,
            init_mean,
            init_cov,
        ),
        lambda: _first_filter_element_masked(
            A_seq[0],
            Q_seq[0],
            init_mean,
            init_cov,
        ),
    )
    elems = tuple(arr.at[0].set(val) for arr, val in zip(elems, first, strict=True))

    # ----- Associative scan -----
    _A_out, b_out, C_out, _eta_out, _J_out = jax.lax.associative_scan(
        _filter_combine, elems
    )
    filtered_means = b_out
    filtered_covs = jax.vmap(_sym)(C_out)

    # Reconstruct predicted means / covs from filtered + transition.
    prev_means = jnp.concatenate([init_mean[None], filtered_means[:-1]], axis=0)
    prev_covs = jnp.concatenate([init_cov[None], filtered_covs[:-1]], axis=0)

    def _predict_step(F, m, P, Q):
        return F @ m, _sym(F @ P @ F.T + Q)

    predicted_means, predicted_covs = jax.vmap(_predict_step)(
        A_seq, prev_means, prev_covs, Q_seq
    )

    # Log-likelihood from innovations. Same safe substitution as the
    # element builder so masked steps don't drive the Cholesky through
    # ill-conditioned user-supplied R / NaN gradients.
    def _ll_contrib(y, m_pred, P_pred, H, R, m, active):
        H_eff, R_eff, y_eff, n_missing = _masked_obs_inputs(H, R, y, m)
        v = y_eff - H_eff @ m_pred
        S = _sym(H_eff @ P_pred @ H_eff.T + R_eff)
        L = jnp.linalg.cholesky(S)
        Sinv_v = jax.scipy.linalg.cho_solve((L, True), v)
        quad = v @ Sinv_v
        logdet = cholesky_logdet(L)
        # Strip the dummy unit block's -0.5 * log(2 pi) per masked
        # channel, so this is the exact marginal over observed entries.
        contrib = -0.5 * (quad + logdet + M_obs * _LOG_2PI) + 0.5 * n_missing * _LOG_2PI
        return jnp.where(active, contrib, jnp.zeros_like(contrib))

    ll_contribs = jax.vmap(_ll_contrib)(
        observations,
        predicted_means,
        predicted_covs,
        H_seq,
        R_seq,
        mask_ch,
        step_active,
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
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator | None = None,
    *,
    solver: AbstractSolverStrategy | None = None,
    form: str = "covariance",
    psd_project: bool = False,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Parallel RTS smoother via reverse `jax.lax.associative_scan`.

    Pairs with `parallel_kalman_filter`. Numerically equivalent to
    `gaussx.rts_smoother` with ``O(log T)`` parallel depth.

    Args:
        filter_state: Output of `parallel_kalman_filter` or
            `gaussx.kalman_filter`.
        transition: State transition matrix or operator.
        process_noise: Deprecated and ignored, as in `gaussx.rts_smoother`;
            it will be removed in 0.5.0.
        solver: Accepted for API symmetry; not currently threaded
            through.
        form: ``"covariance"``. ``"sqrt"`` is a deprecated spelling of
            ``psd_project=True``, removed in 0.5.0.
        psd_project: Build the per-step smoother elements from
            PSD-projected factors and combine the factors with QR, so the
            smoothed covariances are PSD. As in `parallel_kalman_filter`,
            the elements come from covariance-form quantities, so this is
            a PSD safety net rather than a square-root smoother.

    Raises:
        ValueError: If ``form`` is not ``"covariance"`` or ``"sqrt"``.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    _warn_unused_process_noise("parallel_rts_smoother", process_noise)
    if form == "sqrt":
        warnings.warn(
            'form="sqrt" is deprecated: it is a PSD projection of the '
            "covariance-form combinator, not a square-root filter. Pass "
            'psd_project=True instead; form="sqrt" will be removed in 0.5.0.',
            DeprecationWarning,
            stacklevel=2,
        )
        psd_project = True
    elif form != "covariance":
        raise ValueError("form must be 'covariance' or 'sqrt'.")
    if psd_project:
        from gaussx._ssm._parallel_kalman_sqrt import parallel_rts_smoother_sqrt

        return parallel_rts_smoother_sqrt(filter_state, transition, solver=solver)

    del solver

    f_means = filter_state.filtered_means
    f_covs = filter_state.filtered_covs
    p_means = filter_state.predicted_means
    p_covs = filter_state.predicted_covs
    T = f_means.shape[0]
    N = f_means.shape[-1]

    if T == 0:
        return (
            jnp.zeros((0, N), dtype=f_means.dtype),
            jnp.zeros((0, N, N), dtype=f_covs.dtype),
        )

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

    def _build_inner(f_mean, f_cov, p_mean_next, p_cov_next, A_next):
        # G = f_cov @ A_next.T @ inv(p_cov_next); p_cov_next is symmetric.
        rhs = f_cov @ A_next.T  # (N, N)
        G = jnp.linalg.solve(p_cov_next, rhs.T).T
        E = G
        g = f_mean - G @ p_mean_next
        L = _sym(f_cov - G @ p_cov_next @ G.T)
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

    _E_out, smoothed_means, smoothed_covs = jax.lax.associative_scan(
        _smoother_combine, (E, g, L), reverse=True
    )
    smoothed_covs = jax.vmap(_sym)(smoothed_covs)
    return smoothed_means, smoothed_covs
