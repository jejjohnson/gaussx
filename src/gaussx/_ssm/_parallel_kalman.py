"""Parallel Kalman filter and RTS smoother via associative scan.

Implements the Särkkä-García-Fernández (IEEE TAC 2021) parallel
formulation of the linear-Gaussian Kalman filter and Rauch-Tung-Striebel
smoother. The forward (filtering) and backward (smoothing) recurrences
are recast as inclusive associative scans of per-step elements, which
:func:`jax.lax.associative_scan` evaluates with ``O(log T)`` depth on
parallel hardware (GPU / TPU). On sequential hardware (CPU) the total
work is strictly larger than :func:`gaussx.kalman_filter`'s ``O(T)``
``lax.scan``; the win is on accelerators with large ``T``.

The default element math is the covariance-form combinators from §III.A /
§III.B of the paper. Pass ``form="sqrt"`` to additionally maintain lower-
triangular factors alongside the covariance updates and reconstruct PSD
covariances at the API boundary; the associative-scan equations still use
the covariance form internally, so the factor path is a PSD-safety net
for ill-conditioned float32 chains rather than a fully factor-propagating
combinator. See #165 for the latter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._primitives._logdet import cholesky_logdet
from gaussx._ssm._kalman import FilterState, kalman_filter
from gaussx._ssm._utils import _materialise, _normalise_tv_inputs
from gaussx._strategies._base import AbstractSolverStrategy


def _sym(X: Float[Array, "... N N"]) -> Float[Array, "... N N"]:
    """Symmetrise the trailing two axes (works on batched stacks too)."""
    return 0.5 * (X + jnp.swapaxes(X, -1, -2))


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


def _bmv(M, v):
    """Batched matrix-vector product over the trailing axes."""
    return jnp.einsum("...ij,...j->...i", M, v)


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
    mask: Bool[Array, " T"] | None = None,
    solver: AbstractSolverStrategy | None = None,
    woodbury_innovation: bool = False,
    form: str = "covariance",
) -> FilterState:
    """Parallel Kalman filter via :func:`jax.lax.associative_scan`.

    Numerically equivalent to :func:`gaussx.kalman_filter` but with
    ``O(log T)`` parallel depth on accelerators. Same generalised
    contract (TI / TV / operator-typed inputs, optional mask, scalar
    log-likelihood). Empty observation windows (``T == 0``) return a
    zero-length :class:`FilterState` with ``log_likelihood == 0``.

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
            The square-root form also uses dense solves for the affine
            terms.
        woodbury_innovation: When ``True``, delegates to
            :func:`gaussx.kalman_filter` with the same flag so structured
            ``R`` uses the Woodbury innovation path.
        form: Either ``"covariance"`` (default) or ``"sqrt"``. The
            square-root form maintains lower-triangular covariance
            factors alongside the covariance updates and reconstructs
            PSD covariance matrices in the returned :class:`FilterState`.
            Note: the associative-scan equations themselves still use
            the covariance form internally; the factor path is a
            PSD-safety net for ill-conditioned float32 chains rather
            than a fully factor-propagating combinator (see #165).

    Raises:
        ValueError: If ``form`` is not ``"covariance"`` or ``"sqrt"``.

    Returns:
        :class:`FilterState` with filtered / predicted means and covs
        and the total log-likelihood.
    """
    if form == "sqrt":
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
    if form != "covariance":
        raise ValueError("form must be 'covariance' or 'sqrt'.")

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

    del solver  # not currently threaded through; see docstring + #165

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
        transition, obs_model, process_noise, obs_noise, T=T, mask=mask
    )

    # Build per-step elements. ``vmap`` of ``lax.cond`` evaluates both
    # branches and selects, so we instead substitute mask-aware safe
    # inputs (H=0, R=I, y=0 for masked steps) into a single active path.
    # With those substitutions the active builder collapses to
    # (F, 0, Q, 0, 0) — exactly the predict-only element — and the
    # Cholesky operates on the well-conditioned identity, so even
    # garbage in masked H / R / y can't NaN the gradient.
    def _build_step(F, H, Q, R, y, m):
        H_eff = jnp.where(m, H, jnp.zeros_like(H))
        R_eff = jnp.where(m, R, jnp.eye(M_obs, dtype=R.dtype))
        y_eff = jnp.where(m, y, jnp.zeros_like(y))
        return _generic_filter_element_active(F, H_eff, Q, R_eff, y_eff)

    elems = jax.vmap(_build_step)(A_seq, H_seq, Q_seq, R_seq, observations, mask_seq)

    # Patch element 0 to absorb the initial prior. Outer ``lax.cond``
    # genuinely skips the inactive branch (no ``vmap`` wrapping here).
    first = jax.lax.cond(
        mask_seq[0],
        lambda: _first_filter_element_active(
            A_seq[0],
            H_seq[0],
            Q_seq[0],
            R_seq[0],
            observations[0],
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
    def _ll_contrib(y, m_pred, P_pred, H, R, m):
        H_eff = jnp.where(m, H, jnp.zeros_like(H))
        R_eff = jnp.where(m, R, jnp.eye(M_obs, dtype=R.dtype))
        y_eff = jnp.where(m, y, jnp.zeros_like(y))
        v = y_eff - H_eff @ m_pred
        S = _sym(H_eff @ P_pred @ H_eff.T + R_eff)
        L = jnp.linalg.cholesky(S)
        Sinv_v = jax.scipy.linalg.cho_solve((L, True), v)
        quad = v @ Sinv_v
        logdet = cholesky_logdet(L)
        contrib = -0.5 * (quad + logdet + M_obs * _LOG_2PI)
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
    form: str = "covariance",
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
            through.
        form: Either ``"covariance"`` (default) or ``"sqrt"``. The
            square-root form maintains lower-triangular factors
            alongside the smoother associative scan and returns
            PSD-reconstructed covariances (see :func:`parallel_kalman_filter`
            for the same caveat about the internal combinator).

    Raises:
        ValueError: If ``form`` is not ``"covariance"`` or ``"sqrt"``.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    if form == "sqrt":
        from gaussx._ssm._parallel_kalman_sqrt import parallel_rts_smoother_sqrt

        return parallel_rts_smoother_sqrt(
            filter_state,
            transition,
            process_noise,
            solver=solver,
        )
    if form != "covariance":
        raise ValueError("form must be 'covariance' or 'sqrt'.")

    del process_noise, solver

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
