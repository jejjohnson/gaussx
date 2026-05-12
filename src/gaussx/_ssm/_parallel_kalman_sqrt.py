"""Square-root parallel Kalman filter and RTS smoother helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._ssm._kalman import FilterState
from gaussx._ssm._parallel_kalman import (
    _bmv,
    _first_filter_element_active,
    _first_filter_element_masked,
    _generic_filter_element_active,
    _sym,
)
from gaussx._ssm._utils import _materialise, _normalise_tv_inputs
from gaussx._strategies._base import AbstractSolverStrategy


def tria(M: Float[Array, "... K N"]) -> Float[Array, "... N N"]:
    """Return the lower-triangular QR factor ``R`` with ``R @ R.T = M.T @ M``."""
    _, r = jnp.linalg.qr(M, mode="reduced")
    diag = jnp.diagonal(r, axis1=-2, axis2=-1)
    sign = jnp.where(diag < 0, -1.0, 1.0).astype(r.dtype)
    r = sign[..., :, None] * r
    return jnp.tril(jnp.swapaxes(r, -1, -2))


def _factor_from_psd(X: Float[Array, "... N N"]) -> Float[Array, "... N N"]:
    """Build a numerically PSD square root for a symmetric covariance."""
    X = _sym(X)
    w, v = jnp.linalg.eigh(X)
    w = jnp.clip(w, min=0.0)
    factor = v * jnp.sqrt(w)[..., None, :]
    return tria(jnp.swapaxes(factor, -1, -2))


def _cov_from_factor(U: Float[Array, "... N N"]) -> Float[Array, "... N N"]:
    return _sym(U @ jnp.swapaxes(U, -1, -2))


def _factor_filter_element(elem):
    A, b, C, eta, J = elem
    return A, b, C, _factor_from_psd(C), eta, J, _factor_from_psd(J)


def _filter_sqrt_combine(elem1, elem2):
    """Combine filtering elements while carrying factors for ``C`` and ``J``."""
    A1, b1, C1, _U1, eta1, J1, _Z1 = elem1
    A2, b2, C2, _U2, eta2, J2, _Z2 = elem2

    N = A1.shape[-1]
    eye = jnp.eye(N, dtype=A1.dtype)
    A2_T = jnp.swapaxes(A2, -1, -2)

    I_C1J2 = eye + C1 @ J2
    temp1 = jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(I_C1J2, -1, -2), A2_T), -1, -2)

    A = temp1 @ A1
    b = _bmv(temp1, b1 + _bmv(C1, eta2)) + b2
    C = _sym(temp1 @ C1 @ A2_T + C2)

    I_J2C1 = eye + J2 @ C1
    temp2 = jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(I_J2C1, -1, -2), A1), -1, -2)

    eta = _bmv(temp2, eta2 - _bmv(J2, b1)) + eta1
    J = _sym(temp2 @ J2 @ A1 + J1)

    return A, b, C, _factor_from_psd(C), eta, J, _factor_from_psd(J)


def parallel_kalman_filter_sqrt(
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
    """Square-root parallel Kalman filter via associative scan."""
    del solver

    M_obs = observations.shape[-1]
    T = observations.shape[0]
    N = init_mean.shape[0]

    if T == 0:
        return FilterState(
            filtered_means=jnp.zeros((0, N), dtype=init_mean.dtype),
            filtered_covs=jnp.zeros((0, N, N), dtype=init_cov.dtype),
            predicted_means=jnp.zeros((0, N), dtype=init_mean.dtype),
            predicted_covs=jnp.zeros((0, N, N), dtype=init_cov.dtype),
            log_likelihood=jnp.zeros((), dtype=init_mean.dtype),
        )

    log_2pi = jnp.log(2.0 * jnp.pi)

    A_seq, H_seq, Q_seq, R_seq, mask_seq, _ = _normalise_tv_inputs(
        transition, obs_model, process_noise, obs_noise, T=T, mask=mask
    )

    def _build_step(F, H, Q, R, y, m):
        H_eff = jnp.where(m, H, jnp.zeros_like(H))
        R_eff = jnp.where(m, R, jnp.eye(M_obs, dtype=R.dtype))
        y_eff = jnp.where(m, y, jnp.zeros_like(y))
        return _factor_filter_element(
            _generic_filter_element_active(F, H_eff, Q, R_eff, y_eff)
        )

    elems = jax.vmap(_build_step)(A_seq, H_seq, Q_seq, R_seq, observations, mask_seq)

    first = jax.lax.cond(
        mask_seq[0],
        lambda: _factor_filter_element(
            _first_filter_element_active(
                A_seq[0],
                H_seq[0],
                Q_seq[0],
                R_seq[0],
                observations[0],
                init_mean,
                init_cov,
            )
        ),
        lambda: _factor_filter_element(
            _first_filter_element_masked(
                A_seq[0],
                Q_seq[0],
                init_mean,
                init_cov,
            )
        ),
    )
    elems = tuple(arr.at[0].set(val) for arr, val in zip(elems, first, strict=True))

    _A_out, b_out, C_out, U_out, _eta_out, _J_out, _Z_out = jax.lax.associative_scan(
        _filter_sqrt_combine, elems
    )
    filtered_means = b_out
    filtered_covs_psd = _cov_from_factor(U_out)
    # Return PSD covariances while preserving gradients through the scan covariances.
    filtered_covs = C_out + jax.lax.stop_gradient(filtered_covs_psd - C_out)

    prev_means = jnp.concatenate([init_mean[None], filtered_means[:-1]], axis=0)
    init_factor = _factor_from_psd(init_cov)
    prev_factors = jnp.concatenate([init_factor[None], U_out[:-1]], axis=0)
    prev_covs = jnp.concatenate([init_cov[None], C_out[:-1]], axis=0)
    Q_factors = jax.vmap(_factor_from_psd)(Q_seq)

    def _predict_step(F, m, P, U, U_Q, Q):
        m_pred = F @ m
        # Keep the direct covariance path for gradients; use the factor path for values.
        P_pred = _sym(F @ P @ F.T + Q)
        U_pred = tria(
            jnp.concatenate(
                [jnp.swapaxes(U, -1, -2) @ F.T, jnp.swapaxes(U_Q, -1, -2)],
                axis=-2,
            )
        )
        P_pred_psd = _cov_from_factor(U_pred)
        return m_pred, P_pred + jax.lax.stop_gradient(P_pred_psd - P_pred)

    predicted_means, predicted_covs = jax.vmap(_predict_step)(
        A_seq, prev_means, prev_covs, prev_factors, Q_factors, Q_seq
    )

    def _ll_contrib(y, m_pred, P_pred, H, R, m):
        H_eff = jnp.where(m, H, jnp.zeros_like(H))
        R_eff = jnp.where(m, R, jnp.eye(M_obs, dtype=R.dtype))
        y_eff = jnp.where(m, y, jnp.zeros_like(y))
        v = y_eff - H_eff @ m_pred
        S = _sym(H_eff @ P_pred @ H_eff.T + R_eff)
        L = jnp.linalg.cholesky(S)
        Sinv_v = jax.scipy.linalg.cho_solve((L, True), v)
        quad = v @ Sinv_v
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        contrib = -0.5 * (quad + logdet + M_obs * log_2pi)
        return jnp.where(m, contrib, jnp.zeros_like(contrib))

    ll_contribs = jax.vmap(_ll_contrib)(
        observations, predicted_means, predicted_covs, H_seq, R_seq, mask_seq
    )

    return FilterState(
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        log_likelihood=jnp.sum(ll_contribs),
    )


def _smoother_sqrt_combine(elem1, elem2):
    E_later, g_later, U_later = elem1
    E_earlier, g_earlier, U_earlier = elem2
    E = E_earlier @ E_later
    g = _bmv(E_earlier, g_later) + g_earlier
    U = tria(
        jnp.concatenate(
            [
                jnp.swapaxes(U_later, -1, -2) @ jnp.swapaxes(E_earlier, -1, -2),
                jnp.swapaxes(U_earlier, -1, -2),
            ],
            axis=-2,
        )
    )
    return E, g, U


def parallel_rts_smoother_sqrt(
    filter_state: FilterState,
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Square-root parallel RTS smoother via reverse associative scan."""
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
        rhs = f_cov @ A_next.T
        G = jnp.linalg.solve(p_cov_next, rhs.T).T
        E = G
        g = f_mean - G @ p_mean_next
        L = _sym(f_cov - G @ p_cov_next @ G.T)
        return E, g, _factor_from_psd(L)

    inner_E, inner_g, inner_U = jax.vmap(_build_inner)(
        f_means[:-1], f_covs[:-1], p_means[1:], p_covs[1:], A_seq[1:]
    )
    last_E = jnp.zeros((1, N, N), dtype=f_means.dtype)
    last_g = f_means[-1:]
    last_U = _factor_from_psd(f_covs[-1:])

    E = jnp.concatenate([inner_E, last_E], axis=0)
    g = jnp.concatenate([inner_g, last_g], axis=0)
    U = jnp.concatenate([inner_U, last_U], axis=0)

    _E_out, smoothed_means, smoothed_factors = jax.lax.associative_scan(
        _smoother_sqrt_combine, (E, g, U), reverse=True
    )
    return smoothed_means, _cov_from_factor(smoothed_factors)
