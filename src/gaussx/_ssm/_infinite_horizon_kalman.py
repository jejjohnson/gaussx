"""Infinite-horizon Kalman filter and smoother using steady-state gains."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import repeat
from gaussx._linalg._linalg import sandwich, solve_rows
from gaussx._linalg._lyapunov import discrete_lyapunov_solve
from gaussx._linalg._symmetrize import symmetrize
from gaussx._ssm._dare import DAREResult, dare
from gaussx._ssm._kalman import FilterState
from gaussx._ssm._utils import (
    _as_operator,
    _innovation_covariance,
    _left_matmul,
    _materialise,
    _matvec,
    _right_matmul_transpose,
)
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


# ``InfiniteHorizonState`` was a field-for-field copy of `FilterState`; it is
# now a deprecated alias, reached only through the package ``__getattr__``
# so that importing it warns (gh-364).
_DEPRECATED_ALIASES = {"InfiniteHorizonState": FilterState}


def _deprecated_alias(name: str) -> type[FilterState]:
    warnings.warn(
        f"gaussx.{name} is deprecated: infinite_horizon_filter returns a "
        "gaussx.FilterState. The alias will be removed in 0.5.0.",
        DeprecationWarning,
        stacklevel=3,
    )
    return _DEPRECATED_ALIASES[name]


def _checked_p_inf(dare_result: DAREResult) -> Float[Array, "N N"]:
    """``P_inf``, raising at run time if the DARE did not converge (gh-294)."""
    return eqx.error_if(
        dare_result.P_inf,
        ~dare_result.converged,
        "dare did not converge within max_iter doubling steps; the steady-state "
        "gain would be wrong. Raise max_iter, or check that (A, H) is "
        "detectable and R is invertible.",
    )


def infinite_horizon_filter(
    transition: Float[Array, "N N"] | lx.AbstractLinearOperator,
    obs_model: Float[Array, "M N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "N N"] | lx.AbstractLinearOperator,
    obs_noise: Float[Array, "M M"] | lx.AbstractLinearOperator,
    observations: Float[Array, "T M"],
    init_mean: Float[Array, " N"] | None = None,
    *,
    dare_result: DAREResult | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    solver: AbstractSolverStrategy | None = None,
    woodbury_innovation: bool = False,
) -> FilterState:
    """Infinite-horizon Kalman filter with fixed steady-state gain.

    Uses the DARE solution for a constant Kalman gain K∞, avoiding
    per-step Riccati updates.  For dense matrices, the per-step cost is
    O(N² + MN + M²) instead of O(N³) for the standard Kalman filter:

        Predict:  x⁻ₜ = A xₜ₋₁
        Update:   vₜ  = yₜ − H x⁻ₜ
                  xₜ  = x⁻ₜ + K∞ vₜ

    Like the other filters it predicts first: ``init_mean`` is the mean of
    x₀ and ``observations[0]`` is scored against ``A x₀``.

    Every step uses the steady-state innovation covariance S∞ and gain
    K∞, so the log-likelihood equals `kalman_filter`'s only when that
    filter starts at the DARE fixed point
    (``init_cov = dare(...).P_inf``); from any other prior it
    approximates the transient. There is no ``init_cov`` argument for
    that reason, and no ``mask``: a skipped update leaves the covariance
    off the steady state, which a fixed gain cannot follow. Use
    `gaussx.kalman_filter` for gappy data.

    All four operator/array arguments accept either a raw JAX array or
    a `lineax.AbstractLinearOperator`. Operator inputs preserve
    their structural matvec inside the per-step scan; the sandwiches
    materialise once outside the scan.

    Args:
        transition: State transition matrix or operator, shape ``(N, N)``.
        obs_model: Observation matrix or operator, shape ``(M, N)``.
        process_noise: Process noise covariance or operator, shape ``(N, N)``.
        obs_noise: Observation noise covariance or operator, shape ``(M, M)``.
        observations: Observed data y, shape ``(T, M)``.
        init_mean: Initial state mean, shape ``(N,)``. Defaults to zeros.
        dare_result: Precomputed DARE result. If ``None``, calls
            ``dare()`` internally. A result with ``converged=False``
            raises an ``EquinoxRuntimeError`` (also under ``jit``).
        max_iter: Maximum DARE doubling steps (used only if ``dare_result``
            is ``None``).
        tol: DARE convergence tolerance (used only if ``dare_result``
            is ``None``).
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.
        woodbury_innovation: When ``True``, build the steady-state
            innovation covariance as `gaussx.LowRankUpdate` so
            structured ``R`` can use Woodbury solves/log-determinants.

    Returns:
        A `gaussx.FilterState` with filtered/predicted means,
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
            solver=solver,
            woodbury_innovation=woodbury_innovation,
        )

    A_op = _as_operator(transition)
    H_op = _as_operator(obs_model)
    Q_dense = _materialise(process_noise)
    # Keep ``R`` lazy when the Woodbury innovation path will consume the
    # operator directly — avoids an O(M²) allocation for large structured
    # noise (e.g. ``DiagonalLinearOperator`` with large ``M``).
    R_for_innovation = (
        obs_noise
        if woodbury_innovation and isinstance(obs_noise, lx.AbstractLinearOperator)
        else _materialise(obs_noise)
    )

    P_inf = _checked_p_inf(dare_result)  # (N, N)
    K_inf = dare_result.K_inf  # (N, M)
    T = observations.shape[0]
    M = observations.shape[-1]
    N = A_op.out_size()

    # Precompute steady-state quantities
    P_inf_op = lx.MatrixLinearOperator(P_inf, lx.positive_semidefinite_tag)
    P_pred_inf = sandwich(A_op, P_inf_op).as_matrix() + Q_dense  # (N, N)
    S_inf = _innovation_covariance(
        H_op, P_pred_inf, R_for_innovation, woodbury=woodbury_innovation
    )
    ld_inf = dispatch_logdet(S_inf, solver)  # scalar

    # Steady-state filtered covariance: P_filt = (I − K∞ H) P⁻pred
    HP_pred_inf = _left_matmul(H_op, P_pred_inf)
    P_filt_inf = P_pred_inf - K_inf @ HP_pred_inf  # (N, N)

    def step(carry, y_t):
        x_filt, ll = carry

        x_pred = _matvec(transition, x_filt)  # (N,)
        v = y_t - _matvec(obs_model, x_pred)  # (M,)  innovation
        x_filt_new = x_pred + K_inf @ v  # (N,)

        # Log-likelihood increment.
        Sinv_v = dispatch_solve(S_inf, v, solver)  # (M,)
        ll_inc = -0.5 * (v @ Sinv_v + ld_inf + M * _LOG_2PI)

        return (x_filt_new, ll + ll_inc), (x_filt_new, x_pred)

    # The carry takes the model's dtype, not JAX's default float: a
    # default-float mean would meet the float32 S_inf of a float32 model in
    # the solve above, which lineax rejects under x64.
    dtype = jnp.result_type(observations, P_inf, K_inf)
    if init_mean is None:
        init_mean = jnp.zeros(N, dtype=dtype)
    init_carry = (jnp.asarray(init_mean, dtype=dtype), jnp.zeros((), dtype=dtype))
    (_, total_ll), (f_means, p_means) = jax.lax.scan(
        step,
        init_carry,
        observations,
    )

    # Broadcast constant covariances to (T, N, N)
    f_covs = repeat(P_filt_inf, "n1 n2 -> T n1 n2", T=T)
    p_covs = repeat(P_pred_inf, "n1 n2 -> T n1 n2", T=T)

    return FilterState(
        filtered_means=f_means,
        filtered_covs=f_covs,
        predicted_means=p_means,
        predicted_covs=p_covs,
        log_likelihood=total_ll,
    )


def infinite_horizon_smoother(
    filter_state: FilterState,
    transition: Float[Array, "N N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "N N"] | lx.AbstractLinearOperator | DAREResult,
    _legacy_process_noise: Float[Array, "N N"]
    | lx.AbstractLinearOperator
    | None = None,
    *,
    dare_result: DAREResult | None = None,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Infinite-horizon RTS smoother with fixed steady-state gain.

    Precomputes the steady-state smoother gain G∞ = P∞ Aᵀ P⁻pred⁻¹,
    then runs a backward scan with fixed G∞.  The steady-state smoothed
    covariance is the solution of the discrete Lyapunov equation:

        P_smooth = P∞ + G∞ (P_smooth − P⁻pred) G∞ᵀ

    Args:
        filter_state: Output of ``infinite_horizon_filter``.
        transition: State transition matrix or operator, shape ``(N, N)``.
        process_noise: Process noise covariance or operator, shape
            ``(N, N)``. The positional prefix ``(filter_state, transition,
            process_noise)`` matches `gaussx.rts_smoother`.
        dare_result: DARE result used in the filter (keyword-only). A
            result with ``converged=False`` raises an
            ``EquinoxRuntimeError``. The old positional order
            ``(filter_state, transition, dare_result, process_noise)`` still
            works with a ``DeprecationWarning`` until 0.5.0.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` with shapes
        ``(T, N)`` and ``(T, N, N)``.
    """
    noise: Array | lx.AbstractLinearOperator | None
    if isinstance(process_noise, DAREResult):
        # Old order: (filter_state, transition, dare_result, process_noise).
        warnings.warn(
            "infinite_horizon_smoother(filter_state, transition, dare_result, "
            "process_noise) is deprecated; pass process_noise third and "
            "dare_result as a keyword, as in rts_smoother. The old order will "
            "stop working in 0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        dare_result, noise = process_noise, _legacy_process_noise
    elif _legacy_process_noise is not None:
        msg = "infinite_horizon_smoother takes dare_result as a keyword argument."
        raise TypeError(msg)
    else:
        noise = process_noise
    if dare_result is None or noise is None:
        msg = "infinite_horizon_smoother requires process_noise and dare_result."
        raise TypeError(msg)

    A_op = _as_operator(transition)
    Q_dense = _materialise(noise)
    P_inf = _checked_p_inf(dare_result)  # (N, N)
    P_inf_op = lx.MatrixLinearOperator(P_inf, lx.positive_semidefinite_tag)
    P_pred_inf = sandwich(A_op, P_inf_op).as_matrix() + Q_dense  # (N, N)

    # Steady-state smoother gain: G∞ = P∞ Aᵀ P⁻pred⁻¹
    P_pred_inf_op = lx.MatrixLinearOperator(P_pred_inf, lx.positive_semidefinite_tag)
    G_inf = solve_rows(
        P_pred_inf_op,
        _right_matmul_transpose(P_inf, A_op),
        solver=solver,
    )  # (N, N)

    # Solve discrete Lyapunov equation:
    # P_smooth = P∞ + G∞ (P_smooth − P⁻pred) G∞ᵀ
    # ⟺ P_smooth − G∞ P_smooth G∞ᵀ = P∞ − G∞ P⁻pred G∞ᵀ
    # Routed through `discrete_lyapunov_solve` which uses a
    # per-factor eigendecomposition of ``G∞`` instead of materializing
    # the ``(N², N²)`` Kronecker matrix ``I − G∞ ⊗ G∞``.
    rhs = P_inf - G_inf @ P_pred_inf @ G_inf.T  # (N, N)
    P_smooth_inf = discrete_lyapunov_solve(G_inf, rhs)
    P_smooth_inf = symmetrize(P_smooth_inf)

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
