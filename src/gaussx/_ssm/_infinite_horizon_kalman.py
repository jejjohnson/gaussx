"""Infinite-horizon Kalman filter and RTS smoother with steady-state gains.

Both recipes freeze the covariance recursion at the fixed point Σ_x,∞ of the
filtering DARE (`gaussx.dare`) and scan only the means.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import einsum, repeat
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
    warn_deprecated(
        f"gaussx.{name} is deprecated: infinite_horizon_filter returns a "
        "gaussx.FilterState. The alias will be removed in gaussx 0.7.0."
    )
    return _DEPRECATED_ALIASES[name]


def _checked_steady_covariance(dare_result: DAREResult) -> Float[Array, "N N"]:
    """Σ_x,∞ from a DARE result, raising at run time if it did not converge.

    A non-converged doubling iterate would give a wrong steady-state gain
    without any other sign (gh-294), so the check runs under ``jit`` too.

    Args:
        dare_result: Output of `gaussx.dare`.

    Returns:
        The steady-state filtered covariance Σ_x,∞.
    """
    return eqx.error_if(
        dare_result.P_inf,
        ~dare_result.converged,
        "dare did not converge within max_iter doubling steps; the steady-state "
        "gain would be wrong. Raise max_iter, or check that (A, H) is "
        "detectable and R is invertible.",
    )


def _steady_predicted_covariance(
    transition_op: lx.AbstractLinearOperator,
    steady_covariance: Float[Array, "N N"],
    process_noise_dense: Float[Array, "N N"],
) -> Float[Array, "N N"]:
    """Steady-state predicted covariance Σ_pred,∞ = F Σ_x,∞ Fᵀ + Σ_η.

    Args:
        transition_op: Linear dynamics F as a lineax operator.
        steady_covariance: Steady-state filtered covariance Σ_x,∞.
        process_noise_dense: Model-error covariance Σ_η, dense.

    Returns:
        The steady-state predicted covariance Σ_pred,∞.
    """
    steady_covariance_op = lx.MatrixLinearOperator(
        steady_covariance, lx.positive_semidefinite_tag
    )
    # F Σ_x,∞ Fᵀ, structure-preserving
    propagated_op = sandwich(transition_op, steady_covariance_op)
    propagated = propagated_op.as_matrix()
    # Σ_pred,∞ = F Σ_x,∞ Fᵀ + Σ_η
    steady_predicted_covariance = propagated + process_noise_dense
    return steady_predicted_covariance


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
    """Infinite-horizon Kalman filter with the fixed steady-state gain G_∞.

        μ_x,t|t−1 = F μ_x,t−1|t−1
        μ_x,t|t = μ_x,t|t−1 + G_∞ (y_t − H μ_x,t|t−1)

    for the time-invariant model x_t = F x_t−1 + η_t, y_t = H x_t + ε_t
    with η_t ~ 𝒩(0, Σ_η) and ε_t ~ 𝒩(0, Σ_ε). The steady-state filtered
    covariance Σ_x,∞ solves the filtering DARE (`gaussx.dare`), and every
    step uses its predicted covariance Σ_pred,∞ = F Σ_x,∞ Fᵀ + Σ_η, its
    observation covariance Σ_y,∞ = H Σ_pred,∞ Hᵀ + Σ_ε and its gain
    G_∞ = Σ_pred,∞ Hᵀ Σ_y,∞⁻¹. No Riccati update runs per step, so a dense
    step costs O(D_x² + D_x D_y + D_y²) instead of O(D_x³).

    Like the other filters it predicts first: ``init_mean`` is the mean
    μ_x,0 of x₀ and ``observations[0]`` is scored against F x₀. Because
    every step uses Σ_y,∞ and G_∞, the log-likelihood equals
    `kalman_filter`'s only when that filter starts at the DARE fixed point
    (``init_cov = dare(...).P_inf``); from any other prior it approximates
    the transient. There is no ``init_cov`` argument for that reason, and
    no ``mask``: a skipped update leaves the covariance off the steady
    state, which a fixed gain cannot follow. Use `gaussx.kalman_filter` for
    gappy data.

    Every model argument accepts a dense array or a
    `lineax.AbstractLinearOperator`. Operator inputs keep their structural
    ``mv`` inside the scan; the sandwiches are materialised once outside it.

    Algorithm:
        1. Solve the DARE for Σ_x,∞ and G_∞, unless ``dare_result`` is given.
        2. Form Σ_pred,∞, Σ_y,∞ and log det Σ_y,∞ once, and the filtered
           covariance Σ_pred,∞ − G_∞ H Σ_pred,∞, which is Σ_x,∞ at the
           fixed point.
        3. Scan over t: predict μ_x,t|t−1 = F μ_x,t−1|t−1, form the residual
           r_t = y_t − H μ_x,t|t−1 and update μ_x,t|t = μ_x,t|t−1 + G_∞ r_t.
        4. Add log p(y_t ∣ y_1:t−1)
           = −½ (r_tᵀ Σ_y,∞⁻¹ r_t + log det Σ_y,∞ + D_y log 2π).

    Args:
        transition: Linear dynamics F.
        obs_model: Observation operator H.
        process_noise: Model-error covariance Σ_η.
        obs_noise: Observation-error covariance Σ_ε.
        observations: Observations y_1:T, one row per step.
        init_mean: Prior mean μ_x,0 of the initial state x₀. Defaults to
            zeros.
        dare_result: Precomputed `gaussx.dare` result holding Σ_x,∞ and
            G_∞. If ``None``, the DARE is solved here. A result with
            ``converged=False`` raises an ``EquinoxRuntimeError`` (also
            under ``jit``).
        max_iter: Maximum DARE doubling steps, used only when
            ``dare_result`` is ``None``.
        tol: DARE convergence tolerance, used only when ``dare_result`` is
            ``None``.
        solver: Solver strategy for Σ_y,∞; ``None`` uses structural
            dispatch.
        woodbury_innovation: When ``True``, build Σ_y,∞ as a
            `gaussx.LowRankUpdate` on Σ_ε so a structured Σ_ε gets Woodbury
            solves and log-determinants.

    Returns:
        A `gaussx.FilterState` with the filtered and predicted means, the
        constant covariances Σ_pred,∞ − G_∞ H Σ_pred,∞ and Σ_pred,∞ broadcast
        over time, and the total log-likelihood.

    Raises:
        EquinoxRuntimeError: If the DARE did not converge.

    Examples:
        A damped random walk observed through its first component:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> model = (0.9 * jnp.eye(2), jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> obs_noise = jnp.array([[0.5]])
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> state = gaussx.infinite_horizon_filter(*model, obs_noise, observations)
        >>> state.filtered_means.shape, state.filtered_covs.shape
        ((3, 2), (3, 2, 2))

        Started at the DARE fixed point, `gaussx.kalman_filter` agrees:

        >>> steady = gaussx.dare(*model, obs_noise)
        >>> exact = gaussx.kalman_filter(
        ...     *model, obs_noise, observations, jnp.zeros(2), steady.P_inf
        ... )
        >>> bool(jnp.allclose(state.log_likelihood, exact.log_likelihood))
        True
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

    transition_op = _as_operator(transition)
    obs_model_op = _as_operator(obs_model)
    process_noise_dense = _materialise(process_noise)
    # Keep Σ_ε lazy when the Woodbury path consumes the operator directly:
    # it avoids an O(D_y²) allocation for large structured noise (e.g. a
    # DiagonalLinearOperator with large D_y).
    if woodbury_innovation and isinstance(obs_noise, lx.AbstractLinearOperator):
        obs_noise_innov = obs_noise
    else:
        obs_noise_innov = _materialise(obs_noise)

    steady_covariance = _checked_steady_covariance(dare_result)
    steady_gain = dare_result.K_inf
    num_steps = observations.shape[0]
    obs_size = observations.shape[-1]
    state_size = transition_op.out_size()

    # --- steady-state moments, once ---
    # Σ_pred,∞ = F Σ_x,∞ Fᵀ + Σ_η
    steady_predicted_covariance = _steady_predicted_covariance(
        transition_op, steady_covariance, process_noise_dense
    )
    # Σ_y,∞ = H Σ_pred,∞ Hᵀ + Σ_ε
    steady_obs_covariance = _innovation_covariance(
        obs_model_op,
        steady_predicted_covariance,
        obs_noise_innov,
        woodbury=woodbury_innovation,
    )
    # log det Σ_y,∞
    steady_logdet = dispatch_logdet(steady_obs_covariance, solver)
    # H Σ_pred,∞
    obs_projection = _left_matmul(obs_model_op, steady_predicted_covariance)
    # G_∞ H Σ_pred,∞
    correction = einsum(steady_gain, obs_projection, "state obs, obs col -> state col")
    # Σ_pred,∞ − G_∞ H Σ_pred,∞, which is Σ_x,∞ at the fixed point
    steady_filtered_covariance = steady_predicted_covariance - correction

    def step(carry, obs_t):
        filtered_mean, log_likelihood = carry

        # μ_x,t|t−1 = F μ_x,t−1|t−1
        predicted_mean = _matvec(transition, filtered_mean)
        # H μ_x,t|t−1
        predicted_obs = _matvec(obs_model, predicted_mean)
        # r_t = y_t − H μ_x,t|t−1
        residual = obs_t - predicted_obs
        # μ_x,t|t = μ_x,t|t−1 + G_∞ r_t
        new_mean = predicted_mean + steady_gain @ residual

        # Σ_y,∞⁻¹ r_t
        whitened_residual = dispatch_solve(steady_obs_covariance, residual, solver)
        # r_tᵀ Σ_y,∞⁻¹ r_t
        mahalanobis = residual @ whitened_residual
        # log p(y_t ∣ y_1:t−1) = −½ (r_tᵀ Σ_y,∞⁻¹ r_t + log det Σ_y,∞
        #     + D_y log 2π)
        log_likelihood_step = -0.5 * (mahalanobis + steady_logdet + obs_size * _LOG_2PI)
        # log p(y_1:t) = log p(y_1:t−1) + log p(y_t ∣ y_1:t−1)
        new_log_likelihood = log_likelihood + log_likelihood_step

        new_carry = (new_mean, new_log_likelihood)
        return new_carry, (new_mean, predicted_mean)

    # The carry takes the model's dtype, not JAX's default float: a
    # default-float mean would meet the float32 Σ_y,∞ of a float32 model in
    # the solve above, which lineax rejects under x64.
    dtype = jnp.result_type(observations, steady_covariance, steady_gain)
    if init_mean is None:
        init_mean = jnp.zeros(state_size, dtype=dtype)
    init_mean = jnp.asarray(init_mean, dtype=dtype)
    zero_log_likelihood = jnp.zeros((), dtype=dtype)
    init_carry = (init_mean, zero_log_likelihood)
    final_carry, (filtered_means, predicted_means) = jax.lax.scan(
        step, init_carry, observations
    )

    filtered_covs = repeat(
        steady_filtered_covariance, "row col -> t row col", t=num_steps
    )
    predicted_covs = repeat(
        steady_predicted_covariance, "row col -> t row col", t=num_steps
    )

    return FilterState(
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        log_likelihood=final_carry[1],
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
    """Infinite-horizon RTS smoother with the fixed steady-state gain G_smooth,∞.

        μ_x,t|T = μ_x,t|t + G_smooth,∞ (μ_x,t+1|T − μ_x,t+1|t)
        Σ_smooth,∞ = Σ_x,∞ + G_smooth,∞ (Σ_smooth,∞ − Σ_pred,∞) G_smooth,∞ᵀ

    with the steady-state smoother gain G_smooth,∞ = Σ_x,∞ Fᵀ Σ_pred,∞⁻¹,
    where Σ_x,∞ is the DARE's filtered covariance and
    Σ_pred,∞ = F Σ_x,∞ Fᵀ + Σ_η its predicted one. The smoothed covariance
    Σ_smooth,∞ is the same at every step and solves the discrete Lyapunov
    equation above.

    Algorithm:
        1. Form Σ_pred,∞ and solve for G_smooth,∞ against it.
        2. Solve Σ_smooth,∞ − G_smooth,∞ Σ_smooth,∞ G_smooth,∞ᵀ
           = Σ_x,∞ − G_smooth,∞ Σ_pred,∞ G_smooth,∞ᵀ for Σ_smooth,∞.
        3. Scan backwards from μ_x,T|T, correcting each filtered mean with
           G_smooth,∞.

    Args:
        filter_state: Output of `infinite_horizon_filter`.
        transition: Linear dynamics F.
        process_noise: Model-error covariance Σ_η. The positional prefix
            ``(filter_state, transition, process_noise)`` matches
            `gaussx.rts_smoother`.
        dare_result: The `gaussx.dare` result used in the filter, holding
            Σ_x,∞ (keyword-only). A result with ``converged=False`` raises
            an ``EquinoxRuntimeError``. The old positional order
            ``(filter_state, transition, dare_result, process_noise)`` still
            works with a ``DeprecationWarning`` until gaussx 0.7.0.
        solver: Solver strategy for Σ_pred,∞; ``None`` uses structural
            dispatch.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` holding μ_x,t|T and
        Σ_smooth,∞ broadcast over time.

    Raises:
        TypeError: If ``process_noise`` or ``dare_result`` is missing, or
            ``dare_result`` is passed positionally after ``process_noise``.
        EquinoxRuntimeError: If the DARE did not converge.

    Examples:
        Smooth an infinite-horizon filter run:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transition, process_noise = 0.9 * jnp.eye(2), 0.1 * jnp.eye(2)
        >>> model = (transition, jnp.array([[1.0, 0.0]]), process_noise)
        >>> obs_noise = jnp.array([[0.5]])
        >>> steady = gaussx.dare(*model, obs_noise)
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> state = gaussx.infinite_horizon_filter(
        ...     *model, obs_noise, observations, dare_result=steady
        ... )
        >>> means, covs = gaussx.infinite_horizon_smoother(
        ...     state, transition, process_noise, dare_result=steady
        ... )
        >>> means.shape, covs.shape
        ((3, 2), (3, 2, 2))

        Smoothing never inflates the variance:

        >>> bool(jnp.all(jnp.diag(covs[0]) <= jnp.diag(state.filtered_covs[0])))
        True
    """
    noise: Array | lx.AbstractLinearOperator | None
    if isinstance(process_noise, DAREResult):
        # Old order: (filter_state, transition, dare_result, process_noise).
        warn_deprecated(
            "infinite_horizon_smoother(filter_state, transition, dare_result, "
            "process_noise) is deprecated; pass process_noise third and "
            "dare_result as a keyword, as in rts_smoother. The old order will "
            "stop working in gaussx 0.7.0."
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

    transition_op = _as_operator(transition)
    process_noise_dense = _materialise(noise)
    steady_covariance = _checked_steady_covariance(dare_result)
    # Σ_pred,∞ = F Σ_x,∞ Fᵀ + Σ_η
    steady_predicted_covariance = _steady_predicted_covariance(
        transition_op, steady_covariance, process_noise_dense
    )

    steady_predicted_covariance_op = lx.MatrixLinearOperator(
        steady_predicted_covariance, lx.positive_semidefinite_tag
    )
    # Σ_x,∞ Fᵀ
    cross_covariance = _right_matmul_transpose(steady_covariance, transition_op)
    # G_smooth,∞ = Σ_x,∞ Fᵀ Σ_pred,∞⁻¹, row by row since Σ_pred,∞ is symmetric
    smoother_gain = solve_rows(
        steady_predicted_covariance_op, cross_covariance, solver=solver
    )

    # --- steady-state smoothed covariance ---
    # discrete_lyapunov_solve uses an eigendecomposition of G_smooth,∞
    # instead of materialising the (D_x², D_x²) Kronecker matrix
    # I − G_smooth,∞ ⊗ G_smooth,∞.
    # G_smooth,∞ Σ_pred,∞
    gain_covariance = einsum(
        smoother_gain,
        steady_predicted_covariance,
        "state next, next col -> state col",
    )
    # G_smooth,∞ Σ_pred,∞ G_smooth,∞ᵀ
    correction = einsum(
        gain_covariance, smoother_gain, "state col, other col -> state other"
    )
    # Σ_x,∞ − G_smooth,∞ Σ_pred,∞ G_smooth,∞ᵀ
    lyapunov_rhs = steady_covariance - correction
    # Σ_smooth,∞ − G_smooth,∞ Σ_smooth,∞ G_smooth,∞ᵀ
    #     = Σ_x,∞ − G_smooth,∞ Σ_pred,∞ G_smooth,∞ᵀ
    steady_smoothed_covariance = discrete_lyapunov_solve(smoother_gain, lyapunov_rhs)
    steady_smoothed_covariance = symmetrize(steady_smoothed_covariance)

    num_steps = filter_state.filtered_means.shape[0]

    def step(smoothed_mean_next, inputs):
        filtered_mean, predicted_mean_next = inputs
        # μ_x,t+1|T − μ_x,t+1|t
        mean_correction = smoothed_mean_next - predicted_mean_next
        # μ_x,t|T = μ_x,t|t + G_smooth,∞ (μ_x,t+1|T − μ_x,t+1|t)
        smoothed_mean = filtered_mean + smoother_gain @ mean_correction
        return smoothed_mean, smoothed_mean

    init_carry = filter_state.filtered_means[num_steps - 1]
    # Reverse the sequences for the backward pass (excluding the last step).
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
    )

    _, smoothed_means_rev = jax.lax.scan(step, init_carry, inputs)

    # Reverse back and append the last filtered mean.
    smoothed_means = jnp.concatenate(
        [smoothed_means_rev[::-1], filter_state.filtered_means[num_steps - 1 :]]
    )
    smoothed_covs = repeat(
        steady_smoothed_covariance, "row col -> t row col", t=num_steps
    )

    return smoothed_means, smoothed_covs
