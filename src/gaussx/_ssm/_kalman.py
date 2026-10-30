"""Covariance-form Kalman filter, RTS smoother and Kalman gain.

These are the sequential recipes for a linear-Gaussian state-space model;
the parallel and square-root variants live in ``_parallel_kalman.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import einsum, repeat
from gaussx._linalg._linalg import sandwich, solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._ssm._utils import (
    _innovation_covariance,
    _left_matmul,
    _masked_obs_inputs,
    _materialise,
    _normalise_tv_inputs,
    _right_matmul_transpose,
    _warn_unused_process_noise,
)
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


class FilterState(eqx.Module):
    """Output of ``kalman_filter``: the filtered and predicted moments of x_t.

    Row ``t`` holds the state observed by ``observations[t]``, which is
    x_t+1 in the one-based notation of `kalman_filter`. The filter predicts
    before each update, so ``predicted_means[0]`` is F₁ μ_x,0, one
    transition past the prior on x₀, not μ_x,0.

    Attributes:
        filtered_means: Filtered means μ_x,t|t.
        filtered_covs: Filtered covariances Σ_x,t|t.
        predicted_means: Predicted means μ_x,t|t−1.
        predicted_covs: Predicted covariances Σ_x,t|t−1.
        log_likelihood: Total log-likelihood log p(y_1:T).
    """

    filtered_means: Float[Array, "T N"]
    filtered_covs: Float[Array, "T N N"]
    predicted_means: Float[Array, "T N"]
    predicted_covs: Float[Array, "T N N"]
    log_likelihood: Float[Array, ""]


def _empty_filter_state(
    init_mean: Float[Array, " N"],
    init_cov: Float[Array, "N N"],
) -> FilterState:
    """Zero-length `FilterState` for T = 0, as `kalman_filter`'s empty scan.

    Args:
        init_mean: Prior mean μ_x,0, which sets D_x and the dtype.
        init_cov: Prior covariance Σ_x,0.

    Returns:
        A `FilterState` with no steps and ``log_likelihood == 0``.
    """
    state_size = init_mean.shape[0]
    return FilterState(
        filtered_means=jnp.zeros((0, state_size), dtype=init_mean.dtype),
        filtered_covs=jnp.zeros((0, state_size, state_size), dtype=init_cov.dtype),
        predicted_means=jnp.zeros((0, state_size), dtype=init_mean.dtype),
        predicted_covs=jnp.zeros((0, state_size, state_size), dtype=init_cov.dtype),
        log_likelihood=jnp.zeros((), dtype=init_mean.dtype),
    )


def _kalman_predict(
    filtered_mean: Float[Array, " N"],
    filtered_covariance: Float[Array, "N N"],
    transition_t: Float[Array, "N N"],
    process_noise_t: Float[Array, "N N"],
    *,
    transition_op: lx.AbstractLinearOperator | None,
) -> tuple[Float[Array, " N"], Float[Array, "N N"]]:
    """Kalman predict step: push the filtered moments through the dynamics.

        μ_x,t|t−1 = F_t μ_x,t−1|t−1
        Σ_x,t|t−1 = F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t

    Algorithm:
        1. Propagate the mean through F_t.
        2. Sandwich the covariance, F_t Σ_x,t−1|t−1 F_tᵀ, and add Σ_η,t.
        3. Symmetrise the result.

    Args:
        filtered_mean: Filtered mean μ_x,t−1|t−1.
        filtered_covariance: Filtered covariance Σ_x,t−1|t−1.
        transition_t: Dense dynamics F_t for this step; unused when
            ``transition_op`` is given.
        process_noise_t: Model-error covariance Σ_η,t.
        transition_op: Time-invariant dynamics F as a lineax operator, or
            ``None`` for the dense path.

    Returns:
        Tuple ``(predicted_mean, predicted_covariance)`` holding
        μ_x,t|t−1 and Σ_x,t|t−1.
    """
    if transition_op is not None:
        # μ_x,t|t−1 = F μ_x,t−1|t−1, through the operator's structural mv
        predicted_mean = transition_op.mv(filtered_mean)
        filtered_covariance_op = lx.MatrixLinearOperator(
            filtered_covariance, lx.positive_semidefinite_tag
        )
        # F Σ_x,t−1|t−1 Fᵀ, structure-preserving
        propagated_op = sandwich(transition_op, filtered_covariance_op)
        propagated = propagated_op.as_matrix()
    else:
        # μ_x,t|t−1 = F_t μ_x,t−1|t−1
        predicted_mean = transition_t @ filtered_mean
        # F_t Σ_x,t−1|t−1
        transition_covariance = einsum(
            transition_t,
            filtered_covariance,
            "next prev, prev col -> next col",
        )
        # F_t Σ_x,t−1|t−1 F_tᵀ
        propagated = einsum(
            transition_covariance,
            transition_t,
            "next col, other col -> next other",
        )
    # Σ_x,t|t−1 = F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t
    predicted_covariance = propagated + process_noise_t
    # Round-off leaves Σ slightly asymmetric, and numpyro's
    # positive_definite constraint checks symmetry exactly (gh-388).
    predicted_covariance = symmetrize(predicted_covariance)
    return predicted_mean, predicted_covariance


def _kalman_update(
    predicted_mean: Float[Array, " N"],
    predicted_covariance: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"] | lx.AbstractLinearOperator,
    obs_noise_covariance: Float[Array, "M M"] | lx.AbstractLinearOperator,
    residual: Float[Array, " M"],
    n_missing: Float[Array, ""] | float,
    *,
    obs_size: int,
    solver: AbstractSolverStrategy | None,
    woodbury_innovation: bool,
) -> tuple[Float[Array, " N"], Float[Array, "N N"], Float[Array, ""]]:
    """Kalman update step and its log-likelihood term.

        G_t = Σ_x,t|t−1 H_tᵀ Σ_y,t⁻¹,   Σ_y,t = H_t Σ_x,t|t−1 H_tᵀ + Σ_ε,t
        μ_x,t|t = μ_x,t|t−1 + G_t r_t
        Σ_x,t|t = Σ_x,t|t−1 − G_t Σ_y,t G_tᵀ

    Shared by the gated and the per-channel paths of `kalman_filter`.

    Algorithm:
        1. Build the observation covariance Σ_y,t.
        2. Solve for the gain G_t = Σ_xy,t Σ_y,t⁻¹ with
           Σ_xy,t = Σ_x,t|t−1 H_tᵀ.
        3. Update the mean and covariance, then symmetrise the covariance.
        4. Score the residual: log p(y_t ∣ y_1:t−1)
           = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t + D_y log 2π),
           plus ½ log 2π per masked channel.

    Args:
        predicted_mean: Predicted mean μ_x,t|t−1.
        predicted_covariance: Predicted covariance Σ_x,t|t−1.
        obs_operator: Observation operator H_t, as a lineax operator
            (structural path) or a dense array.
        obs_noise_covariance: Observation-error covariance Σ_ε,t, as an
            operator or a dense array.
        residual: Residual r_t = y_t − H_t μ_x,t|t−1.
        n_missing: Number of masked channels N_missing, whose dummy unit
            blocks are stripped from the log-likelihood.
        obs_size: Observation length D_y, masked channels included.
        solver: Solver strategy for Σ_y,t; ``None`` uses structural
            dispatch.
        woodbury_innovation: When ``True``, Σ_y,t is a
            `gaussx.LowRankUpdate` and is never materialised.

    Returns:
        Tuple ``(updated_mean, updated_covariance, log_likelihood_step)``
        holding μ_x,t|t, Σ_x,t|t and log p(y_t ∣ y_1:t−1).
    """
    obs_operator_is_op = isinstance(obs_operator, lx.AbstractLinearOperator)
    # Σ_y,t = H_t Σ_x,t|t−1 H_tᵀ + Σ_ε,t
    obs_covariance = _innovation_covariance(
        obs_operator,
        predicted_covariance,
        obs_noise_covariance,
        woodbury=woodbury_innovation,
    )

    # Σ_xy,t = Σ_x,t|t−1 H_tᵀ
    if obs_operator_is_op:
        cross_covariance = _right_matmul_transpose(predicted_covariance, obs_operator)
    else:
        cross_covariance = einsum(
            predicted_covariance,
            obs_operator,
            "state col, obs col -> state obs",
        )
    # G_t = Σ_xy,t Σ_y,t⁻¹, row by row since Σ_y,t is symmetric
    gain = solve_rows(obs_covariance, cross_covariance, solver=solver)

    # μ_x,t|t = μ_x,t|t−1 + G_t r_t
    updated_mean = predicted_mean + gain @ residual
    if woodbury_innovation:
        # Avoid materialising Σ_y for the covariance update.
        # H_t Σ_x,t|t−1
        if obs_operator_is_op:
            obs_projection = _left_matmul(obs_operator, predicted_covariance)
        else:
            obs_projection = einsum(
                obs_operator,
                predicted_covariance,
                "obs state, state col -> obs col",
            )
        # G_t H_t Σ_x,t|t−1
        correction = einsum(gain, obs_projection, "state obs, obs col -> state col")
    else:
        obs_covariance_dense = obs_covariance.as_matrix()
        # G_t Σ_y,t
        gain_obs_covariance = einsum(
            gain, obs_covariance_dense, "state obs, obs col -> state col"
        )
        # G_t Σ_y,t G_tᵀ
        correction = einsum(
            gain_obs_covariance,
            gain,
            "state col, other col -> state other",
        )
    # Σ_x,t|t = Σ_x,t|t−1 − G_t Σ_y,t G_tᵀ
    updated_covariance = predicted_covariance - correction
    # The Woodbury form (I − G H) Σ is not symmetric even in exact
    # arithmetic unless G is exact.
    updated_covariance = symmetrize(updated_covariance)

    # Σ_y,t⁻¹ r_t
    whitened_residual = dispatch_solve(obs_covariance, residual, solver)
    # r_tᵀ Σ_y,t⁻¹ r_t
    mahalanobis = residual @ whitened_residual
    # log det Σ_y,t
    logdet = dispatch_logdet(obs_covariance, solver)
    # The dummy unit block contributes −½ log 2π per masked channel to the
    # full-vector density; strip it here, per step, so the result is the
    # exact marginal over the observed entries and is invariant to the
    # dummy variance.
    # log p(y_t ∣ y_1:t−1) = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t
    #     + D_y log 2π) + ½ N_missing log 2π
    log_likelihood_step = (
        -0.5 * (mahalanobis + logdet + obs_size * _LOG_2PI) + 0.5 * n_missing * _LOG_2PI
    )
    return updated_mean, updated_covariance, log_likelihood_step


def kalman_filter(
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
) -> FilterState:
    """Kalman filter forward pass for a linear-Gaussian state-space model.

        x_t = F_t x_t−1 + η_t,    η_t ~ 𝒩(0, Σ_η,t)
        y_t = H_t x_t + ε_t,      ε_t ~ 𝒩(0, Σ_ε,t)

    with x₀ ~ 𝒩(μ_x,0, Σ_x,0), a state x_t of length D_x, an observation
    y_t of length D_y and t = 1, …, T. Every step predicts, then updates, so
    ``observations[0]`` is y₁ (predict first, as in `gaussx.LGSSM`).

    This is a covariance-form filter. For covariances that are PSD by
    construction use `gaussx.parallel_kalman_filter` with
    ``square_root=True`` (gh-454); ``psd_project=True`` projects its
    outputs onto the PSD cone instead.

    Time-invariant inputs are broadcast along the time axis; time-varying
    inputs are passed as stacks over time (e.g. from `discretise_sequence`).
    Operator inputs (``BlockDiag``, ``Kronecker``, ``MaskedOperator``, …)
    are accepted in the time-invariant signature only: F_t x and H_t x run
    through the operator's ``mv``, operator-typed Σ_η and Σ_ε are
    materialised once outside the scan, and the sandwiches F Σ Fᵀ and
    H Σ Hᵀ run inside it because they depend on the evolving covariance.

    Algorithm:
        1. Predict: μ_x,t|t−1 = F_t μ_x,t−1|t−1 and
           Σ_x,t|t−1 = F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t.
        2. Form the residual r_t = y_t − H_t μ_x,t|t−1 and the observation
           covariance Σ_y,t = H_t Σ_x,t|t−1 H_tᵀ + Σ_ε,t.
        3. Solve for the gain G_t = Σ_xy,t Σ_y,t⁻¹ with
           Σ_xy,t = Σ_x,t|t−1 H_tᵀ.
        4. Update: μ_x,t|t = μ_x,t|t−1 + G_t r_t and
           Σ_x,t|t = Σ_x,t|t−1 − G_t Σ_y,t G_tᵀ.
        5. Add log p(y_t | y_1:t−1)
           = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t + D_y log 2π).

    Args:
        transition: Linear dynamics F_t, one for all steps or a stack over
            time, as an array or a `lineax.AbstractLinearOperator`.
        obs_model: Observation operator H_t, one or a stack over time.
        process_noise: Model-error covariance Σ_η,t, one or a stack.
        obs_noise: Observation-error covariance Σ_ε,t, one or a stack.
        observations: Observations y_1:T, one row per step.
        init_mean: Prior mean μ_x,0 of the initial state x₀.
        init_cov: Prior covariance Σ_x,0 of x₀. The filter predicts before
            each update, so ``observations[0]`` is scored against F₁ x₀:
            it observes x₁, not x₀. To observe the prior directly at the
            first step, pass a time-varying transition with
            ``transition[0] = I`` and ``process_noise[0] = 0``.
        mask: Optional observation mask, disambiguated by rank (with one
            channel the per-step and per-channel forms coincide).

            - Per step, shaped like ``observations[:, 0]``: ``True`` runs
              the full predict and update; ``False`` runs the predict only
              and adds nothing to the log-likelihood. Useful for
              prediction on merged train/test grids.
            - Per channel, shaped like ``observations``, for partially
              observed series. ``False`` entries are marginalised out
              exactly: their rows of H_t are zeroed, a unit block replaces
              them in Σ_ε,t and their residual is zero, so
              ``log_likelihood`` is the exact marginal log p(y_obs).
              Masked entries are never read, so they may be ``NaN``. An
              all-``False`` row equals a ``False`` per-step entry.

            Defaults to all ``True``. A per-channel mask materialises
            operator-typed ``obs_model`` and ``obs_noise``, since zeroing
            rows is a dense operation.
        solver: Solver strategy for Σ_y,t; ``None`` uses structural
            dispatch.
        woodbury_innovation: When ``True``, build Σ_y,t as a
            `gaussx.LowRankUpdate` on Σ_ε,t so a structured Σ_ε,t gets
            Woodbury solves and log-determinants. Defaults to ``False``,
            the dense path.

    Raises:
        TypeError: If operator-typed inputs are mixed with time-stacked
            arrays. Per-step structured stacks are not supported; pass
            dense stacks for the time-varying path.

    Returns:
        A ``FilterState`` with the filtered and predicted means and
        covariances and the total log-likelihood.

    Examples:
        A random walk observed through its first component:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> model = (jnp.eye(2), jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> obs_noise = jnp.array([[0.5]])
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> prior = (jnp.zeros(2), jnp.eye(2))
        >>> state = gaussx.kalman_filter(*model, obs_noise, observations, *prior)
        >>> state.filtered_means.shape, state.log_likelihood.shape
        ((3, 2), ())

        A masked step runs the predict only:

        >>> mask = jnp.array([True, False, True])
        >>> gated = gaussx.kalman_filter(
        ...     *model, obs_noise, observations, *prior, mask=mask
        ... )
        >>> bool(jnp.allclose(gated.filtered_means[1], gated.predicted_means[1]))
        True

        A lineax operator transition matches its dense matrix:

        >>> operator = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> structured = gaussx.kalman_filter(
        ...     operator, *model[1:], obs_noise, observations, *prior
        ... )
        >>> bool(jnp.allclose(structured.log_likelihood, state.log_likelihood))
        True
    """
    obs_size = observations.shape[-1]
    num_steps = observations.shape[0]

    # Closure-friendly matvec: when an operator is supplied, prefer its
    # structural ``mv`` over the dense product. Otherwise the broadcast
    # stack holds one dense matrix per step.
    transition_op = (
        transition if isinstance(transition, lx.AbstractLinearOperator) else None
    )
    obs_model_op = (
        obs_model if isinstance(obs_model, lx.AbstractLinearOperator) else None
    )
    obs_noise_op = (
        obs_noise if isinstance(obs_noise, lx.AbstractLinearOperator) else None
    )

    # A per-channel mask rewrites the rows of H and the block structure
    # of Σ_ε, neither of which survives as a structured operator — so the
    # per-channel path drops to dense observation inputs.
    channel_mask = mask is not None and jnp.ndim(mask) == 2
    if channel_mask:
        obs_model_op = None
        obs_noise_op = None

    transitions, obs_models, process_noises, obs_noises, masks, _ = (
        _normalise_tv_inputs(
            transition,
            obs_model,
            process_noise,
            obs_noise,
            T=num_steps,
            mask=mask,
            M=obs_size,
            materialise_transition=transition_op is None,
            materialise_obs=obs_model_op is None,
            # Skip the O(T D_y²) dense broadcast of structured Σ_ε when
            # the Woodbury path consumes the operator directly.
            materialise_obs_noise=not (
                woodbury_innovation and obs_noise_op is not None
            ),
        )
    )

    def step(carry, inputs):
        filtered_mean, filtered_covariance, log_likelihood = carry
        (
            transition_t,
            obs_model_t,
            process_noise_t,
            obs_noise_t,
            obs_t,
            mask_t,
        ) = inputs

        predicted_mean, predicted_covariance = _kalman_predict(
            filtered_mean,
            filtered_covariance,
            transition_t,
            process_noise_t,
            transition_op=transition_op,
        )

        def _update(obs_operator, obs_noise_covariance, residual, n_missing):
            return _kalman_update(
                predicted_mean,
                predicted_covariance,
                obs_operator,
                obs_noise_covariance,
                residual,
                n_missing,
                obs_size=obs_size,
                solver=solver,
                woodbury_innovation=woodbury_innovation,
            )

        if channel_mask:
            # No lax.cond: an all-False row degenerates to the
            # predict-only step on its own, so the masked path is
            # branch-free.
            masked_obs_model, masked_obs_noise, masked_obs, n_missing = (
                _masked_obs_inputs(obs_model_t, obs_noise_t, obs_t, mask_t)
            )
            # H_t μ_x,t|t−1
            predicted_obs = masked_obs_model @ predicted_mean
            # r_t = y_t − H_t μ_x,t|t−1
            residual = masked_obs - predicted_obs
            new_mean, new_covariance, log_likelihood_step = _update(
                masked_obs_model, masked_obs_noise, residual, n_missing
            )
        else:
            # Gate the whole step via lax.cond so the predict-only
            # branch evaluates neither the update arithmetic nor
            # produces gradients for the dropped path.
            def _do_update(_):
                # H_t μ_x,t|t−1
                if obs_model_op is not None:
                    predicted_obs = obs_model_op.mv(predicted_mean)
                else:
                    predicted_obs = obs_model_t @ predicted_mean
                # r_t = y_t − H_t μ_x,t|t−1
                residual = obs_t - predicted_obs
                # Resolve Σ_ε for the update: the operator path uses the
                # closed-over operator (kept structural for Woodbury);
                # the array path falls back to the per-step matrix.
                obs_noise_innov = (
                    obs_noise_op if obs_noise_op is not None else obs_noise_t
                )
                # Resolve H the same way so the operator keeps its
                # structure in both the Woodbury and the
                # structural-sandwich paths.
                obs_model_innov = (
                    obs_model_op if obs_model_op is not None else obs_model_t
                )
                return _update(obs_model_innov, obs_noise_innov, residual, 0.0)

            def _skip_update(_):
                # Match the update branch's dtype: a bare ``jnp.array(0.0)``
                # is float64 under x64 and makes ``lax.cond`` reject the
                # branches on float32 inputs.
                zero = jnp.zeros((), dtype=predicted_covariance.dtype)
                return predicted_mean, predicted_covariance, zero

            new_mean, new_covariance, log_likelihood_step = jax.lax.cond(
                mask_t, _do_update, _skip_update, operand=None
            )
        # log p(y_1:t) = log p(y_1:t−1) + log p(y_t | y_1:t−1)
        new_log_likelihood = log_likelihood + log_likelihood_step

        new_carry = (new_mean, new_covariance, new_log_likelihood)
        outputs = (new_mean, new_covariance, predicted_mean, predicted_covariance)
        return new_carry, outputs

    zero_log_likelihood = jnp.zeros((), dtype=init_cov.dtype)
    init_carry = (init_mean, init_cov, zero_log_likelihood)
    scan_inputs = (
        transitions,
        obs_models,
        process_noises,
        obs_noises,
        observations,
        masks,
    )
    final_carry, scan_outputs = jax.lax.scan(step, init_carry, scan_inputs)
    filtered_means, filtered_covs, predicted_means, predicted_covs = scan_outputs

    return FilterState(
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        log_likelihood=final_carry[2],
    )


def rts_smoother(
    filter_state: FilterState,
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator | None = None,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Rauch-Tung-Striebel backward smoother for `kalman_filter` output.

        μ_x,t|T = μ_x,t|t + G_smooth,t (μ_x,t+1|T − μ_x,t+1|t)
        Σ_x,t|T = Σ_x,t|t + G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ

    with the smoother gain G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹, run
    backwards from μ_x,T|T and Σ_x,T|T. It accepts the same time-invariant,
    time-varying and operator forms of ``transition`` as `kalman_filter`.
    A step masked off in the filter has μ_x,t|t = μ_x,t|t−1, so the
    recursion passes through it harmlessly.

    Algorithm:
        1. Start from the last filtered moments μ_x,T|T and Σ_x,T|T.
        2. For t = T − 1, …, 1, solve for G_smooth,t against Σ_x,t+1|t.
        3. Correct the filtered moments at t with the smoothed and
           predicted moments at t + 1.

    Args:
        filter_state: Output of `kalman_filter`.
        transition: Linear dynamics F_t, as a matrix, a stack over time or
            an operator; the same one passed to the filter.
        process_noise: Deprecated and ignored: the RTS recurrence never
            reads it, since the filter's predicted covariances already
            include Σ_η. Passing it warns; it will be removed in gaussx 0.7.0.
        solver: Solver strategy for Σ_x,t+1|t; ``None`` uses structural
            dispatch.

    Raises:
        TypeError: If an operator-typed ``transition`` has a time axis.
        ValueError: If ``transition`` is neither a matrix nor a stack.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` holding μ_x,t|T and
        Σ_x,t|T.

    Examples:
        Smooth the output of a filter run:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transition = jnp.eye(2)
        >>> model = (transition, jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> state = gaussx.kalman_filter(
        ...     *model, jnp.array([[0.5]]), observations, jnp.zeros(2), jnp.eye(2)
        ... )
        >>> means, covs = gaussx.rts_smoother(state, transition)
        >>> means.shape, covs.shape
        ((3, 2), (3, 2, 2))

        The last smoothed moments are the last filtered ones:

        >>> bool(jnp.allclose(means[-1], state.filtered_means[-1]))
        True
    """
    _warn_unused_process_noise("rts_smoother", process_noise)

    num_steps = filter_state.filtered_means.shape[0]

    # Materialise once outside the scan for the sandwich; matvec stays
    # structural via the operator's mv.
    transition_dense = _materialise(transition)
    transition_op = (
        transition if isinstance(transition, lx.AbstractLinearOperator) else None
    )
    if transition_dense.ndim == 2:
        transitions = repeat(transition_dense, "row col -> t row col", t=num_steps)
    elif transition_dense.ndim == 3:
        if transition_op is not None:
            raise TypeError(
                "Operator-typed transition cannot have a leading time axis."
            )
        transitions = transition_dense
    else:
        raise ValueError(
            f"transition must have ndim 2 or 3, got {transition_dense.ndim}."
        )

    def step(carry, inputs):
        smoothed_mean_next, smoothed_covariance_next = carry
        (
            filtered_mean,
            filtered_covariance,
            predicted_mean_next,
            predicted_covariance_next,
            transition_next,
        ) = inputs

        predicted_covariance_next_op = lx.MatrixLinearOperator(
            predicted_covariance_next, lx.positive_semidefinite_tag
        )
        # Σ_x,t|t F_t+1ᵀ
        cross_covariance = einsum(
            filtered_covariance,
            transition_next,
            "state col, next col -> state next",
        )
        # G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹, row by row (symmetric)
        smoother_gain = solve_rows(
            predicted_covariance_next_op, cross_covariance, solver=solver
        )

        # μ_x,t+1|T − μ_x,t+1|t
        mean_correction = smoothed_mean_next - predicted_mean_next
        # μ_x,t|T = μ_x,t|t + G_smooth,t (μ_x,t+1|T − μ_x,t+1|t)
        smoothed_mean = filtered_mean + smoother_gain @ mean_correction
        # Σ_x,t+1|T − Σ_x,t+1|t
        covariance_correction = smoothed_covariance_next - predicted_covariance_next
        # G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t)
        gain_correction = einsum(
            smoother_gain,
            covariance_correction,
            "state next, next col -> state col",
        )
        # G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ
        correction = einsum(
            gain_correction, smoother_gain, "state col, other col -> state other"
        )
        # Σ_x,t|T = Σ_x,t|t + G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ
        smoothed_covariance = symmetrize(filtered_covariance + correction)

        new_carry = (smoothed_mean, smoothed_covariance)
        return new_carry, new_carry

    init_carry = (
        filter_state.filtered_means[num_steps - 1],
        filter_state.filtered_covs[num_steps - 1],
    )

    # Reverse the sequences for the backward pass (excluding the last
    # step). ``transitions[t + 1]`` maps step ``t`` to step ``t + 1``.
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.filtered_covs[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
        filter_state.predicted_covs[1:][::-1],
        transitions[1:][::-1],
    )

    _, (smoothed_means_rev, smoothed_covs_rev) = jax.lax.scan(step, init_carry, inputs)

    # Reverse back and append the last filtered moments.
    smoothed_means = jnp.concatenate(
        [smoothed_means_rev[::-1], filter_state.filtered_means[num_steps - 1 :]]
    )
    smoothed_covs = jnp.concatenate(
        [smoothed_covs_rev[::-1], filter_state.filtered_covs[num_steps - 1 :]]
    )

    return smoothed_means, smoothed_covs


def kalman_gain(
    P: lx.AbstractLinearOperator,
    H: lx.AbstractLinearOperator,
    R: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
    woodbury_innovation: bool = False,
) -> Float[Array, "N M"]:
    """Kalman gain of a linear-Gaussian observation of the state x.

        G = Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ = Σ_xy Σ_y⁻¹

    Algorithm:
        1. Build the observation covariance Σ_y = H Σ_x Hᵀ + Σ_ε.
        2. Form the cross-covariance Σ_xy = Σ_x Hᵀ.
        3. Solve G Σ_y = Σ_xy row by row, using the symmetry of Σ_y.

    Args:
        P: Prior covariance Σ_x of the state.
        H: Observation operator H.
        R: Observation-error covariance Σ_ε.
        solver: Solver strategy for Σ_y; ``None`` uses structural
            dispatch.
        woodbury_innovation: When ``True``, build Σ_y as a
            `gaussx.LowRankUpdate` on Σ_ε.

    Returns:
        The gain G, one row per state component.

    Examples:
        Observe the first of two independent components:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> prior_covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> obs_operator = lx.MatrixLinearOperator(jnp.array([[1.0, 0.0]]))
        >>> obs_noise = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> gain = gaussx.kalman_gain(prior_covariance, obs_operator, obs_noise)
        >>> gain.shape
        (2, 1)
        >>> bool(jnp.allclose(gain, jnp.array([[0.5], [0.0]])))
        True

        The Woodbury path gives the same gain:

        >>> woodbury = gaussx.kalman_gain(
        ...     prior_covariance, obs_operator, obs_noise, woodbury_innovation=True
        ... )
        >>> bool(jnp.allclose(woodbury, gain))
        True
    """
    prior_covariance = _materialise(P)
    obs_operator = _materialise(H)

    # Σ_y = H Σ_x Hᵀ + Σ_ε
    obs_covariance = _innovation_covariance(H, P, R, woodbury=woodbury_innovation)

    # Σ_xy = Σ_x Hᵀ
    cross_covariance = einsum(
        prior_covariance, obs_operator, "state col, obs col -> state obs"
    )
    # G = Σ_xy Σ_y⁻¹, row by row since Σ_y is symmetric
    gain = solve_rows(obs_covariance, cross_covariance, solver=solver)
    return gain
