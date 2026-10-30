"""Parallel Kalman filter and RTS smoother via associative scan.

The covariance-form parallel filter and smoother of Särkkä &
García-Fernández (IEEE TAC 2021). The forward (filtering) and backward
(smoothing) recurrences are recast as inclusive associative scans of
per-step elements, which `jax.lax.associative_scan` evaluates with
``O(log T)`` depth on parallel hardware (GPU / TPU). On sequential hardware
(CPU) the total work is strictly larger than the ``O(T)`` ``lax.scan`` of
`gaussx.kalman_filter`; the win is on accelerators with large ``T``.

A filtering element of step t holds the conditional p(x_t ∣ x_t−1, y_t)
and the likelihood p(y_t ∣ x_t−1) of the previous state, in information
form:

    p(x_t ∣ x_t−1, y_t) = 𝒩(x_t; F_cond,t x_t−1 + μ_cond,t, Σ_cond,t)
    p(y_t ∣ x_t−1) ∝ exp(−½ x_t−1ᵀ Λ_lik,t x_t−1 + Λμ_lik,tᵀ x_t−1)

The paper's letters map to these as A → F_cond, b → μ_cond, C → Σ_cond,
J → Λ_lik and η → Λμ_lik; η is renamed because it is the model noise here.
Λμ_lik,t is one symbol, the first natural parameter, since Λ_lik,t is
singular whenever D_y < D_x. A smoothing element of step t holds

    p(x_t ∣ x_t+1, y_1:t) = 𝒩(x_t; G_smooth,t x_t+1 + μ_back,t, Σ_back,t),

with the paper's E → G_smooth (the RTS smoother gain), g → μ_back and
L → Σ_back. After the scans μ_cond,t and Σ_cond,t are the filtered moments
μ_x,t|t and Σ_x,t|t, and μ_back,t and Σ_back,t the smoothed μ_x,t|T and
Σ_x,t|T.

``psd_project=True`` projects the returned covariances onto the PSD cone,
a safety net for ill-conditioned float32 chains, not a square-root filter:
the scan itself still runs in covariance form
(`gaussx._ssm._parallel_kalman_sqrt`). ``square_root=True`` runs the
factor-propagating combinator of Yaghoobi et al. (2022) instead
(`gaussx._ssm._parallel_kalman_factor`, gh-454).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._deprecation import warn_deprecated
from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import einsum, rearrange, reduce, repeat
from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._logdet import cholesky_logdet
from gaussx._ssm._kalman import (
    FilterState,
    _empty_filter_state,
    _kalman_predict,
    kalman_filter,
)
from gaussx._ssm._parallel_kalman_factor import parallel_kalman_filter_factor
from gaussx._ssm._scan_elements import (
    _filter_combine,
    _filter_elements,
    _smoother_combine,
    _smoother_element,
)
from gaussx._ssm._utils import (
    _masked_obs_inputs,
    _normalise_tv_inputs,
    _transition_stack,
    _warn_unused_process_noise,
)
from gaussx._strategies._base import AbstractSolverStrategy


# ----------------------------------------------------------------
# Moments and likelihood from the scanned elements
# ----------------------------------------------------------------


def _predicted_moments(
    filtered_means: Float[Array, "T N"],
    filtered_covs: Float[Array, "T N N"],
    transitions: Float[Array, "T N N"],
    process_noises: Float[Array, "T N N"],
    init_mean: Float[Array, " N"],
    init_cov: Float[Array, "N N"],
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Predicted moments of every step, from the filtered ones of the last.

        μ_x,t|t−1 = F_t μ_x,t−1|t−1
        Σ_x,t|t−1 = F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t

    with μ_x,0|0 = μ_x,0 and Σ_x,0|0 = Σ_x,0.

    Args:
        filtered_means: Filtered means μ_x,t|t.
        filtered_covs: Filtered covariances Σ_x,t|t.
        transitions: Dynamics F_t, stacked over time.
        process_noises: Model-error covariances Σ_η,t, stacked over time.
        init_mean: Prior mean μ_x,0.
        init_cov: Prior covariance Σ_x,0.

    Returns:
        Tuple ``(predicted_means, predicted_covs)`` holding μ_x,t|t−1 and
        Σ_x,t|t−1.
    """
    # μ_x,0:T−1|0:T−1, the prior first
    previous_means = jnp.concatenate(
        [rearrange(init_mean, "state -> 1 state"), filtered_means[:-1]]
    )
    # Σ_x,0:T−1|0:T−1, the prior first
    previous_covs = jnp.concatenate(
        [rearrange(init_cov, "row col -> 1 row col"), filtered_covs[:-1]]
    )
    predict = functools.partial(_kalman_predict, transition_op=None)
    return jax.vmap(predict)(previous_means, previous_covs, transitions, process_noises)


def _log_likelihood_step(
    obs_t: Float[Array, " M"],
    predicted_mean: Float[Array, " N"],
    predicted_covariance: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    obs_noise_covariance: Float[Array, "M M"],
    mask_t: Bool[Array, " M"],
    active: Bool[Array, ""],
    *,
    obs_size: int,
) -> Float[Array, ""]:
    """Log-likelihood term of one step over its observed channels.

        log p(y_t ∣ y_1:t−1) = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t + D_y log 2π)
                               + ½ N_missing log 2π

    Masked channels get the same substitution as the filtering elements,
    so they cannot drive the Cholesky through an ill-conditioned Σ_ε,t or
    make the gradient NaN.

    Args:
        obs_t: Observation y_t.
        predicted_mean: Predicted mean μ_x,t|t−1.
        predicted_covariance: Predicted covariance Σ_x,t|t−1.
        obs_operator: Observation operator H_t.
        obs_noise_covariance: Observation-error covariance Σ_ε,t.
        mask_t: Per-channel mask of y_t; ``True`` keeps a channel.
        active: Whether the step observes any channel; an inactive step
            contributes 0.
        obs_size: Observation length D_y, masked channels included.

    Returns:
        log p(y_t ∣ y_1:t−1), or 0 for an inactive step.
    """
    masked_obs_operator, masked_obs_noise, masked_obs, n_missing = _masked_obs_inputs(
        obs_operator, obs_noise_covariance, obs_t, mask_t
    )
    # H_t μ_x,t|t−1
    predicted_obs = masked_obs_operator @ predicted_mean
    # r_t = y_t − H_t μ_x,t|t−1
    residual = masked_obs - predicted_obs
    # H_t Σ_x,t|t−1
    obs_projection = einsum(
        masked_obs_operator, predicted_covariance, "obs state, state col -> obs col"
    )
    # H_t Σ_x,t|t−1 H_tᵀ
    obs_sandwich = einsum(
        obs_projection, masked_obs_operator, "obs col, other col -> obs other"
    )
    # Σ_y,t = H_t Σ_x,t|t−1 H_tᵀ + Σ_ε,t
    obs_covariance = symmetrize(obs_sandwich + masked_obs_noise)
    # L_y,t L_y,tᵀ = Σ_y,t
    obs_factor = jnp.linalg.cholesky(obs_covariance)
    # Σ_y,t⁻¹ r_t
    whitened_residual = jax.scipy.linalg.cho_solve((obs_factor, True), residual)
    # r_tᵀ Σ_y,t⁻¹ r_t
    mahalanobis = residual @ whitened_residual
    # log det Σ_y,t
    logdet = cholesky_logdet(obs_factor)
    # Strip the dummy unit block's −½ log 2π per masked channel, so this is
    # the exact marginal over the observed entries.
    # log p(y_t ∣ y_1:t−1) = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t
    #     + D_y log 2π) + ½ N_missing log 2π
    log_likelihood_step = (
        -0.5 * (mahalanobis + logdet + obs_size * _LOG_2PI) + 0.5 * n_missing * _LOG_2PI
    )
    return jnp.where(active, log_likelihood_step, jnp.zeros_like(log_likelihood_step))


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
    square_root: bool = False,
) -> FilterState:
    """Parallel Kalman filter via `jax.lax.associative_scan`.

        x_t = F_t x_t−1 + η_t,    η_t ~ 𝒩(0, Σ_η,t)
        y_t = H_t x_t + ε_t,      ε_t ~ 𝒩(0, Σ_ε,t)

    the model of `gaussx.kalman_filter`, with x₀ ~ 𝒩(μ_x,0, Σ_x,0) and
    t = 1, …, T. The result matches `gaussx.kalman_filter` to round-off for
    the default ``solver=None``, with ``O(log T)`` parallel depth on
    accelerators: the same predict-first time convention, the same
    time-invariant, time-varying and operator-typed inputs, the same masks
    and a scalar log-likelihood. An empty window (T = 0) returns a
    zero-length `FilterState` with ``log_likelihood == 0``.

    Algorithm:
        1. Build one filtering element (F_cond,t, μ_cond,t, Σ_cond,t,
           Λμ_lik,t, Λ_lik,t) per step, the first absorbing the prior; the
           module docstring defines them.
        2. Run the inclusive associative scan; μ_cond,t and Σ_cond,t of the
           result are μ_x,t|t and Σ_x,t|t.
        3. Recover μ_x,t|t−1 and Σ_x,t|t−1 by predicting from the filtered
           moments of the step before.
        4. Sum log p(y_t ∣ y_1:t−1) = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t
           + D_y log 2π) over the observed steps.

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
        mask: Optional observation mask, dispatched on rank exactly as in
            `gaussx.kalman_filter`. Shape ``(T,)`` gates whole steps
            (``False`` runs the predict only and adds 0 to the
            log-likelihood); shape ``(T, M)`` gates single channels and
            gives the exact marginal log-likelihood over the observed
            entries. Defaults to all ``True``. A ``(T, M)`` mask is not
            supported with ``psd_project=True``.
        solver: Used only with ``woodbury_innovation=True``, which
            delegates to `gaussx.kalman_filter`. The associative-scan
            combinators use dense solves, so passing ``solver`` without
            ``woodbury_innovation`` is deprecated (it warns, and will raise
            in gaussx 0.7.0).
        woodbury_innovation: When ``True``, delegate to
            `gaussx.kalman_filter` with the same flag, so a structured
            Σ_ε,t takes the Woodbury innovation path.
        form: ``"covariance"``. The former ``"sqrt"`` is a deprecated
            spelling of ``psd_project=True`` (it warns, and will be removed
            in gaussx 0.7.0): it never was a square-root filter.
        psd_project: Project each returned covariance onto the PSD cone
            (eigenvalue clip) and keep lower-triangular factors of the
            projections. The associative scan still runs in covariance
            form, so this has the covariance form's conditioning; it only
            guarantees PSD outputs, which float32 chains with very small
            observation noise can otherwise lose (the covariance form can
            return an indefinite covariance and a NaN log-likelihood
            there). Gradients are those of the unprojected path. For
            covariances that are PSD by construction use ``square_root``.
        square_root: Run the factor-propagating square-root filter of
            Yaghoobi, Corenflos, Hassan & Särkkä (2022, §III): covariances
            are PSD by construction and stay accurate in float32. See
            `gaussx._ssm._parallel_kalman_factor` for the method. Supports
            both mask ranks; not combinable with ``psd_project`` or
            ``woodbury_innovation``.

    Raises:
        ValueError: If ``form`` is not ``"covariance"`` or ``"sqrt"``, or
            ``square_root`` is combined with ``psd_project`` or
            ``woodbury_innovation``.
        NotImplementedError: If ``psd_project=True`` gets a ``(T, M)``
            mask.

    Returns:
        A `FilterState` with the filtered and predicted means and
        covariances and the total log-likelihood.

    Examples:
        A random walk observed through its first component matches the
        sequential filter:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> model = (jnp.eye(2), jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> obs_noise = jnp.array([[0.5]])
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> prior = (jnp.zeros(2), jnp.eye(2))
        >>> state = gaussx.parallel_kalman_filter(
        ...     *model, obs_noise, observations, *prior
        ... )
        >>> state.filtered_means.shape, state.log_likelihood.shape
        ((3, 2), ())
        >>> sequential = gaussx.kalman_filter(*model, obs_noise, observations, *prior)
        >>> bool(jnp.allclose(state.log_likelihood, sequential.log_likelihood))
        True

        A masked step runs the predict only:

        >>> mask = jnp.array([True, False, True])
        >>> gated = gaussx.parallel_kalman_filter(
        ...     *model, obs_noise, observations, *prior, mask=mask
        ... )
        >>> bool(jnp.allclose(gated.filtered_means[1], gated.predicted_means[1]))
        True

        The square-root form agrees with the covariance form:

        >>> factored = gaussx.parallel_kalman_filter(
        ...     *model, obs_noise, observations, *prior, square_root=True
        ... )
        >>> bool(jnp.allclose(factored.filtered_covs, state.filtered_covs, atol=1e-5))
        True

    References:
        Särkkä, S. & García-Fernández, Á. F. (2021). Temporal parallelization
        of Bayesian smoothers. *IEEE Transactions on Automatic Control* 66(1),
        299-306.

        Yaghoobi, F., Corenflos, A., Hassan, S. & Särkkä, S. (2022). Parallel
        square-root statistical linear regression for inference in nonlinear
        state space models. arXiv:2207.00426.
    """
    if solver is not None and not woodbury_innovation:
        warn_deprecated(
            "parallel_kalman_filter(solver=...) has no effect unless "
            "woodbury_innovation=True: the associative-scan combinators use "
            "dense solves. Passing it otherwise is deprecated and will raise "
            "in gaussx 0.7.0."
        )
    if form == "sqrt":
        warn_deprecated(
            'form="sqrt" is deprecated: it is a PSD projection of the '
            "covariance-form combinator, not a square-root filter. Pass "
            'psd_project=True instead; form="sqrt" will be removed in gaussx 0.7.0.'
        )
        psd_project = True
    elif form != "covariance":
        raise ValueError("form must be 'covariance' or 'sqrt'.")
    if square_root:
        if psd_project or woodbury_innovation:
            raise ValueError(
                "square_root=True cannot be combined with psd_project=True "
                "(or form='sqrt') or woodbury_innovation=True."
            )
        return parallel_kalman_filter_factor(
            transition,
            obs_model,
            process_noise,
            obs_noise,
            observations,
            init_mean,
            init_cov,
            mask=mask,
        )
    if psd_project:
        # lazy import, cycle: _ssm._parallel_kalman_sqrt -> _ssm._parallel_kalman
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

    obs_size = observations.shape[-1]
    num_steps = observations.shape[0]

    # Empty observation window: match kalman_filter's empty-scan output.
    if num_steps == 0:
        return _empty_filter_state(init_mean, init_cov)

    transitions, obs_operators, process_noises, obs_noises, masks, _ = (
        _normalise_tv_inputs(
            transition,
            obs_model,
            process_noise,
            obs_noise,
            T=num_steps,
            mask=mask,
            M=obs_size,
        )
    )
    # Work per channel throughout: a (T,) gate is the special case where
    # every channel of a step shares one flag, so broadcasting it
    # reproduces the whole-step path exactly.
    if masks.ndim == 2:
        channel_masks = masks
    else:
        channel_masks = repeat(masks, "t -> t obs", obs=obs_size)
    step_active = reduce(channel_masks, "t obs -> t", "any")

    elements = _filter_elements(
        transitions,
        obs_operators,
        process_noises,
        obs_noises,
        observations,
        channel_masks,
        step_active,
        init_mean,
        init_cov,
    )
    # (F_cond,1:t, μ_x,t|t, Σ_x,t|t, Λμ_lik,1:t, Λ_lik,1:t) for every t
    _, filtered_means, scanned_covs, _, _ = jax.lax.associative_scan(
        _filter_combine, elements
    )
    filtered_covs = jax.vmap(symmetrize)(scanned_covs)

    predicted_means, predicted_covs = _predicted_moments(
        filtered_means,
        filtered_covs,
        transitions,
        process_noises,
        init_mean,
        init_cov,
    )

    log_likelihood_step = functools.partial(_log_likelihood_step, obs_size=obs_size)
    log_likelihood_steps = jax.vmap(log_likelihood_step)(
        observations,
        predicted_means,
        predicted_covs,
        obs_operators,
        obs_noises,
        channel_masks,
        step_active,
    )
    # log p(y_1:T) = ∑_t log p(y_t ∣ y_1:t−1)
    log_likelihood = jnp.sum(log_likelihood_steps)

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

        μ_x,t|T = μ_x,t|t + G_smooth,t (μ_x,t+1|T − μ_x,t+1|t)
        Σ_x,t|T = Σ_x,t|t + G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ

    with G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹. Pairs with
    `parallel_kalman_filter` and matches `gaussx.rts_smoother` to round-off,
    with ``O(log T)`` parallel depth.

    Algorithm:
        1. Build one smoothing element (G_smooth,t, μ_back,t, Σ_back,t) per
           step t < T, and (0, μ_x,T|T, Σ_x,T|T) for the last; the module
           docstring defines them.
        2. Run the reverse associative scan; μ_back,t and Σ_back,t of the
           result are μ_x,t|T and Σ_x,t|T.

    Args:
        filter_state: Output of `parallel_kalman_filter` or
            `gaussx.kalman_filter`.
        transition: Linear dynamics F_t, as a matrix, a stack over time or
            an operator; the same one passed to the filter.
        process_noise: Deprecated and ignored, as in `gaussx.rts_smoother`;
            it will be removed in gaussx 0.7.0.
        solver: Accepted for API symmetry; not threaded through.
        form: ``"covariance"``. ``"sqrt"`` is a deprecated spelling of
            ``psd_project=True``, removed in gaussx 0.7.0.
        psd_project: Build the smoothing elements from PSD-projected
            factors and combine the factors with QR, so the smoothed
            covariances are PSD. As in `parallel_kalman_filter`, the
            elements come from covariance-form quantities, so this is a
            PSD safety net rather than a square-root smoother.

    Raises:
        ValueError: If ``form`` is not ``"covariance"`` or ``"sqrt"``, or
            ``transition`` is neither a matrix nor a stack.
        TypeError: If an operator-typed ``transition`` has a time axis.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` holding μ_x,t|T and
        Σ_x,t|T.

    Examples:
        Smooth a filter run; it matches the sequential smoother:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transition = jnp.eye(2)
        >>> model = (transition, jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> state = gaussx.parallel_kalman_filter(
        ...     *model, jnp.array([[0.5]]), observations, jnp.zeros(2), jnp.eye(2)
        ... )
        >>> means, covs = gaussx.parallel_rts_smoother(state, transition)
        >>> means.shape, covs.shape
        ((3, 2), (3, 2, 2))
        >>> sequential_means, _ = gaussx.rts_smoother(state, transition)
        >>> bool(jnp.allclose(means, sequential_means))
        True

        The last smoothed moments are the last filtered ones:

        >>> bool(jnp.allclose(means[-1], state.filtered_means[-1]))
        True
    """
    _warn_unused_process_noise("parallel_rts_smoother", process_noise)
    if form == "sqrt":
        warn_deprecated(
            'form="sqrt" is deprecated: it is a PSD projection of the '
            "covariance-form combinator, not a square-root filter. Pass "
            'psd_project=True instead; form="sqrt" will be removed in gaussx 0.7.0.'
        )
        psd_project = True
    elif form != "covariance":
        raise ValueError("form must be 'covariance' or 'sqrt'.")
    if psd_project:
        # lazy import, cycle: _ssm._parallel_kalman_sqrt -> _ssm._parallel_kalman
        from gaussx._ssm._parallel_kalman_sqrt import parallel_rts_smoother_sqrt

        return parallel_rts_smoother_sqrt(filter_state, transition, solver=solver)

    del solver

    filtered_means = filter_state.filtered_means
    filtered_covs = filter_state.filtered_covs
    num_steps = filtered_means.shape[0]
    state_size = filtered_means.shape[-1]

    if num_steps == 0:
        return (
            jnp.zeros((0, state_size), dtype=filtered_means.dtype),
            jnp.zeros((0, state_size, state_size), dtype=filtered_covs.dtype),
        )

    transitions = _transition_stack(transition, num_steps)
    inner_gains, inner_means, inner_covs = jax.vmap(_smoother_element)(
        filtered_means[:-1],
        filtered_covs[:-1],
        filter_state.predicted_means[1:],
        filter_state.predicted_covs[1:],
        transitions[1:],
    )
    # Last element: G_smooth,T = 0, μ_back,T = μ_x,T|T, Σ_back,T = Σ_x,T|T
    last_gain = jnp.zeros((1, state_size, state_size), dtype=filtered_means.dtype)
    smoother_gains = jnp.concatenate([inner_gains, last_gain])
    back_means = jnp.concatenate([inner_means, filtered_means[-1:]])
    back_covs = jnp.concatenate([inner_covs, filtered_covs[-1:]])

    # (G_smooth,t:T, μ_x,t|T, Σ_x,t|T) for every t
    _, smoothed_means, smoothed_covs = jax.lax.associative_scan(
        _smoother_combine, (smoother_gains, back_means, back_covs), reverse=True
    )
    smoothed_covs = jax.vmap(symmetrize)(smoothed_covs)
    return smoothed_means, smoothed_covs
