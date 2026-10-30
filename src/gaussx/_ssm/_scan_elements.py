"""Elements and combinators of the parallel Kalman filter and smoother.

The per-step filtering and smoothing elements of Särkkä &
García-Fernández (IEEE TAC 2021) and the associative operators that
`jax.lax.associative_scan` combines them with, shared by the covariance
filter (`gaussx._ssm._parallel_kalman`) and the PSD-projected square-root
filter (`gaussx._ssm._parallel_kalman_sqrt`). The notation is defined in
the module docstring of `gaussx._ssm._parallel_kalman`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array, Bool, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._symmetrize import symmetrize
from gaussx._ssm._utils import _masked_obs_inputs


SmootherElement = tuple[
    Float[Array, "*T N N"], Float[Array, "*T N"], Float[Array, "*T N N"]
]


FilterElement = tuple[
    Float[Array, "*T N N"],
    Float[Array, "*T N"],
    Float[Array, "*T N N"],
    Float[Array, "*T N"],
    Float[Array, "*T N N"],
]


# ----------------------------------------------------------------
# Filter element builders
# ----------------------------------------------------------------


def _first_filter_element_active(
    transition_t: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    process_noise_t: Float[Array, "N N"],
    obs_noise_covariance: Float[Array, "M M"],
    obs_t: Float[Array, " M"],
    init_mean: Float[Array, " N"],
    init_cov: Float[Array, "N N"],
) -> FilterElement:
    """First filtering element: absorb the prior on x₀, predict, then update.

        F_cond,1 = 0,   μ_cond,1 = μ_x,1|1,   Σ_cond,1 = Σ_x,1|1,
        Λμ_lik,1 = 0,   Λ_lik,1 = 0

    Algorithm:
        1. Predict: μ_x,1|0 = F₁ μ_x,0 and Σ_x,1|0 = F₁ Σ_x,0 F₁ᵀ + Σ_η,1.
        2. Factor Σ_y,1 = H₁ Σ_x,1|0 H₁ᵀ + Σ_ε,1 and solve for the gain
           G₁ = Σ_x,1|0 H₁ᵀ Σ_y,1⁻¹.
        3. Update: μ_x,1|1 = μ_x,1|0 + G₁ r₁ and
           Σ_x,1|1 = Σ_x,1|0 − G₁ H₁ Σ_x,1|0.

    Args:
        transition_t: Dynamics F₁.
        obs_operator: Observation operator H₁, masked rows zeroed.
        process_noise_t: Model-error covariance Σ_η,1.
        obs_noise_covariance: Observation-error covariance Σ_ε,1.
        obs_t: Observation y₁.
        init_mean: Prior mean μ_x,0.
        init_cov: Prior covariance Σ_x,0.

    Returns:
        The element ``(F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik)`` of step 1.
    """
    state_size = transition_t.shape[0]
    # μ_x,1|0 = F₁ μ_x,0
    predicted_mean = transition_t @ init_mean
    # F₁ Σ_x,0
    transition_covariance = einsum(
        transition_t, init_cov, "next prev, prev col -> next col"
    )
    # F₁ Σ_x,0 F₁ᵀ
    propagated = einsum(
        transition_covariance, transition_t, "next col, other col -> next other"
    )
    # Σ_x,1|0 = F₁ Σ_x,0 F₁ᵀ + Σ_η,1
    predicted_covariance = symmetrize(propagated + process_noise_t)
    # H₁ Σ_x,1|0 = Σ_xy,1ᵀ
    obs_projection = einsum(
        obs_operator, predicted_covariance, "obs state, state col -> obs col"
    )
    # H₁ Σ_x,1|0 H₁ᵀ
    obs_sandwich = einsum(
        obs_projection, obs_operator, "obs col, other col -> obs other"
    )
    # Σ_y,1 = H₁ Σ_x,1|0 H₁ᵀ + Σ_ε,1
    obs_covariance = symmetrize(obs_sandwich + obs_noise_covariance)
    # L_y,1 L_y,1ᵀ = Σ_y,1
    obs_factor = jnp.linalg.cholesky(obs_covariance)
    # G₁ᵀ = Σ_y,1⁻¹ H₁ Σ_x,1|0
    gain_transpose = jax.scipy.linalg.cho_solve((obs_factor, True), obs_projection)
    # G₁ = Σ_xy,1 Σ_y,1⁻¹
    gain = rearrange(gain_transpose, "obs state -> state obs")
    # F_cond,1 = 0: the prior is absorbed, nothing earlier to condition on
    cond_transition = jnp.zeros((state_size, state_size), dtype=transition_t.dtype)
    # H₁ μ_x,1|0
    predicted_obs = obs_operator @ predicted_mean
    # r₁ = y₁ − H₁ μ_x,1|0
    residual = obs_t - predicted_obs
    # μ_cond,1 = μ_x,1|0 + G₁ r₁
    cond_mean = predicted_mean + gain @ residual
    # G₁ H₁ Σ_x,1|0
    correction = einsum(gain, obs_projection, "state obs, obs col -> state col")
    # Σ_cond,1 = Σ_x,1|0 − G₁ H₁ Σ_x,1|0
    cond_covariance = symmetrize(predicted_covariance - correction)
    # Λμ_lik,1 = 0 and Λ_lik,1 = 0
    lik_information = jnp.zeros(state_size, dtype=transition_t.dtype)
    lik_precision = jnp.zeros((state_size, state_size), dtype=transition_t.dtype)
    return (
        cond_transition,
        cond_mean,
        cond_covariance,
        lik_information,
        lik_precision,
    )


def _first_filter_element_masked(
    transition_t: Float[Array, "N N"],
    process_noise_t: Float[Array, "N N"],
    init_mean: Float[Array, " N"],
    init_cov: Float[Array, "N N"],
) -> FilterElement:
    """First filtering element when y₁ is masked: absorb the prior, predict.

        F_cond,1 = 0,   μ_cond,1 = F₁ μ_x,0,   Σ_cond,1 = F₁ Σ_x,0 F₁ᵀ + Σ_η,1

    Args:
        transition_t: Dynamics F₁.
        process_noise_t: Model-error covariance Σ_η,1.
        init_mean: Prior mean μ_x,0.
        init_cov: Prior covariance Σ_x,0.

    Returns:
        The element ``(F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik)`` of step 1.
    """
    state_size = transition_t.shape[0]
    # F_cond,1 = 0
    cond_transition = jnp.zeros((state_size, state_size), dtype=transition_t.dtype)
    # μ_cond,1 = F₁ μ_x,0
    cond_mean = transition_t @ init_mean
    # F₁ Σ_x,0
    transition_covariance = einsum(
        transition_t, init_cov, "next prev, prev col -> next col"
    )
    # F₁ Σ_x,0 F₁ᵀ
    propagated = einsum(
        transition_covariance, transition_t, "next col, other col -> next other"
    )
    # Σ_cond,1 = F₁ Σ_x,0 F₁ᵀ + Σ_η,1
    cond_covariance = symmetrize(propagated + process_noise_t)
    # Λμ_lik,1 = 0 and Λ_lik,1 = 0
    lik_information = jnp.zeros(state_size, dtype=transition_t.dtype)
    lik_precision = jnp.zeros((state_size, state_size), dtype=transition_t.dtype)
    return (
        cond_transition,
        cond_mean,
        cond_covariance,
        lik_information,
        lik_precision,
    )


def _generic_filter_element_active(
    transition_t: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    process_noise_t: Float[Array, "N N"],
    obs_noise_covariance: Float[Array, "M M"],
    obs_t: Float[Array, " M"],
) -> FilterElement:
    """Filtering element of step t ≥ 2: x_t−1 given, predict, then update.

        Σ_y∣x,t = H_t Σ_η,t H_tᵀ + Σ_ε,t,   G_t = Σ_η,t H_tᵀ Σ_y∣x,t⁻¹
        F_cond,t = (I − G_t H_t) F_t,       μ_cond,t = G_t y_t
        Σ_cond,t = Σ_η,t − G_t H_t Σ_η,t
        Λμ_lik,t = F_tᵀ H_tᵀ Σ_y∣x,t⁻¹ y_t,  Λ_lik,t = F_tᵀ H_tᵀ Σ_y∣x,t⁻¹ H_t F_t

    Σ_y∣x,t is the covariance of y_t given x_t−1, so Σ_η,t stands in for
    the predicted covariance. Σ_y∣x,t is factored once; the gain and both
    information solves reuse its Cholesky factor.

    Args:
        transition_t: Dynamics F_t.
        obs_operator: Observation operator H_t, masked rows zeroed.
        process_noise_t: Model-error covariance Σ_η,t.
        obs_noise_covariance: Observation-error covariance Σ_ε,t, with a
            unit block on masked channels.
        obs_t: Observation y_t, masked entries zeroed.

    Returns:
        The element ``(F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik)`` of step t.
    """
    state_size = transition_t.shape[0]
    # H_t Σ_η,t
    obs_projection = einsum(
        obs_operator, process_noise_t, "obs state, state col -> obs col"
    )
    # H_t Σ_η,t H_tᵀ
    obs_sandwich = einsum(
        obs_projection, obs_operator, "obs col, other col -> obs other"
    )
    # Σ_y∣x,t = H_t Σ_η,t H_tᵀ + Σ_ε,t
    obs_covariance = symmetrize(obs_sandwich + obs_noise_covariance)
    # L_y,t L_y,tᵀ = Σ_y∣x,t
    obs_factor = jnp.linalg.cholesky(obs_covariance)
    # G_tᵀ = Σ_y∣x,t⁻¹ H_t Σ_η,t
    gain_transpose = jax.scipy.linalg.cho_solve((obs_factor, True), obs_projection)
    # G_t = Σ_η,t H_tᵀ Σ_y∣x,t⁻¹
    gain = rearrange(gain_transpose, "obs state -> state obs")
    # G_t H_t
    gain_obs = einsum(gain, obs_operator, "state obs, obs col -> state col")
    # I − G_t H_t
    identity_minus_gain_obs = jnp.eye(state_size, dtype=transition_t.dtype) - gain_obs
    # F_cond,t = (I − G_t H_t) F_t
    cond_transition = einsum(
        identity_minus_gain_obs, transition_t, "state col, col prev -> state prev"
    )
    # μ_cond,t = G_t y_t
    cond_mean = gain @ obs_t
    # G_t H_t Σ_η,t
    correction = einsum(gain, obs_projection, "state obs, obs col -> state col")
    # Σ_cond,t = Σ_η,t − G_t H_t Σ_η,t
    cond_covariance = symmetrize(process_noise_t - correction)
    # H_t F_t
    obs_transition = einsum(
        obs_operator, transition_t, "obs state, state prev -> obs prev"
    )
    # Σ_y∣x,t⁻¹ y_t
    whitened_obs = jax.scipy.linalg.cho_solve((obs_factor, True), obs_t)
    # Σ_y∣x,t⁻¹ H_t F_t
    whitened_obs_transition = jax.scipy.linalg.cho_solve(
        (obs_factor, True), obs_transition
    )
    # Λμ_lik,t = F_tᵀ H_tᵀ Σ_y∣x,t⁻¹ y_t
    lik_information = einsum(obs_transition, whitened_obs, "obs prev, obs -> prev")
    # F_tᵀ H_tᵀ Σ_y∣x,t⁻¹ H_t F_t
    lik_precision_raw = einsum(
        obs_transition, whitened_obs_transition, "obs row, obs col -> row col"
    )
    # Λ_lik,t = F_tᵀ H_tᵀ Σ_y∣x,t⁻¹ H_t F_t
    lik_precision = symmetrize(lik_precision_raw)
    return (
        cond_transition,
        cond_mean,
        cond_covariance,
        lik_information,
        lik_precision,
    )


def _masked_filter_element(
    transition_t: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    process_noise_t: Float[Array, "N N"],
    obs_noise_covariance: Float[Array, "M M"],
    obs_t: Float[Array, " M"],
    mask_t: Bool[Array, " M"],
) -> FilterElement:
    """Filtering element of step t ≥ 2 with masked channels made inert.

    ``vmap`` of ``lax.cond`` evaluates both branches and selects, so a
    masked step does not branch: `_masked_obs_inputs` zeroes the rows of
    H_t and the entries of y_t and puts a unit block in Σ_ε,t. On a fully
    masked step that collapses the active element to the predict-only one,
    (F_t, 0, Σ_η,t, 0, 0), and the Cholesky factors the identity, so even
    garbage in masked inputs cannot make the gradient NaN.

    Args:
        transition_t: Dynamics F_t.
        obs_operator: Observation operator H_t.
        process_noise_t: Model-error covariance Σ_η,t.
        obs_noise_covariance: Observation-error covariance Σ_ε,t.
        obs_t: Observation y_t.
        mask_t: Per-channel mask of y_t; ``True`` keeps a channel.

    Returns:
        The element ``(F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik)`` of step t.
    """
    masked_obs_operator, masked_obs_noise, masked_obs, _ = _masked_obs_inputs(
        obs_operator, obs_noise_covariance, obs_t, mask_t
    )
    return _generic_filter_element_active(
        transition_t, masked_obs_operator, process_noise_t, masked_obs_noise, masked_obs
    )


def _filter_elements(
    transitions: Float[Array, "T N N"],
    obs_operators: Float[Array, "T M N"],
    process_noises: Float[Array, "T N N"],
    obs_noises: Float[Array, "T M M"],
    observations: Float[Array, "T M"],
    channel_masks: Bool[Array, "T M"],
    step_active: Bool[Array, " T"],
    init_mean: Float[Array, " N"],
    init_cov: Float[Array, "N N"],
) -> FilterElement:
    """Stack the filtering elements of every step, the first absorbing x₀.

    Algorithm:
        1. Build the generic element of every step, masks applied.
        2. Replace element 1 by one that absorbs 𝒩(μ_x,0, Σ_x,0). An outer
           ``lax.cond`` skips the inactive branch here, since nothing is
           vmapped; a partly observed step 1 takes the active branch on
           the masked inputs.

    Args:
        transitions: Dynamics F_t, stacked over time.
        obs_operators: Observation operators H_t, stacked over time.
        process_noises: Model-error covariances Σ_η,t, stacked over time.
        obs_noises: Observation-error covariances Σ_ε,t, stacked over time.
        observations: Observations y_1:T.
        channel_masks: Per-channel masks; ``True`` keeps a channel.
        step_active: Whether step t observes any channel.
        init_mean: Prior mean μ_x,0.
        init_cov: Prior covariance Σ_x,0.

    Returns:
        The elements ``(F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik)``, stacked
        over time.
    """
    elements = jax.vmap(_masked_filter_element)(
        transitions,
        obs_operators,
        process_noises,
        obs_noises,
        observations,
        channel_masks,
    )
    first_obs_operator, first_obs_noise, first_obs, _ = _masked_obs_inputs(
        obs_operators[0], obs_noises[0], observations[0], channel_masks[0]
    )
    first = jax.lax.cond(
        step_active[0],
        lambda: _first_filter_element_active(
            transitions[0],
            first_obs_operator,
            process_noises[0],
            first_obs_noise,
            first_obs,
            init_mean,
            init_cov,
        ),
        lambda: _first_filter_element_masked(
            transitions[0],
            process_noises[0],
            init_mean,
            init_cov,
        ),
    )
    cond_transitions, cond_means, cond_covs, lik_informations, lik_precisions = elements
    return (
        cond_transitions.at[0].set(first[0]),
        cond_means.at[0].set(first[1]),
        cond_covs.at[0].set(first[2]),
        lik_informations.at[0].set(first[3]),
        lik_precisions.at[0].set(first[4]),
    )


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


def _filter_combine(
    earlier: FilterElement,
    later: FilterElement,
) -> FilterElement:
    """Combine an earlier filtering element i with a later element j.

        F_cond,ij = F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ F_cond,i
        μ_cond,ij = F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ (μ_cond,i + Σ_cond,i Λμ_lik,j)
                    + μ_cond,j
        Σ_cond,ij = F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ Σ_cond,i F_cond,jᵀ + Σ_cond,j
        Λμ_lik,ij = F_cond,iᵀ (I + Λ_lik,j Σ_cond,i)⁻¹ (Λμ_lik,j − Λ_lik,j μ_cond,i)
                    + Λμ_lik,i
        Λ_lik,ij = F_cond,iᵀ (I + Λ_lik,j Σ_cond,i)⁻¹ Λ_lik,j F_cond,i + Λ_lik,i

    This is Särkkä & García-Fernández (2021, §III.A). ``lax.associative_scan``
    passes chunks with a leading scan axis that may have length zero, so
    the rule works on the trailing two axes with ``@``, ``swapaxes`` and
    `_bmv`, never einx.

    Args:
        earlier: Element i, ``(F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik)``.
        later: Element j, in the same layout.

    Returns:
        The combined element ij.
    """
    (
        cond_transition_i,
        cond_mean_i,
        cond_covariance_i,
        lik_information_i,
        lik_precision_i,
    ) = earlier
    (
        cond_transition_j,
        cond_mean_j,
        cond_covariance_j,
        lik_information_j,
        lik_precision_j,
    ) = later
    state_size = cond_transition_i.shape[-1]
    identity = jnp.eye(state_size, dtype=cond_transition_i.dtype)
    # F_cond,jᵀ
    cond_transition_j_transpose = jnp.swapaxes(cond_transition_j, -1, -2)

    # --- forward map F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ ---
    # I + Σ_cond,i Λ_lik,j
    forward_system = identity + cond_covariance_i @ lik_precision_j
    # (I + Σ_cond,i Λ_lik,j)ᵀ
    forward_system_transpose = jnp.swapaxes(forward_system, -1, -2)
    # (I + Σ_cond,i Λ_lik,j)⁻ᵀ F_cond,jᵀ
    forward_map_transpose = jnp.linalg.solve(
        forward_system_transpose, cond_transition_j_transpose
    )
    # F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹
    forward_map = jnp.swapaxes(forward_map_transpose, -1, -2)

    # F_cond,ij = F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ F_cond,i
    cond_transition = forward_map @ cond_transition_i
    # Σ_cond,i Λμ_lik,j
    information_shift = _bmv(cond_covariance_i, lik_information_j)
    # F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ (μ_cond,i + Σ_cond,i Λμ_lik,j)
    propagated_mean = _bmv(forward_map, cond_mean_i + information_shift)
    # μ_cond,ij = … + μ_cond,j
    cond_mean = propagated_mean + cond_mean_j
    # F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ Σ_cond,i F_cond,jᵀ
    propagated_covariance = (
        forward_map @ cond_covariance_i @ cond_transition_j_transpose
    )
    # Σ_cond,ij = … + Σ_cond,j
    cond_covariance = symmetrize(propagated_covariance + cond_covariance_j)

    # --- backward map F_cond,iᵀ (I + Λ_lik,j Σ_cond,i)⁻¹ ---
    # I + Λ_lik,j Σ_cond,i
    backward_system = identity + lik_precision_j @ cond_covariance_i
    # (I + Λ_lik,j Σ_cond,i)ᵀ
    backward_system_transpose = jnp.swapaxes(backward_system, -1, -2)
    # (I + Λ_lik,j Σ_cond,i)⁻ᵀ F_cond,i
    backward_map_transpose = jnp.linalg.solve(
        backward_system_transpose, cond_transition_i
    )
    # F_cond,iᵀ (I + Λ_lik,j Σ_cond,i)⁻¹
    backward_map = jnp.swapaxes(backward_map_transpose, -1, -2)

    # Λ_lik,j μ_cond,i
    precision_mean = _bmv(lik_precision_j, cond_mean_i)
    # F_cond,iᵀ (I + Λ_lik,j Σ_cond,i)⁻¹ (Λμ_lik,j − Λ_lik,j μ_cond,i)
    propagated_information = _bmv(backward_map, lik_information_j - precision_mean)
    # Λμ_lik,ij = … + Λμ_lik,i
    lik_information = propagated_information + lik_information_i
    # F_cond,iᵀ (I + Λ_lik,j Σ_cond,i)⁻¹ Λ_lik,j F_cond,i
    propagated_precision = backward_map @ lik_precision_j @ cond_transition_i
    # Λ_lik,ij = … + Λ_lik,i
    lik_precision = symmetrize(propagated_precision + lik_precision_i)

    return (
        cond_transition,
        cond_mean,
        cond_covariance,
        lik_information,
        lik_precision,
    )


def _smoother_combine(
    later: SmootherElement,
    earlier: SmootherElement,
) -> SmootherElement:
    """Combine a later smoothing element j with an earlier element i.

        G_smooth,ij = G_smooth,i G_smooth,j
        μ_back,ij = G_smooth,i μ_back,j + μ_back,i
        Σ_back,ij = G_smooth,i Σ_back,j G_smooth,iᵀ + Σ_back,i

    This is Särkkä & García-Fernández (2021, §III.B). It runs inside
    ``lax.associative_scan(..., reverse=True)``, which reverses the
    sequence, runs a forward scan and reverses the result, so the first
    argument is the accumulated later-time chain and the second the next
    earlier-time element. The result spans from the earlier element's time
    to the end of the later chain.

    Args:
        later: Element j, ``(G_smooth, μ_back, Σ_back)``.
        earlier: Element i, in the same layout.

    Returns:
        The combined element ij.
    """
    smoother_gain_j, back_mean_j, back_covariance_j = later
    smoother_gain_i, back_mean_i, back_covariance_i = earlier
    # G_smooth,ij = G_smooth,i G_smooth,j
    smoother_gain = smoother_gain_i @ smoother_gain_j
    # G_smooth,i μ_back,j
    propagated_mean = _bmv(smoother_gain_i, back_mean_j)
    # μ_back,ij = G_smooth,i μ_back,j + μ_back,i
    back_mean = propagated_mean + back_mean_i
    # G_smooth,iᵀ
    smoother_gain_i_transpose = jnp.swapaxes(smoother_gain_i, -1, -2)
    # G_smooth,i Σ_back,j G_smooth,iᵀ
    propagated_covariance = (
        smoother_gain_i @ back_covariance_j @ smoother_gain_i_transpose
    )
    # Σ_back,ij = G_smooth,i Σ_back,j G_smooth,iᵀ + Σ_back,i
    back_covariance = symmetrize(propagated_covariance + back_covariance_i)
    return smoother_gain, back_mean, back_covariance


def _smoother_element(
    filtered_mean: Float[Array, " N"],
    filtered_covariance: Float[Array, "N N"],
    predicted_mean_next: Float[Array, " N"],
    predicted_covariance_next: Float[Array, "N N"],
    transition_next: Float[Array, "N N"],
) -> SmootherElement:
    """Smoothing element of step t < T: the backward conditional of x_t.

        G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹
        μ_back,t = μ_x,t|t − G_smooth,t μ_x,t+1|t
        Σ_back,t = Σ_x,t|t − G_smooth,t Σ_x,t+1|t G_smooth,tᵀ

    Args:
        filtered_mean: Filtered mean μ_x,t|t.
        filtered_covariance: Filtered covariance Σ_x,t|t.
        predicted_mean_next: Predicted mean μ_x,t+1|t.
        predicted_covariance_next: Predicted covariance Σ_x,t+1|t.
        transition_next: Dynamics F_t+1.

    Returns:
        The element ``(G_smooth, μ_back, Σ_back)`` of step t.
    """
    # Σ_x,t|t F_t+1ᵀ
    cross_covariance = einsum(
        filtered_covariance, transition_next, "state col, next col -> state next"
    )
    # F_t+1 Σ_x,t|t
    cross_covariance_transpose = rearrange(cross_covariance, "state next -> next state")
    # G_smooth,tᵀ = Σ_x,t+1|t⁻¹ F_t+1 Σ_x,t|t, using the symmetry of Σ_x,t+1|t
    smoother_gain_transpose = jnp.linalg.solve(
        predicted_covariance_next, cross_covariance_transpose
    )
    # G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹
    smoother_gain = rearrange(smoother_gain_transpose, "next state -> state next")
    # G_smooth,t μ_x,t+1|t
    predicted_shift = smoother_gain @ predicted_mean_next
    # μ_back,t = μ_x,t|t − G_smooth,t μ_x,t+1|t
    back_mean = filtered_mean - predicted_shift
    # G_smooth,t Σ_x,t+1|t
    gain_covariance = einsum(
        smoother_gain, predicted_covariance_next, "state next, next col -> state col"
    )
    # G_smooth,t Σ_x,t+1|t G_smooth,tᵀ
    correction = einsum(
        gain_covariance, smoother_gain, "state col, other col -> state other"
    )
    # Σ_back,t = Σ_x,t|t − G_smooth,t Σ_x,t+1|t G_smooth,tᵀ
    back_covariance = symmetrize(filtered_covariance - correction)
    return smoother_gain, back_mean, back_covariance
