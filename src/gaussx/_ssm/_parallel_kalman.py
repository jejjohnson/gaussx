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
from gaussx._ssm._utils import (
    SmootherElement,
    _masked_obs_inputs,
    _normalise_tv_inputs,
    _smoother_element,
    _transition_stack,
    _warn_unused_process_noise,
)
from gaussx._strategies._base import AbstractSolverStrategy


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
