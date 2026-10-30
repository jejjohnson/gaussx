"""PSD-projected parallel Kalman filter and RTS smoother (``psd_project=True``).

Not a square-root filter: the filter's associative scan runs the
covariance-form combinator of `gaussx._ssm._parallel_kalman` and projects
its outputs onto the PSD cone (gh-306). Each filtering element carries,
besides (F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik), the lower-triangular
factors L_cond and L_lik of the PSD projections of Σ_cond and Λ_lik; the
smoothing elements carry L_back in place of Σ_back. The returned
covariances take their values from the factors and their gradients from
the covariance path. The factor-propagating combinator is
`gaussx._ssm._parallel_kalman_factor` (``square_root=True``, gh-454).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import einsum, rearrange
from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._logdet import cholesky_logdet
from gaussx._ssm._kalman import FilterState, _empty_filter_state, _kalman_predict
from gaussx._ssm._parallel_kalman import (
    FilterElement,
    _bmv,
    _filter_combine,
    _first_filter_element_active,
    _first_filter_element_masked,
    _generic_filter_element_active,
)
from gaussx._ssm._utils import (
    _normalise_tv_inputs,
    _smoother_element,
    _transition_stack,
)
from gaussx._strategies._base import AbstractSolverStrategy


FactorFilterElement = tuple[
    Float[Array, "*T N N"],
    Float[Array, "*T N"],
    Float[Array, "*T N N"],
    Float[Array, "*T N N"],
    Float[Array, "*T N"],
    Float[Array, "*T N N"],
    Float[Array, "*T N N"],
]
FactorSmootherElement = tuple[
    Float[Array, "*T N N"], Float[Array, "*T N"], Float[Array, "*T N N"]
]


# ----------------------------------------------------------------
# Factors
# ----------------------------------------------------------------


def tria(M: Float[Array, "... K N"]) -> Float[Array, "... N N"]:
    """Lower-triangular factor L with L Lᵀ = Mᵀ M, from a QR of M.

        M = Q R,   L = Rᵀ,   diag(R) ≥ 0

    Works on the trailing two axes of a batch, which may be empty, so it
    uses ``swapaxes`` rather than einx.

    Args:
        M: Stacked factor, one row per term of Mᵀ M.

    Returns:
        The lower-triangular L, unique when M has full column rank.

    Examples:
        The factor of a stack reproduces its Gram matrix:

        >>> import jax.numpy as jnp
        >>> stack = jnp.array([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]])
        >>> factor = tria(stack)
        >>> factor.shape
        (2, 2)
        >>> bool(jnp.allclose(factor @ factor.T, stack.T @ stack))
        True
    """
    _, upper = jnp.linalg.qr(M, mode="reduced")
    diagonal = jnp.diagonal(upper, axis1=-2, axis2=-1)
    # Flip rows so diag(R) ≥ 0, which makes L unique
    sign = jnp.where(diagonal < 0, -1.0, 1.0).astype(upper.dtype)
    upper = sign[..., :, None] * upper
    # L = Rᵀ
    return jnp.tril(jnp.swapaxes(upper, -1, -2))


def _factor_from_psd(X: Float[Array, "... N N"]) -> Float[Array, "... N N"]:
    """Lower-triangular factor of the PSD projection of a symmetric matrix.

        X = V diag(σ²) Vᵀ,   L Lᵀ = V diag(max(σ², 0)) Vᵀ

    Args:
        X: Symmetric matrix, or a batch of them (possibly empty).

    Returns:
        The lower-triangular L of the eigenvalue-clipped X.
    """
    X = symmetrize(X)
    # X = V diag(σ²) Vᵀ
    variances, eigenvectors = jnp.linalg.eigh(X)
    # max(σ², 0)
    variances = jnp.clip(variances, min=0.0)
    # V diag(σ)
    root = eigenvectors * jnp.sqrt(variances)[..., None, :]
    # L = tria((V diag(σ))ᵀ)
    return tria(jnp.swapaxes(root, -1, -2))


def _cov_from_factor(factor: Float[Array, "... N N"]) -> Float[Array, "... N N"]:
    """Gram matrix L Lᵀ of a factor, symmetrised.

    Args:
        factor: Factor L, or a batch of them.

    Returns:
        L Lᵀ.
    """
    # L Lᵀ
    gram = factor @ jnp.swapaxes(factor, -1, -2)
    return symmetrize(gram)


def _factor_filter_element(element: FilterElement) -> FactorFilterElement:
    """Add the factors L_cond and L_lik to a covariance-form element.

    Args:
        element: ``(F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik)``.

    Returns:
        ``(F_cond, μ_cond, Σ_cond, L_cond, Λμ_lik, Λ_lik, L_lik)`` with
        L_cond L_condᵀ and L_lik L_likᵀ the PSD projections of Σ_cond and
        Λ_lik.
    """
    (
        cond_transition,
        cond_mean,
        cond_covariance,
        lik_information,
        lik_precision,
    ) = element
    return (
        cond_transition,
        cond_mean,
        cond_covariance,
        _factor_from_psd(cond_covariance),
        lik_information,
        lik_precision,
        _factor_from_psd(lik_precision),
    )


def _filter_sqrt_combine(
    earlier: FactorFilterElement,
    later: FactorFilterElement,
) -> FactorFilterElement:
    """Combine filtering elements i and j, then refactor Σ_cond and Λ_lik.

    The covariance-form rule `_filter_combine` gives the combined element;
    L_cond,ij and L_lik,ij are the factors of its PSD projections. The
    incoming factors are carried only to keep the element layout.

    Args:
        earlier: Element i, ``(F_cond, μ_cond, Σ_cond, L_cond, Λμ_lik,
            Λ_lik, L_lik)``.
        later: Element j, in the same layout.

    Returns:
        The combined element ij.
    """
    cond_transition_i, cond_mean_i, cond_covariance_i, _, info_i, precision_i, _ = (
        earlier
    )
    cond_transition_j, cond_mean_j, cond_covariance_j, _, info_j, precision_j, _ = later
    combined = _filter_combine(
        (cond_transition_i, cond_mean_i, cond_covariance_i, info_i, precision_i),
        (cond_transition_j, cond_mean_j, cond_covariance_j, info_j, precision_j),
    )
    return _factor_filter_element(combined)


def _gated_filter_element(
    transition_t: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    process_noise_t: Float[Array, "N N"],
    obs_noise_covariance: Float[Array, "M M"],
    obs_t: Float[Array, " M"],
    mask_t: Bool[Array, ""],
) -> FactorFilterElement:
    """Factored filtering element of step t ≥ 2 under a per-step gate.

    A gated-off step zeroes H_t and y_t and puts the identity in Σ_ε,t,
    which collapses the element to the predict-only one without branching.

    Args:
        transition_t: Dynamics F_t.
        obs_operator: Observation operator H_t.
        process_noise_t: Model-error covariance Σ_η,t.
        obs_noise_covariance: Observation-error covariance Σ_ε,t.
        obs_t: Observation y_t.
        mask_t: Whether step t is observed.

    Returns:
        The element ``(F_cond, μ_cond, Σ_cond, L_cond, Λμ_lik, Λ_lik,
        L_lik)`` of step t.
    """
    obs_size = obs_t.shape[-1]
    gated_obs_operator = jnp.where(mask_t, obs_operator, jnp.zeros_like(obs_operator))
    unit_noise = jnp.eye(obs_size, dtype=obs_noise_covariance.dtype)
    gated_obs_noise = jnp.where(mask_t, obs_noise_covariance, unit_noise)
    gated_obs = jnp.where(mask_t, obs_t, jnp.zeros_like(obs_t))
    element = _generic_filter_element_active(
        transition_t, gated_obs_operator, process_noise_t, gated_obs_noise, gated_obs
    )
    return _factor_filter_element(element)


def _projected_predict(
    transition_t: Float[Array, "N N"],
    filtered_mean: Float[Array, " N"],
    filtered_covariance: Float[Array, "N N"],
    filtered_factor: Float[Array, "N N"],
    process_noise_factor: Float[Array, "N N"],
    process_noise_t: Float[Array, "N N"],
) -> tuple[Float[Array, " N"], Float[Array, "N N"]]:
    """Predict step whose covariance takes its value from the factor path.

        μ_x,t|t−1 = F_t μ_x,t−1|t−1
        L_x,t|t−1 = tria([L_x,t−1|t−1ᵀ F_tᵀ; L_ηᵀ])
        Σ_x,t|t−1 = L_x,t|t−1 L_x,t|t−1ᵀ

    The value is the factor form, PSD by construction; the gradient is that
    of the covariance form F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t.

    Args:
        transition_t: Dynamics F_t.
        filtered_mean: Filtered mean μ_x,t−1|t−1.
        filtered_covariance: Filtered covariance Σ_x,t−1|t−1.
        filtered_factor: Factor L_x,t−1|t−1 of the filtered covariance.
        process_noise_factor: Factor L_η of Σ_η,t.
        process_noise_t: Model-error covariance Σ_η,t.

    Returns:
        Tuple ``(predicted_mean, predicted_covariance)`` holding
        μ_x,t|t−1 and Σ_x,t|t−1.
    """
    # μ_x,t|t−1 and the covariance form F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t
    predicted_mean, predicted_covariance = _kalman_predict(
        filtered_mean,
        filtered_covariance,
        transition_t,
        process_noise_t,
        transition_op=None,
    )
    # L_x,t−1|t−1ᵀ F_tᵀ
    propagated_factor = einsum(
        filtered_factor, transition_t, "state col, next state -> col next"
    )
    # L_ηᵀ
    process_noise_factor_transpose = rearrange(
        process_noise_factor, "row col -> col row"
    )
    # [L_x,t−1|t−1ᵀ F_tᵀ; L_ηᵀ]
    stack = jnp.concatenate([propagated_factor, process_noise_factor_transpose])
    # L_x,t|t−1 = tria([L_x,t−1|t−1ᵀ F_tᵀ; L_ηᵀ])
    predicted_factor = tria(stack)
    # Σ_x,t|t−1 = L_x,t|t−1 L_x,t|t−1ᵀ
    projected_covariance = _cov_from_factor(predicted_factor)
    # Value of the factor path, gradient of the covariance path
    projection_shift = jax.lax.stop_gradient(
        projected_covariance - predicted_covariance
    )
    return predicted_mean, predicted_covariance + projection_shift


def _gated_log_likelihood_step(
    obs_t: Float[Array, " M"],
    predicted_mean: Float[Array, " N"],
    predicted_covariance: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    obs_noise_covariance: Float[Array, "M M"],
    mask_t: Bool[Array, ""],
) -> Float[Array, ""]:
    """Log-likelihood term of one step under a per-step gate.

        log p(y_t ∣ y_1:t−1) = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t + D_y log 2π)

    Args:
        obs_t: Observation y_t.
        predicted_mean: Predicted mean μ_x,t|t−1.
        predicted_covariance: Predicted covariance Σ_x,t|t−1.
        obs_operator: Observation operator H_t.
        obs_noise_covariance: Observation-error covariance Σ_ε,t.
        mask_t: Whether step t is observed; a gated-off step contributes 0.

    Returns:
        log p(y_t ∣ y_1:t−1), or 0 for a gated-off step.
    """
    obs_size = obs_t.shape[-1]
    gated_obs_operator = jnp.where(mask_t, obs_operator, jnp.zeros_like(obs_operator))
    unit_noise = jnp.eye(obs_size, dtype=obs_noise_covariance.dtype)
    gated_obs_noise = jnp.where(mask_t, obs_noise_covariance, unit_noise)
    gated_obs = jnp.where(mask_t, obs_t, jnp.zeros_like(obs_t))
    # H_t μ_x,t|t−1
    predicted_obs = gated_obs_operator @ predicted_mean
    # r_t = y_t − H_t μ_x,t|t−1
    residual = gated_obs - predicted_obs
    # H_t Σ_x,t|t−1
    obs_projection = einsum(
        gated_obs_operator, predicted_covariance, "obs state, state col -> obs col"
    )
    # H_t Σ_x,t|t−1 H_tᵀ
    obs_sandwich = einsum(
        obs_projection, gated_obs_operator, "obs col, other col -> obs other"
    )
    # Σ_y,t = H_t Σ_x,t|t−1 H_tᵀ + Σ_ε,t
    obs_covariance = symmetrize(obs_sandwich + gated_obs_noise)
    # L_y,t L_y,tᵀ = Σ_y,t
    obs_factor = jnp.linalg.cholesky(obs_covariance)
    # Σ_y,t⁻¹ r_t
    whitened_residual = jax.scipy.linalg.cho_solve((obs_factor, True), residual)
    # r_tᵀ Σ_y,t⁻¹ r_t
    mahalanobis = residual @ whitened_residual
    # log det Σ_y,t
    logdet = cholesky_logdet(obs_factor)
    # log p(y_t ∣ y_1:t−1) = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t + D_y log 2π)
    log_likelihood_step = -0.5 * (mahalanobis + logdet + obs_size * _LOG_2PI)
    return jnp.where(mask_t, log_likelihood_step, jnp.zeros_like(log_likelihood_step))


# ----------------------------------------------------------------
# Filter
# ----------------------------------------------------------------


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
    """PSD-projected parallel Kalman filter via associative scan.

        Σ_x,t|t = L_cond,t L_cond,tᵀ,   L_cond,t L_cond,tᵀ = Π_PSD(Σ_cond,t)

    with Π_PSD the eigenvalue clip onto the PSD cone and Σ_cond,t the
    covariance-form scan output. This is ``parallel_kalman_filter(...,
    psd_project=True)``; values are PSD, gradients are those of the
    unprojected covariance form.

    Algorithm:
        1. Build the covariance-form filtering elements of every step, the
           first absorbing the prior, and factor each Σ_cond and Λ_lik.
        2. Run the associative scan, refactoring after every combination.
        3. Predict from the filtered factors, L_x,t|t−1 =
           tria([L_x,t−1|t−1ᵀ F_tᵀ; L_ηᵀ]), and score each observed step.

    Args:
        transition: Linear dynamics F_t, one or a stack over time.
        obs_model: Observation operator H_t, one or a stack over time.
        process_noise: Model-error covariance Σ_η,t, one or a stack.
        obs_noise: Observation-error covariance Σ_ε,t, one or a stack.
        observations: Observations y_1:T, one row per step.
        init_mean: Prior mean μ_x,0 of the initial state x₀.
        init_cov: Prior covariance Σ_x,0 of x₀.
        mask: Optional per-step mask, shaped like ``observations[:, 0]``.
            A per-channel ``(T, M)`` mask makes Σ_y,t block-degenerate,
            which this path does not handle; use the covariance form or
            `gaussx.kalman_filter` for per-channel masking.
        solver: Accepted for API symmetry; not threaded through.

    Raises:
        NotImplementedError: If ``mask`` is a ``(T, M)`` per-channel mask.

    Returns:
        A `FilterState` with the filtered and predicted means and
        PSD-projected covariances and the total log-likelihood.

    Examples:
        Covariances are PSD and the log-likelihood matches the covariance
        form:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> model = (jnp.eye(2), jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> data = (jnp.array([[0.5]]), observations, jnp.zeros(2), jnp.eye(2))
        >>> state = parallel_kalman_filter_sqrt(*model, *data)
        >>> state.filtered_covs.shape
        (3, 2, 2)
        >>> bool(jnp.all(jnp.linalg.eigvalsh(state.filtered_covs) >= -1e-6))
        True
        >>> reference = gaussx.parallel_kalman_filter(*model, *data)
        >>> bool(jnp.allclose(state.log_likelihood, reference.log_likelihood))
        True
    """
    del solver

    if mask is not None and jnp.ndim(mask) == 2:
        raise NotImplementedError(
            "psd_project=True does not support per-channel (T, M) masks; pass a "
            "(T,) per-step mask, or use psd_project=False / kalman_filter."
        )

    num_steps = observations.shape[0]

    if num_steps == 0:
        return _empty_filter_state(init_mean, init_cov)

    transitions, obs_operators, process_noises, obs_noises, masks, _ = (
        _normalise_tv_inputs(
            transition, obs_model, process_noise, obs_noise, T=num_steps, mask=mask
        )
    )

    # --- elements ---
    elements = jax.vmap(_gated_filter_element)(
        transitions, obs_operators, process_noises, obs_noises, observations, masks
    )
    first = jax.lax.cond(
        masks[0],
        lambda: _factor_filter_element(
            _first_filter_element_active(
                transitions[0],
                obs_operators[0],
                process_noises[0],
                obs_noises[0],
                observations[0],
                init_mean,
                init_cov,
            )
        ),
        lambda: _factor_filter_element(
            _first_filter_element_masked(
                transitions[0],
                process_noises[0],
                init_mean,
                init_cov,
            )
        ),
    )
    elements = tuple(
        stack.at[0].set(value) for stack, value in zip(elements, first, strict=True)
    )

    # --- scan ---
    # (F_cond, μ_x,t|t, Σ_cond,t, L_cond,t, Λμ_lik, Λ_lik, L_lik) for every t
    _, filtered_means, scanned_covs, filtered_factors, _, _, _ = (
        jax.lax.associative_scan(_filter_sqrt_combine, elements)
    )
    # L_cond,t L_cond,tᵀ
    projected_covs = _cov_from_factor(filtered_factors)
    # Return PSD covariances while keeping gradients through the scan.
    projection_shift = jax.lax.stop_gradient(projected_covs - scanned_covs)
    filtered_covs = scanned_covs + projection_shift

    # --- predict and score ---
    # μ_x,0:T−1|0:T−1, the prior first
    previous_means = jnp.concatenate(
        [rearrange(init_mean, "state -> 1 state"), filtered_means[:-1]]
    )
    # L_x,0 of the projected prior covariance
    init_factor = _factor_from_psd(init_cov)
    # L_x,0:T−1|0:T−1
    previous_factors = jnp.concatenate(
        [rearrange(init_factor, "row col -> 1 row col"), filtered_factors[:-1]]
    )
    # Σ_x,0:T−1|0:T−1 from the unprojected scan
    previous_covs = jnp.concatenate(
        [rearrange(init_cov, "row col -> 1 row col"), scanned_covs[:-1]]
    )
    # L_η,t
    process_noise_factors = jax.vmap(_factor_from_psd)(process_noises)

    predicted_means, predicted_covs = jax.vmap(_projected_predict)(
        transitions,
        previous_means,
        previous_covs,
        previous_factors,
        process_noise_factors,
        process_noises,
    )

    log_likelihood_steps = jax.vmap(_gated_log_likelihood_step)(
        observations, predicted_means, predicted_covs, obs_operators, obs_noises, masks
    )

    return FilterState(
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        # log p(y_1:T) = ∑_t log p(y_t ∣ y_1:t−1)
        log_likelihood=jnp.sum(log_likelihood_steps),
    )


# ----------------------------------------------------------------
# Smoother
# ----------------------------------------------------------------


def _smoother_sqrt_combine(
    later: FactorSmootherElement,
    earlier: FactorSmootherElement,
) -> FactorSmootherElement:
    """Combine a later smoothing element j with an earlier element i, factored.

        G_smooth,ij = G_smooth,i G_smooth,j
        μ_back,ij = G_smooth,i μ_back,j + μ_back,i
        L_back,ij = tria([L_back,jᵀ G_smooth,iᵀ; L_back,iᵀ])

    The argument order follows `_smoother_combine` under a reverse scan.
    Works on the trailing two axes of a possibly empty batch.

    Args:
        later: Element j, ``(G_smooth, μ_back, L_back)``.
        earlier: Element i, in the same layout.

    Returns:
        The combined element ij.
    """
    smoother_gain_j, back_mean_j, back_factor_j = later
    smoother_gain_i, back_mean_i, back_factor_i = earlier
    # G_smooth,ij = G_smooth,i G_smooth,j
    smoother_gain = smoother_gain_i @ smoother_gain_j
    # G_smooth,i μ_back,j
    propagated_mean = _bmv(smoother_gain_i, back_mean_j)
    # μ_back,ij = G_smooth,i μ_back,j + μ_back,i
    back_mean = propagated_mean + back_mean_i
    # L_back,jᵀ
    back_factor_j_transpose = jnp.swapaxes(back_factor_j, -1, -2)
    # G_smooth,iᵀ
    smoother_gain_i_transpose = jnp.swapaxes(smoother_gain_i, -1, -2)
    # L_back,jᵀ G_smooth,iᵀ
    propagated_factor = back_factor_j_transpose @ smoother_gain_i_transpose
    # L_back,iᵀ
    back_factor_i_transpose = jnp.swapaxes(back_factor_i, -1, -2)
    # [L_back,jᵀ G_smooth,iᵀ; L_back,iᵀ]
    stack = jnp.concatenate([propagated_factor, back_factor_i_transpose], axis=-2)
    # L_back,ij = tria([L_back,jᵀ G_smooth,iᵀ; L_back,iᵀ])
    back_factor = tria(stack)
    return smoother_gain, back_mean, back_factor


def parallel_rts_smoother_sqrt(
    filter_state: FilterState,
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """PSD-projected parallel RTS smoother via reverse associative scan.

        Σ_x,t|T = L_back,t L_back,tᵀ

    This is ``parallel_rts_smoother(..., psd_project=True)``. The smoothing
    elements come from covariance-form quantities and are factored after a
    PSD projection; the scan combines the factors with QR.

    Algorithm:
        1. Build the smoothing elements (G_smooth,t, μ_back,t, Σ_back,t) of
           every step t < T and factor each Σ_back,t; the last element is
           (0, μ_x,T|T, L_x,T|T).
        2. Run the reverse associative scan on (G_smooth, μ_back, L_back).
        3. Return μ_x,t|T and Σ_x,t|T = L_back,t L_back,tᵀ.

    Args:
        filter_state: Output of `parallel_kalman_filter` or
            `gaussx.kalman_filter`.
        transition: Linear dynamics F_t, as a matrix, a stack over time or
            an operator; the same one passed to the filter.
        solver: Accepted for API symmetry; not threaded through.

    Raises:
        TypeError: If an operator-typed ``transition`` has a time axis.
        ValueError: If ``transition`` is neither a matrix nor a stack.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` holding μ_x,t|T and
        Σ_x,t|T.

    Examples:
        Smoothed covariances are PSD:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transition = jnp.eye(2)
        >>> model = (transition, jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> state = gaussx.parallel_kalman_filter(
        ...     *model, jnp.array([[0.5]]), observations, jnp.zeros(2), jnp.eye(2)
        ... )
        >>> means, covs = parallel_rts_smoother_sqrt(state, transition)
        >>> means.shape, covs.shape
        ((3, 2), (3, 2, 2))
        >>> bool(jnp.all(jnp.linalg.eigvalsh(covs) >= -1e-6))
        True
    """
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
    # L_back,t of the projected Σ_back,t
    inner_factors = jax.vmap(_factor_from_psd)(inner_covs)
    # Last element: G_smooth,T = 0, μ_back,T = μ_x,T|T, L_back,T = L_x,T|T
    last_gain = jnp.zeros((1, state_size, state_size), dtype=filtered_means.dtype)
    last_factor = _factor_from_psd(filtered_covs[-1:])

    smoother_gains = jnp.concatenate([inner_gains, last_gain])
    back_means = jnp.concatenate([inner_means, filtered_means[-1:]])
    back_factors = jnp.concatenate([inner_factors, last_factor])

    # (G_smooth,t:T, μ_x,t|T, L_back,t:T) for every t
    _, smoothed_means, smoothed_factors = jax.lax.associative_scan(
        _smoother_sqrt_combine,
        (smoother_gains, back_means, back_factors),
        reverse=True,
    )
    # Σ_x,t|T = L_back,t L_back,tᵀ
    return smoothed_means, _cov_from_factor(smoothed_factors)
