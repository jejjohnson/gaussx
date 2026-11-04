"""Smoothed marginals of a state-space model at steps with no observation.

Three consecutive steps t − 1, t and t + 1 of a linear-Gaussian model, with
no observation at t, are joined by the dynamics

    x_t = F_t x_t−1 + η_t,        η_t ~ 𝒩(0, Σ_η,t)
    x_t+1 = F_t+1 x_t + η_t+1,    η_t+1 ~ 𝒩(0, Σ_η,t+1)

with a state x_t of length D_x, in the notation of `gaussx.kalman_filter`
and `gaussx.rts_smoother`. Both functions return the smoothed marginal
p(x_t ∣ y_1:T) = 𝒩(μ_x,t|T, Σ_x,t|T) of the unobserved step.
"""

from __future__ import annotations

import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._linalg._linalg import solve_columns, solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._inv import inv
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_solve


def conditional_interpolate(
    A_fwd: Float[Array, "d d"],
    Q_fwd: Float[Array, "d d"],
    A_bwd: Float[Array, "d d"],
    Q_bwd: Float[Array, "d d"],
    mu_prev: Float[Array, " d"],
    P_prev: Float[Array, "d d"],
    mu_next: Float[Array, " d"],
    P_next: Float[Array, "d d"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
    """Two-filter fusion at t from a filtered state and a backward message.

        Λ_x,t|T = Λ_x,t|t−1 + Λ_back,t,   Σ_x,t|T = Λ_x,t|T⁻¹
        μ_x,t|T = Σ_x,t|T (Λ_x,t|t−1 μ_x,t|t−1 + Λμ_back,t)

    Combines two **conditionally independent** sources of information about
    x_t by adding precisions (the two-filter smoother):

    - the **forward-filtered** state at t − 1,
      p(x_t−1 ∣ y_1:t−1) = 𝒩(μ_x,t−1|t−1, Σ_x,t−1|t−1), predicted to t as
      μ_x,t|t−1 = F_t μ_x,t−1|t−1 and
      Σ_x,t|t−1 = F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t, with precision
      Λ_x,t|t−1 = Σ_x,t|t−1⁻¹;
    - the **backward-information message** at t + 1, the likelihood of the
      data at and after t + 1,
      p(y_t+1:T ∣ x_t+1) ∝ 𝒩(μ_back,t+1; x_t+1, Σ_back,t+1) (no prior on
      x_t+1), carried back to t as
      Λ_back,t = F_t+1ᵀ (Σ_back,t+1 + Σ_η,t+1)⁻¹ F_t+1 and
      Λμ_back,t = F_t+1ᵀ (Σ_back,t+1 + Σ_η,t+1)⁻¹ μ_back,t+1, written as
      one symbol because Λ_back,t may be singular.

    With those inputs it returns p(x_t ∣ y_1:t−1, y_t+1:T) exactly.

    Algorithm:
        1. Predict the filtered state to t and invert Σ_x,t|t−1 into
           Λ_x,t|t−1, with the information vector Λ_x,t|t−1 μ_x,t|t−1.
        2. Carry the backward message from t + 1 to t: Λ_back,t and
           Λμ_back,t.
        3. Add the precisions and the information vectors, then invert
           Λ_x,t|T into Σ_x,t|T and form μ_x,t|T.

    Warning:
        Do **not** pass smoothed marginals (e.g. from `rts_smoother`) for
        both ends: each already contains all the data, so the data is
        counted twice and the result is over-confident (about 20% too small
        a covariance on a 3-state chain, gh-288). To interpolate from a
        filtered state at t − 1 and a smoothed state at t + 1, use
        `rts_interpolate`.

    Args:
        A_fwd: Transition F_t from step t − 1 to t.
        Q_fwd: Model-error covariance Σ_η,t of that transition.
        A_bwd: Transition F_t+1 from step t to t + 1.
        Q_bwd: Model-error covariance Σ_η,t+1 of that transition.
        mu_prev: Filtered mean μ_x,t−1|t−1.
        P_prev: Filtered covariance Σ_x,t−1|t−1.
        mu_next: Mean μ_back,t+1 of the backward-information message.
        P_next: Covariance Σ_back,t+1 of the backward-information message.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(mean, cov)`` holding μ_x,t|T and Σ_x,t|T, the moments of
        x_t given the data on both sides.

    Examples:
        A scalar random walk, with a message 𝒩(2, 1) one step ahead:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> one = jnp.ones((1, 1))
        >>> mean, cov = gaussx.conditional_interpolate(
        ...     one, one, one, one, jnp.zeros(1), one, jnp.array([2.0]), one
        ... )
        >>> bool(jnp.allclose(mean, 1.0)), bool(jnp.allclose(cov, 1.0))
        (True, True)

        An uninformative message leaves the forward prediction:

        >>> mean, cov = gaussx.conditional_interpolate(
        ...     one, one, one, one, jnp.zeros(1), one, jnp.zeros(1), 1e6 * one
        ... )
        >>> bool(jnp.allclose(cov, 2.0, rtol=1e-4))
        True

        An empty state (D_x = 0) gives empty moments:

        >>> empty = jnp.zeros((0, 0))
        >>> mean, cov = gaussx.conditional_interpolate(
        ...     empty, empty, empty, empty, jnp.zeros(0), empty, jnp.zeros(0), empty
        ... )
        >>> mean.shape, cov.shape
        ((0,), (0, 0))
    """
    # jnp, not einx, throughout: the state may be empty (D_x = 0), which
    # einx rejects and these products accept.

    # --- forward prediction ---
    # μ_x,t|t−1 = F_t μ_x,t−1|t−1
    predicted_mean = A_fwd @ mu_prev
    # F_t Σ_x,t−1|t−1
    transition_covariance = A_fwd @ P_prev
    # F_t Σ_x,t−1|t−1 F_tᵀ
    propagated = transition_covariance @ A_fwd.T
    # Σ_x,t|t−1 = F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t
    predicted_covariance = propagated + Q_fwd

    # --- forward information ---
    predicted_covariance_op = lx.MatrixLinearOperator(
        predicted_covariance, lx.positive_semidefinite_tag
    )
    # Λ_x,t|t−1 = Σ_x,t|t−1⁻¹
    predicted_precision = inv(predicted_covariance_op).as_matrix()
    # Λ_x,t|t−1 μ_x,t|t−1 = Σ_x,t|t−1⁻¹ μ_x,t|t−1
    predicted_information = dispatch_solve(
        predicted_covariance_op, predicted_mean, solver
    )

    # --- backward information from t + 1 ---
    # Σ_back,t+1 + Σ_η,t+1
    message_covariance = P_next + Q_bwd
    message_covariance_op = lx.MatrixLinearOperator(
        message_covariance, lx.positive_semidefinite_tag
    )
    # F_t+1ᵀ, taken before the solve so the traced order (and with it the
    # order in which gradients accumulate) is unchanged
    transition_t = A_bwd.T
    # (Σ_back,t+1 + Σ_η,t+1)⁻¹ F_t+1
    solved_transition = solve_columns(message_covariance_op, A_bwd, solver=solver)
    # Λ_back,t = F_t+1ᵀ (Σ_back,t+1 + Σ_η,t+1)⁻¹ F_t+1
    backward_precision = transition_t @ solved_transition
    # F_t+1ᵀ, again taken before its solve
    message_transition_t = A_bwd.T
    # (Σ_back,t+1 + Σ_η,t+1)⁻¹ μ_back,t+1
    solved_message = dispatch_solve(message_covariance_op, mu_next, solver)
    # Λμ_back,t = F_t+1ᵀ (Σ_back,t+1 + Σ_η,t+1)⁻¹ μ_back,t+1
    backward_information = message_transition_t @ solved_message

    # --- fuse forward and backward ---
    # Λ_x,t|T = Λ_x,t|t−1 + Λ_back,t
    smoothed_precision = predicted_precision + backward_precision
    smoothed_precision_op = lx.MatrixLinearOperator(
        smoothed_precision, lx.positive_semidefinite_tag
    )
    # Σ_x,t|T = Λ_x,t|T⁻¹
    smoothed_covariance = inv(smoothed_precision_op).as_matrix()
    # Λ_x,t|t−1 μ_x,t|t−1 + Λμ_back,t
    total_information = predicted_information + backward_information
    # μ_x,t|T = Σ_x,t|T (Λ_x,t|t−1 μ_x,t|t−1 + Λμ_back,t)
    smoothed_mean = smoothed_covariance @ total_information

    return smoothed_mean, smoothed_covariance


def rts_interpolate(
    A_fwd: Float[Array, "d d"],
    Q_fwd: Float[Array, "d d"],
    A_bwd: Float[Array, "d d"],
    Q_bwd: Float[Array, "d d"],
    filtered_mean_prev: Float[Array, " d"],
    filtered_cov_prev: Float[Array, "d d"],
    smoothed_mean_next: Float[Array, " d"],
    smoothed_cov_next: Float[Array, "d d"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
    """Smoothed marginal at an unobserved step t from filtered and smoothed states.

        μ_x,t|T = μ_x,t|t + G_smooth,t (μ_x,t+1|T − μ_x,t+1|t)
        Σ_x,t|T = Σ_x,t|t + G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ

    with the smoother gain G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹. For the
    model of `conditional_interpolate` (no observation at t), it takes the
    **forward-filtered** state at t − 1 and the **smoothed** state at
    t + 1, which is what `kalman_filter` and `rts_smoother` return, and
    runs one Rauch-Tung-Striebel backward step through t. With no
    observation at t the filtered moments there are the predicted ones,
    μ_x,t|t = μ_x,t|t−1 and Σ_x,t|t = Σ_x,t|t−1.

    This is exact: given x_t+1, x_t is independent of the data at and after
    t + 1, so p(x_t ∣ y_1:T) = ∫ p(x_t ∣ x_t+1, y_1:t−1) p(x_t+1 ∣ y_1:T)
    dx_t+1.

    Algorithm:
        1. Predict to t: μ_x,t|t−1 = F_t μ_x,t−1|t−1 and
           Σ_x,t|t−1 = F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t, which are also the
           filtered moments at t.
        2. Predict to t + 1: μ_x,t+1|t = F_t+1 μ_x,t|t and
           Σ_x,t+1|t = F_t+1 Σ_x,t|t F_t+1ᵀ + Σ_η,t+1.
        3. Solve for the smoother gain G_smooth,t against Σ_x,t+1|t.
        4. Correct the moments at t with the smoothed and predicted moments
           at t + 1, and symmetrise the covariance.

    Args:
        A_fwd: Transition F_t from step t − 1 to t.
        Q_fwd: Model-error covariance Σ_η,t of that transition.
        A_bwd: Transition F_t+1 from step t to t + 1.
        Q_bwd: Model-error covariance Σ_η,t+1 of that transition.
        filtered_mean_prev: Filtered mean μ_x,t−1|t−1.
        filtered_cov_prev: Filtered covariance Σ_x,t−1|t−1.
        smoothed_mean_next: Smoothed mean μ_x,t+1|T.
        smoothed_cov_next: Smoothed covariance Σ_x,t+1|T.
        solver: Optional solver strategy for the solve against Σ_x,t+1|t.

    Returns:
        Tuple ``(mean, cov)`` holding μ_x,t|T and Σ_x,t|T, the smoothed
        marginal of x_t.

    Examples:
        A scalar random walk, smoothed to 𝒩(2, ½) one step ahead:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> one = jnp.ones((1, 1))
        >>> mean, cov = gaussx.rts_interpolate(
        ...     one, one, one, one, jnp.zeros(1), one, jnp.array([2.0]), 0.5 * one
        ... )
        >>> round(float(mean[0]), 4), round(float(cov[0, 0]), 4)
        (1.3333, 0.8889)

        A step masked off in `kalman_filter` gets its `rts_smoother` value:

        >>> model = (jnp.eye(2), jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> observations = jnp.array([[0.1], [0.0], [0.2]])
        >>> state = gaussx.kalman_filter(
        ...     *model, jnp.array([[0.5]]), observations, jnp.zeros(2),
        ...     jnp.eye(2), mask=jnp.array([True, False, True]),
        ... )
        >>> means, covs = gaussx.rts_smoother(state, jnp.eye(2))
        >>> dynamics = (jnp.eye(2), 0.1 * jnp.eye(2))
        >>> mean, cov = gaussx.rts_interpolate(
        ...     *dynamics, *dynamics, state.filtered_means[0],
        ...     state.filtered_covs[0], means[2], covs[2],
        ... )
        >>> bool(jnp.allclose(mean, means[1])), bool(jnp.allclose(cov, covs[1]))
        (True, True)
    """
    # --- predict to t ---
    # μ_x,t|t−1 = F_t μ_x,t−1|t−1
    predicted_mean = A_fwd @ filtered_mean_prev
    # F_t Σ_x,t−1|t−1 F_tᵀ
    propagated = einsum(
        A_fwd,
        filtered_cov_prev,
        A_fwd,
        "next prev, prev col, other col -> next other",
    )
    # Σ_x,t|t−1 = F_t Σ_x,t−1|t−1 F_tᵀ + Σ_η,t, which is also Σ_x,t|t
    predicted_covariance = propagated + Q_fwd
    # ½ (Σ_x,t|t−1 + Σ_x,t|t−1ᵀ), against round-off asymmetry
    predicted_covariance = symmetrize(predicted_covariance)

    # --- predict to t + 1 ---
    # μ_x,t+1|t = F_t+1 μ_x,t|t
    predicted_mean_next = A_bwd @ predicted_mean
    # F_t+1 Σ_x,t|t F_t+1ᵀ
    propagated_next = einsum(
        A_bwd,
        predicted_covariance,
        A_bwd,
        "next prev, prev col, other col -> next other",
    )
    # Σ_x,t+1|t = F_t+1 Σ_x,t|t F_t+1ᵀ + Σ_η,t+1
    predicted_covariance_next = propagated_next + Q_bwd
    # ½ (Σ_x,t+1|t + Σ_x,t+1|tᵀ)
    predicted_covariance_next = symmetrize(predicted_covariance_next)
    predicted_covariance_next_op = lx.MatrixLinearOperator(
        predicted_covariance_next, lx.positive_semidefinite_tag
    )

    # --- smoother gain ---
    # Σ_x,t|t F_t+1ᵀ
    cross_covariance = einsum(
        predicted_covariance, A_bwd, "state col, next col -> state next"
    )
    # G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹, row by row (symmetric)
    smoother_gain = solve_rows(
        predicted_covariance_next_op, cross_covariance, solver=solver
    )

    # --- correct ---
    # μ_x,t+1|T − μ_x,t+1|t
    mean_correction = smoothed_mean_next - predicted_mean_next
    # μ_x,t|T = μ_x,t|t + G_smooth,t (μ_x,t+1|T − μ_x,t+1|t)
    smoothed_mean = predicted_mean + smoother_gain @ mean_correction
    # Σ_x,t+1|T − Σ_x,t+1|t
    covariance_correction = smoothed_cov_next - predicted_covariance_next
    # G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ
    correction = einsum(
        smoother_gain,
        covariance_correction,
        smoother_gain,
        "state next, next col, other col -> state other",
    )
    # Σ_x,t|T = Σ_x,t|t + G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ
    smoothed_covariance = symmetrize(predicted_covariance + correction)
    return smoothed_mean, smoothed_covariance
