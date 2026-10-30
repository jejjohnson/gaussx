"""Moment-matched nonlinear Kalman filter and RTS smoother.

For the state-space model

    x_t = f(x_t−1) + η_t,    η_t ~ 𝒩(0, Σ_η,t)
    y_t = h(x_t) + ε_t,      ε_t ~ 𝒩(0, Σ_ε,t)

every Gaussian filter reduces to one repeated operation: given a Gaussian
over x and a map g (f or h), approximate the joint Gaussian over
(x, g(x)) by its moment triple

    (E[g(x)], Cov[g(x)], Cov[x, g(x)]),    x ~ 𝒩(μ_x, Σ_x).

The filter loop, the smoother backward pass and the log-likelihood are
identical across methods; only the moment transform changes. That is the
contract `gaussx.AbstractIntegrator` already specifies, so each integrator
supplies one realisation of the transform and this module supplies the
loop. The integrators own the sigma points and quadrature nodes:

    Taylor          linearise at the mean, Cov[x, g(x)] = Σ_x J_gᵀ    EKF
    unscented       2 D_x + 1 sigma points                         UKF
    cubature        2 D_x cubature points                          CKF
    Gauss-Hermite   a tensor-product grid                          GHKF
    Monte Carlo     samples                                        MC filter

The design this follows is written up in gaussx#161 (the moment-transform
protocol, sections 3.2-3.4); this module implements the discrete-time
filter and smoother of that design.
"""

from __future__ import annotations

import functools as ft
from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._symmetrize import symmetrize
from gaussx._quadrature._integrator import AbstractIntegrator, moment_transform
from gaussx._quadrature._unscented import UnscentedIntegrator
from gaussx._ssm._kalman import FilterState
from gaussx._ssm._nonlinear_update import (
    _broadcast_noise,
    _normalise_mask,
    _reject_indefinite,
    _resolve_validate,
    _solve_psd_or_lstsq,
    nonlinear_kalman_predict,
    nonlinear_kalman_update,
)
from gaussx._ssm._utils import _warn_unused_process_noise
from gaussx._strategies._base import AbstractSolverStrategy


def nonlinear_rts_step(
    dynamics: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    mean_filtered: Float[Array, " N"],
    cov_filtered: Float[Array, "N N"],
    mean_predicted: Float[Array, " N"],
    cov_predicted: Float[Array, "N N"],
    mean_smoothed: Float[Array, " N"],
    cov_smoothed: Float[Array, "N N"],
    *,
    integrator: AbstractIntegrator | None = None,
    solver: AbstractSolverStrategy | None = None,
    validate: bool | None = None,
) -> tuple[Float[Array, " N"], Float[Array, "N N"]]:
    """One moment-matched RTS backward step.

        G_smooth,t = Σ_xf,t Σ_x,t+1|t⁻¹
        μ_x,t|T = μ_x,t|t + G_smooth,t (μ_x,t+1|T − μ_x,t+1|t)
        Σ_x,t|T = Σ_x,t|t + G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ

    with Σ_xf,t = Cov[x_t, f(x_t)] from the integrator's moment transform at
    x_t ~ 𝒩(μ_x,t|t, Σ_x,t|t). For linear f = F_t+1 this is
    Σ_x,t|t F_t+1ᵀ, the textbook RTS gain. Exposed for the same reason as
    the filter steps: `gaussx.nonlinear_rts_smoother` is a `jax.lax.scan`
    over this.

    Algorithm:
        1. Re-run the filter's moment transform of f at the filtered belief
           for Σ_xf,t.
        2. Solve for G_smooth,t against Σ_x,t+1|t, by least squares when
           Σ_x,t+1|t is singular.
        3. Correct the filtered moments with the smoothed and predicted
           moments at t + 1.
        4. Optionally reject an indefinite Σ_x,t|T.

    Args:
        dynamics: Deterministic dynamics f.
        mean_filtered: Filtered mean μ_x,t|t.
        cov_filtered: Filtered covariance Σ_x,t|t.
        mean_predicted: Predicted mean μ_x,t+1|t.
        cov_predicted: Predicted covariance Σ_x,t+1|t.
        mean_smoothed: Smoothed mean μ_x,t+1|T.
        cov_smoothed: Smoothed covariance Σ_x,t+1|T.
        integrator: Moment-matching rule; use the one the filter used.
        solver: Accepted for API symmetry with `gaussx.rts_smoother` and
            unused. The smoother gain falls back to a least-squares solve
            so that a singular Σ_x,t+1|t (a deterministic or rank-deficient
            process) still yields the correction defined on its supported
            subspace, which supersedes the strategy.
        validate: Check that Σ_x,t|T is PSD. Defaults to
            ``not integrator.guarantees_psd(D_x)``.

    Returns:
        Tuple ``(mean, cov)`` holding μ_x,t|T and Σ_x,t|T.

    Raises:
        EquinoxRuntimeError: If validation is on and Σ_x,t|T is not PSD.

    Examples:
        At t + 1 the smoothed and predicted beliefs agree, so the step
        returns the filtered belief:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> mean, cov = jnp.zeros(2), jnp.eye(2)
        >>> predicted = (jnp.zeros(2), 2.0 * jnp.eye(2))
        >>> smoothed_mean, smoothed_cov = gaussx.nonlinear_rts_step(
        ...     lambda x: x, mean, cov, *predicted, *predicted
        ... )
        >>> bool(jnp.allclose(smoothed_cov, cov, atol=1e-6))
        True
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    # Re-run the same moment transform the filter used on the dynamics, at
    # the filtered belief for step t. Only the third element of the triple
    # is wanted: Cov[x_t, x_t+1] = Cov[x_t, f(x_t)], because the additive
    # model noise is independent of x_t and contributes nothing to the cross
    # term. That is also why Σ_η never appears here, and why the predicted
    # covariance passed in already accounts for it.
    # Σ_xf,t = Cov[x_t, f(x_t)]
    _, _, dynamics_cross_covariance = moment_transform(
        dynamics, mean_filtered, cov_filtered, integrator=integrator
    )

    # Least squares when Σ_x,t+1|t is singular, for the same reason as the
    # filter's Joseph linearisation: a deterministic or rank-deficient
    # process leaves Σ_x,t+1|t singular while the RTS correction stays well
    # defined on the subspace the belief actually occupies. A well-posed
    # solver returns NaN there, so a filter run that handles a singular
    # covariance would still fail the moment its result reached the
    # smoother.
    del solver  # the rank policy below supersedes the solver strategy
    # Σ_xf,tᵀ
    dynamics_cross_covariance_transpose = rearrange(
        dynamics_cross_covariance, "state next -> next state"
    )
    # G_smooth,tᵀ = Σ_x,t+1|t⁻¹ Σ_xf,tᵀ
    smoother_gain_transpose = _solve_psd_or_lstsq(
        cov_predicted, dynamics_cross_covariance_transpose
    )
    # G_smooth,t = Σ_xf,t Σ_x,t+1|t⁻¹
    smoother_gain = rearrange(smoother_gain_transpose, "next state -> state next")

    # Push the filtered belief toward the smoothed future, by however much
    # that future disagreed with what was predicted from here.
    # μ_x,t+1|T − μ_x,t+1|t
    mean_correction = mean_smoothed - mean_predicted
    # μ_x,t|T = μ_x,t|t + G_smooth,t (μ_x,t+1|T − μ_x,t+1|t)
    mean_new = mean_filtered + smoother_gain @ mean_correction
    # Σ_x,t+1|T − Σ_x,t+1|t, negative semi-definite in the exact case, which
    # is what makes smoothed variances no larger than filtered ones
    covariance_correction = cov_smoothed - cov_predicted
    # G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t)
    gain_correction = einsum(
        smoother_gain, covariance_correction, "state next, next col -> state col"
    )
    # G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ
    correction = einsum(
        gain_correction, smoother_gain, "state col, other col -> state other"
    )
    # Σ_x,t|T = Σ_x,t|t + G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ
    cov_new = symmetrize(cov_filtered + correction)
    if not _resolve_validate(validate, integrator, mean_filtered.shape[-1]):
        return mean_new, cov_new

    # Validated for the same reason predict and update are. The filtered and
    # predicted covariances can each be PSD while an inconsistent
    # cross-covariance still drives the correction indefinite: in one
    # dimension Σ_x,t|t = Σ_x,t+1|t = 1 with Σ_xf,t = 2 gives
    # G_smooth,t = 2 and a smoothed variance of −2.6.
    cov_new = _reject_indefinite(
        cov_new,
        "nonlinear_rts_step: the smoothed covariance is not positive "
        "semi-definite. The dynamics moment triple is not a consistent "
        "joint, which a negative-weight quadrature rule can produce. Use a "
        "positive-weight rule such as CubatureIntegrator or "
        "UnscentedIntegrator(alpha=1.0).",
    )
    return mean_new, cov_new


def _filter_step(
    dynamics: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    obs_fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    integrator: AbstractIntegrator,
    channel_mask: bool,
    joseph: bool,
    solver: AbstractSolverStrategy | None,
    validate: bool | None,
    carry: tuple[Float[Array, " N"], Float[Array, "N N"], Float[Array, ""]],
    inputs: tuple[Array, Array, Array, Array],
):
    """One `nonlinear_kalman_filter` scan step: predict, then a gated update.

    Args:
        dynamics: Deterministic dynamics f.
        obs_fn: Observation map h.
        integrator: Moment-matching rule.
        channel_mask: Whether ``inputs`` holds a per-channel mask.
        joseph: Use the Joseph-form covariance update.
        solver: Solver strategy for Σ_y,t.
        validate: Definiteness checks, as in `nonlinear_kalman_filter`.
        carry: ``(μ_x,t−1|t−1, Σ_x,t−1|t−1, log p(y_1:t−1))``.
        inputs: ``(Σ_η,t, Σ_ε,t, y_t, mask_t)``.

    Returns:
        The new carry and ``(μ_x,t|t, Σ_x,t|t, μ_x,t|t−1, Σ_x,t|t−1)``.
    """
    filtered_mean, filtered_covariance, log_likelihood = carry
    process_noise_t, obs_noise_t, obs_t, mask_t = inputs

    # The loop is exactly predict then update; both are public, so a caller
    # who wants a different loop can use them directly.
    predicted_mean, predicted_covariance = nonlinear_kalman_predict(
        dynamics,
        filtered_mean,
        filtered_covariance,
        process_noise_t,
        integrator=integrator,
        validate=validate,
    )

    def _update(_):
        return nonlinear_kalman_update(
            obs_fn,
            predicted_mean,
            predicted_covariance,
            obs_t,
            obs_noise_t,
            integrator=integrator,
            mask=mask_t if channel_mask else None,
            joseph=joseph,
            solver=solver,
            validate=validate,
        )

    def _skip(_):
        # Gated-off step: keep the prediction and contribute no likelihood.
        # Filtered equals predicted here, which is also what makes the
        # smoother's gain degenerate harmlessly at this step.
        zero = jnp.zeros((), dtype=predicted_covariance.dtype)
        return predicted_mean, predicted_covariance, zero

    if channel_mask:
        # No lax.cond needed: an all-False row already reduces the update to
        # the identity via the substitutions inside
        # `nonlinear_kalman_update`, so this path is branch-free.
        new_mean, new_covariance, log_likelihood_step = _update(None)
    else:
        # Gate the whole step so the predict-only branch evaluates neither
        # the update arithmetic nor its gradients.
        new_mean, new_covariance, log_likelihood_step = jax.lax.cond(
            mask_t, _update, _skip, operand=None
        )

    # log p(y_1:t) = log p(y_1:t−1) + log p(y_t ∣ y_1:t−1)
    new_log_likelihood = log_likelihood + log_likelihood_step
    new_carry = (new_mean, new_covariance, new_log_likelihood)
    outputs = (new_mean, new_covariance, predicted_mean, predicted_covariance)
    return new_carry, outputs


def nonlinear_kalman_filter(
    dynamics: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    obs_fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    obs_noise: Float[Array, "*T M M"] | lx.AbstractLinearOperator,
    observations: Float[Array, "T M"],
    init_mean: Float[Array, " N"],
    init_cov: Float[Array, "N N"],
    *,
    integrator: AbstractIntegrator | None = None,
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    joseph: bool = True,
    solver: AbstractSolverStrategy | None = None,
    validate: bool | None = None,
) -> FilterState:
    """Moment-matched nonlinear Kalman filter (EKF, UKF, CKF, GHKF, …).

        x_t = f(x_t−1) + η_t,    η_t ~ 𝒩(0, Σ_η,t)
        y_t = h(x_t) + ε_t,      ε_t ~ 𝒩(0, Σ_ε,t)

    with x₀ ~ 𝒩(μ_x,0, Σ_x,0) and t = 1, …, T. The Gaussian belief is
    pushed through f and h by ``integrator``'s moment transform in both the
    predict and the update step, so the choice of integrator is the choice
    of filter:

    | integrator | filter |
    |---|---|
    | `gaussx.TaylorIntegrator` | extended Kalman filter (EKF) |
    | `gaussx.UnscentedIntegrator` | unscented Kalman filter (UKF) |
    | `gaussx.CubatureIntegrator` | cubature Kalman filter (CKF), 2 D_x points |
    | `gaussx.FifthOrderCubatureIntegrator` | degree-5 cubature, 2 D_x² + 1 points |
    | `gaussx.GaussHermiteIntegrator` | Gauss-Hermite Kalman filter (GHKF) |
    | `gaussx.MonteCarloIntegrator` | Monte-Carlo Kalman filter |

    The gain comes from the integrator's cross-covariance Σ_xy,t, so no
    Jacobian appears anywhere in the loop: the Taylor rule forms
    H_t = J_h(μ_x,t|t−1) inside its own transform, which is what makes the
    EKF and the UKF the same code.

    The returned ``log_likelihood`` is a moment-matched surrogate, not the
    exact marginal likelihood: Σ_y,t is the matched observation covariance
    rather than the true one. It is exact when f and h are affine.
    Maximising it to tune hyperparameters maximises a surrogate, which is
    standard practice for nonlinear Gaussian filters but worth knowing.

    Unlike `gaussx.kalman_filter`, the covariance update defaults to the
    Joseph form. G_t = Σ_xy,t Σ_y,t⁻¹ is only approximately the optimal
    gain, and Σ_x,t|t−1 − G_t Σ_y,t G_tᵀ is PSD only for the optimal gain,
    whereas the Joseph form is a sum of two PSD terms for any G_t. The
    Joseph form needs an H_t, which a moment-matched filter does not have;
    the stand-in is the statistical linearisation H_t = Σ_xy,tᵀ Σ_x,t|t−1⁻¹
    (what `gaussx.statistical_linear_regression` returns as ``A``, and H_t
    itself when h is linear). Its noise is Σ_ε,t + Σ_lin,t, with Σ_lin,t
    the linearisation residual, not Σ_ε,t alone: dropping Σ_lin,t would
    understate the posterior covariance by G_t Σ_lin,t G_tᵀ on nonlinear
    maps. With it the two forms agree analytically, so ``joseph`` selects
    how the same covariance is computed, not which covariance you get.

    Algorithm:
        1. Predict: μ_x,t|t−1 = E[f(x_t−1)] and
           Σ_x,t|t−1 = Cov[f(x_t−1)] + Σ_η,t, by the moment transform of f.
        2. Moment-match h at the predicted belief for μ_y,t = E[h(x_t)],
           Σ_y,t = Cov[h(x_t)] + Σ_ε,t and Σ_xy,t = Cov[x_t, h(x_t)].
        3. Update: G_t = Σ_xy,t Σ_y,t⁻¹ and
           μ_x,t|t = μ_x,t|t−1 + G_t (y_t − μ_y,t); Σ_x,t|t in Joseph or
           standard form.
        4. Add log p(y_t ∣ y_1:t−1)
           ≈ −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t + D_y log 2π).

    Args:
        dynamics: Deterministic dynamics f; the model noise is added
            separately through ``process_noise``.
        obs_fn: Observation map h.
        process_noise: Additive model-error covariance Σ_η,t, one for all
            steps, a stack over time, or an operator (materialised once).
        obs_noise: Additive observation-error covariance Σ_ε,t, in the same
            forms.
        observations: Observations y_1:T, one row per step.
        init_mean: Prior mean μ_x,0 of the initial state x₀.
        init_cov: Prior covariance Σ_x,0 of x₀. The filter predicts before
            each update, so ``observations[0]`` observes x₁ = f(x₀) + η₁,
            not x₀. f is time-invariant, so to score a first observation
            against the prior itself, update the prior with
            `gaussx.nonlinear_kalman_update` and filter the remaining
            observations from there.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``, which is derivative-free and
            exact for affine maps. Must supply a cross-covariance.

            The ``alpha`` matters: `gaussx.UnscentedIntegrator`'s own
            default of ``1e-3`` places the sigma points ~1e-3 from the mean
            and recovers the moments by cancellation, which costs roughly
            seven digits. That is invisible in float64 but ruinous in
            float32 (JAX's default), where it misplaces the log-likelihood
            of a linear problem by over one nat. Pass ``alpha=1e-3``
            explicitly only for the classic scaled transform in x64.
        mask: Optional observation mask, with the semantics of
            `gaussx.kalman_filter`: a per-step mask gates whole steps, a
            per-channel mask gates individual channels. Masked entries of
            ``observations`` are never read, so they may be ``NaN``.
        joseph: Use the Joseph-form covariance update. Defaults to ``True``.
        solver: Solver strategy for Σ_y,t. With ``None``, Σ_y,t is
            factorised once by Cholesky.
        validate: Check at every step that Σ_x,t|t−1, Σ_y,t and Σ_x,t|t are
            positive (semi-)definite, raising an ``EquinoxRuntimeError``
            otherwise. Only a quadrature rule with negative weights can
            violate this, so it defaults to
            ``not integrator.guarantees_psd(D_x)``: on for the scaled
            unscented transform with small ``alpha``, the degree-5 cubature
            rule above D_x = 4 and any custom integrator; off (and free) for
            the default ``UnscentedIntegrator(alpha=1.0)`` and the
            cubature, Gauss-Hermite, Taylor and Monte Carlo rules. Pass
            ``True`` to force the checks.

    Returns:
        A `gaussx.FilterState`, identical in shape to the output of
        `gaussx.kalman_filter`.

    Raises:
        TypeError: If ``integrator`` does not supply a cross-covariance
            (raised at trace time).
        ValueError: If ``mask`` or the noise covariances are misshapen.

    Examples:
        A pendulum-like model filtered with the default UKF:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> def dynamics(x):
        ...     return jnp.array([x[0] + 0.1 * x[1], x[1] - 0.1 * jnp.sin(x[0])])
        >>> def obs_fn(x):
        ...     return jnp.sin(x[:1])
        >>> noises = (0.01 * jnp.eye(2), jnp.array([[0.1]]))
        >>> observations = jnp.array([[0.1], [0.2], [0.25]])
        >>> prior = (jnp.zeros(2), jnp.eye(2))
        >>> state = gaussx.nonlinear_kalman_filter(
        ...     dynamics, obs_fn, *noises, observations, *prior
        ... )
        >>> state.filtered_means.shape, state.log_likelihood.shape
        ((3, 2), ())

        With linear maps it matches `gaussx.kalman_filter`:

        >>> transition, obs_operator = jnp.eye(2), jnp.array([[1.0, 0.0]])
        >>> linear = gaussx.nonlinear_kalman_filter(
        ...     lambda x: transition @ x,
        ...     lambda x: obs_operator @ x,
        ...     *noises,
        ...     observations,
        ...     *prior,
        ... )
        >>> exact = gaussx.kalman_filter(
        ...     transition, obs_operator, *noises, observations, *prior
        ... )
        >>> bool(jnp.allclose(linear.log_likelihood, exact.log_likelihood))
        True
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    num_steps, obs_size = observations.shape
    state_size = init_mean.shape[-1]

    process_noises = _broadcast_noise(
        process_noise, num_steps, state_size, "process_noise"
    )
    obs_noises = _broadcast_noise(obs_noise, num_steps, obs_size, "obs_noise")
    masks = _normalise_mask(mask, num_steps, obs_size)
    channel_mask = masks.ndim == 2

    step = ft.partial(
        _filter_step,
        dynamics,
        obs_fn,
        integrator,
        channel_mask,
        joseph,
        solver,
        validate,
    )

    zero_log_likelihood = jnp.zeros((), dtype=init_cov.dtype)
    init_carry = (init_mean, init_cov, zero_log_likelihood)
    scan_inputs = (process_noises, obs_noises, observations, masks)
    final_carry, scan_outputs = jax.lax.scan(step, init_carry, scan_inputs)
    filtered_means, filtered_covs, predicted_means, predicted_covs = scan_outputs

    return FilterState(
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        log_likelihood=final_carry[2],
    )


def nonlinear_rts_smoother(
    filter_state: FilterState,
    dynamics: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    process_noise: Float[Array, "*T N N"] | lx.AbstractLinearOperator | None = None,
    *,
    integrator: AbstractIntegrator | None = None,
    solver: AbstractSolverStrategy | None = None,
    validate: bool | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T N N"]]:
    """Moment-matched nonlinear Rauch-Tung-Striebel smoother.

        G_smooth,t = Cov[x_t, f(x_t)] Σ_x,t+1|t⁻¹
        μ_x,t|T = μ_x,t|t + G_smooth,t (μ_x,t+1|T − μ_x,t+1|t)
        Σ_x,t|T = Σ_x,t|t + G_smooth,t (Σ_x,t+1|T − Σ_x,t+1|t) G_smooth,tᵀ

    The backward pass of `gaussx.nonlinear_kalman_filter`, sharing its
    moment transform: the smoother gain uses the integrator's
    cross-covariance between x_t and f(x_t). As in the filter, no Jacobian
    is formed; for linear f = F the gain reduces to
    Σ_x,t|t Fᵀ Σ_x,t+1|t⁻¹ and the whole pass to `gaussx.rts_smoother`.

    Algorithm:
        1. Start from the last filtered moments μ_x,T|T and Σ_x,T|T.
        2. For t = T − 1, …, 1, run `gaussx.nonlinear_rts_step` on the
           filtered moments at t and the predicted and smoothed moments at
           t + 1.

    Args:
        filter_state: Output of `gaussx.nonlinear_kalman_filter`. Pass the
            same ``dynamics`` and ``integrator`` used to produce it.
        dynamics: Deterministic dynamics f.
        process_noise: Deprecated and ignored: the predicted covariances in
            ``filter_state`` already include Σ_η. Passing it warns; it will
            be removed in gaussx 0.7.0.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``; use the one the filter used.
        solver: Accepted for API symmetry with `gaussx.rts_smoother` and
            unused; see `gaussx.nonlinear_rts_step`.
        validate: Check that each Σ_x,t|T is PSD; defaults to
            ``not integrator.guarantees_psd(D_x)``, as in
            `gaussx.nonlinear_kalman_filter`.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` holding μ_x,t|T and
        Σ_x,t|T.

    Examples:
        Smooth a nonlinear filter run:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> def dynamics(x):
        ...     return jnp.array([x[0] + 0.1 * x[1], x[1] - 0.1 * jnp.sin(x[0])])
        >>> state = gaussx.nonlinear_kalman_filter(
        ...     dynamics,
        ...     lambda x: jnp.sin(x[:1]),
        ...     0.01 * jnp.eye(2),
        ...     jnp.array([[0.1]]),
        ...     jnp.array([[0.1], [0.2], [0.25]]),
        ...     jnp.zeros(2),
        ...     jnp.eye(2),
        ... )
        >>> means, covs = gaussx.nonlinear_rts_smoother(state, dynamics)
        >>> means.shape, covs.shape
        ((3, 2), (3, 2, 2))

        The last smoothed moments are the last filtered ones:

        >>> bool(jnp.allclose(means[-1], state.filtered_means[-1]))
        True
    """
    _warn_unused_process_noise("nonlinear_rts_smoother", process_noise)

    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    num_steps = filter_state.filtered_means.shape[0]

    def step(carry, inputs):
        smoothed_mean_next, smoothed_covariance_next = carry
        (
            filtered_mean,
            filtered_covariance,
            predicted_mean_next,
            predicted_covariance_next,
        ) = inputs

        smoothed_mean, smoothed_covariance = nonlinear_rts_step(
            dynamics,
            filtered_mean,
            filtered_covariance,
            predicted_mean_next,
            predicted_covariance_next,
            smoothed_mean_next,
            smoothed_covariance_next,
            integrator=integrator,
            solver=solver,
            validate=validate,
        )

        new_carry = (smoothed_mean, smoothed_covariance)
        return new_carry, new_carry

    # The backward pass is seeded at the final step, where smoothed and
    # filtered coincide because there is no future left to condition on.
    init_carry = (
        filter_state.filtered_means[num_steps - 1],
        filter_state.filtered_covs[num_steps - 1],
    )

    # Step t consumes the filtered belief at t and the predicted belief at
    # t + 1, hence the offset slices; both are reversed so the scan runs
    # backwards through time.
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.filtered_covs[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
        filter_state.predicted_covs[1:][::-1],
    )

    _, (smoothed_means_rev, smoothed_covs_rev) = jax.lax.scan(step, init_carry, inputs)

    # Undo the reversal and re-attach the final step, which the scan never
    # produced because it was the seed.
    smoothed_means = jnp.concatenate(
        [smoothed_means_rev[::-1], filter_state.filtered_means[num_steps - 1 :]]
    )
    smoothed_covs = jnp.concatenate(
        [smoothed_covs_rev[::-1], filter_state.filtered_covs[num_steps - 1 :]]
    )
    return smoothed_means, smoothed_covs
