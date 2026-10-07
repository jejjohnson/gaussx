"""Moment-matched nonlinear Kalman filter and RTS smoother.

Every Gaussian filter for a nonlinear system reduces to one repeated
operation: given a Gaussian over ``x`` and a map ``g``, approximate the
joint Gaussian over ``(x, g(x))`` by its moment triple

    T[g; mu, Sigma] = (mu_g, Sigma_g, Sigma_xg),      x ~ N(mu, Sigma)

    mu_g     = E[g(x)]
    Sigma_g  = Cov[g(x)]
    Sigma_xg = Cov[x, g(x)]

The filter loop, the smoother backward pass and the log-likelihood are
*identical* across methods; only ``T`` changes. That is exactly the
contract `gaussx.AbstractIntegrator` already specifies, so each integrator
supplies one realisation of ``T`` and this module supplies the loop:

    Taylor          -> Sigma_xg = Sigma J^T          -> EKF
    unscented       -> from 2N+1 sigma points        -> UKF
    cubature        -> from 2N cubature points       -> CKF
    Gauss-Hermite   -> from a tensor-product grid    -> GHKF
    Monte Carlo     -> from samples                  -> MC filter

The design this follows is written up in gaussx#161 (the moment-transform
protocol, sections 3.2-3.4); this module implements the discrete-time
filter and smoother of that design.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

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
    r"""One moment-matched RTS backward step.

    $$
    G_t = \Sigma_{xx_+} (P^-_{t+1})^{-1}, \qquad
    m^s_t = m_t + G_t (m^s_{t+1} - m^-_{t+1}),
    $$

    with the matching covariance recursion. Exposed for the same reason as
    the filter steps: `gaussx.nonlinear_rts_smoother` is a
    `jax.lax.scan` over this.

    Args:
        dynamics: State transition ``(N,) -> (N,)``.
        mean_filtered: Filtered mean at $t$.
        cov_filtered: Filtered covariance at $t$.
        mean_predicted: Predicted mean at $t + 1$.
        cov_predicted: Predicted covariance at $t + 1$.
        mean_smoothed: Smoothed mean at $t + 1$.
        cov_smoothed: Smoothed covariance at $t + 1$.
        integrator: Moment-matching rule; use the one the filter used.
        solver: Accepted for API symmetry with `gaussx.rts_smoother` and
            unused. The smoother gain falls back to a least-squares solve
            so that a singular predicted covariance -- a deterministic or
            rank-deficient process -- still yields the correction defined
            on its supported subspace, which supersedes the strategy.
        validate: Check the smoothed covariance is PSD. Defaults to
            ``not integrator.guarantees_psd(N)``.

    Returns:
        Tuple ``(mean, cov)`` smoothed at $t$.
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    # Re-run the *same* moment transform the filter used on the dynamics,
    # at the filtered belief for step t. Only the third element of the
    # triple is wanted here:
    #
    #     Sigma_xx+ = Cov[x_t, x_{t+1}^-] = Cov[x_t, f(x_t)]
    #
    # The two are equal because the additive process noise is independent
    # of x_t, so it contributes nothing to the cross term -- which is also
    # why Q never appears here, and why the predicted covariance passed in
    # already accounts for it.
    _, _, cross = moment_transform(
        dynamics, mean_filtered, cov_filtered, integrator=integrator
    )

    # G = Sigma_xx+ (P-_{t+1})^-1. For linear f this is
    # P_t A^T (P-_{t+1})^-1, the textbook RTS gain.
    #
    # Least-squares when P^- is singular, for the same reason as the
    # filter's Joseph linearisation: a deterministic or rank-deficient
    # process leaves P^- singular while the RTS correction stays perfectly
    # well defined on the subspace the belief actually occupies. A
    # well-posed solver returns NaN there, so a filter run that handles a
    # singular covariance would still fail the moment its result reached
    # the smoother.
    del solver  # the rank policy below supersedes the solver strategy
    gain = _solve_psd_or_lstsq(cov_predicted, cross.T).T  # (N, N)

    # The RTS corrections: push the filtered belief toward the smoothed
    # future, by however much that future disagreed with what was predicted
    # from here.
    #
    #     m^s_t = m_t + G (m^s_{t+1} - m^-_{t+1})
    #     P^s_t = P_t + G (P^s_{t+1} - P^-_{t+1}) G^T
    #
    # Note P^s_{t+1} - P^-_{t+1} is negative semi-definite in the exact
    # case, which is what makes smoothed variances no larger than filtered
    # ones.
    mean_new = mean_filtered + gain @ (mean_smoothed - mean_predicted)
    cov_new = symmetrize(cov_filtered + gain @ (cov_smoothed - cov_predicted) @ gain.T)
    if not _resolve_validate(validate, integrator, mean_filtered.shape[-1]):
        return mean_new, cov_new

    # Validated for the same reason predict and update are. The filtered and
    # predicted covariances can each be PSD while an inconsistent
    # cross-covariance still drives the correction indefinite -- in one
    # dimension P_f = P_pred = 1 with cross = 2 gives G = 2 and a smoothed
    # variance of -2.6.
    cov_new = _reject_indefinite(
        cov_new,
        "nonlinear_rts_step: the smoothed covariance is not positive "
        "semi-definite. The dynamics moment triple is not a consistent "
        "joint, which a negative-weight quadrature rule can produce. Use a "
        "positive-weight rule such as CubatureIntegrator or "
        "UnscentedIntegrator(alpha=1.0).",
    )
    return mean_new, cov_new


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
    r"""Moment-matched nonlinear Kalman filter.

    Propagates a Gaussian belief through nonlinear ``dynamics`` and
    ``obs_fn`` by moment matching, using ``integrator`` for both the
    predict and the update step. **The choice of integrator is the choice
    of filter:**

    | integrator | filter |
    |---|---|
    | `gaussx.TaylorIntegrator` | extended Kalman filter (EKF) |
    | `gaussx.UnscentedIntegrator` | unscented Kalman filter (UKF) |
    | `gaussx.CubatureIntegrator` | cubature Kalman filter (CKF), $2N$ pts |
    | `gaussx.FifthOrderCubatureIntegrator` | degree-5 cubature, $2N^2+1$ pts |
    | `gaussx.GaussHermiteIntegrator` | Gauss-Hermite Kalman filter (GHKF) |
    | `gaussx.MonteCarloIntegrator` | Monte-Carlo Kalman filter |

    Each step is

    $$
    \begin{aligned}
    m^-, P^- &= \mathcal{T}[f](m, P), \quad P^- \mathrel{+}= Q, \\
    \hat y, S_{yy}, C &= \mathcal{T}[h](m^-, P^-), \quad S = S_{yy} + R, \\
    K &= C S^{-1}, \\
    m^+ &= m^- + K(y - \hat y),
    \end{aligned}
    $$

    where $\mathcal{T}$ is the integrator's moment transform. The gain
    comes from the integrator's cross-covariance, so **no Jacobian appears
    anywhere** — that is what makes the EKF and the UKF the same code.

    Note:
        ``log_likelihood`` is a **moment-matched surrogate**, not the exact
        marginal likelihood: $S$ is the matched innovation covariance
        rather than the true one. It reduces to the exact value when
        ``dynamics`` and ``obs_fn`` are affine. Users maximising it to tune
        hyperparameters are maximising a surrogate, which is standard
        practice for nonlinear Gaussian filters but worth knowing.

    Note:
        Unlike `gaussx.kalman_filter`, the covariance update defaults to
        Joseph form. $K = C S^{-1}$ is only approximately the optimal gain,
        and $P^- - K S K^\top$ is guaranteed PSD only *for* the optimal
        gain, whereas Joseph form is a sum of two PSD terms for any $K$.

        Joseph form needs an $H$, which a moment-matched filter does not
        have; the stand-in is the statistical-linearisation gain
        $H_{\text{eff}} = C^\top (P^-)^{-1}$ — what
        `gaussx.statistical_linear_regression` returns as ``A``, and
        exactly $H$ when ``obs_fn`` is linear. Its noise is $R + \Omega$,
        with $\Omega$ the linearisation residual, **not** $R$: dropping
        $\Omega$ would understate the posterior covariance by
        $K \Omega K^\top$ on nonlinear maps. With it included the two
        forms agree analytically, so ``joseph`` selects how the same
        covariance is computed, not which covariance you get.

    Args:
        dynamics: State transition ``(N,) -> (N,)``. Deterministic; process
            noise is added separately via ``process_noise``.
        obs_fn: Observation operator ``(N,) -> (M,)``.
        process_noise: $Q$, additive in state space. Shape ``(N, N)``,
            ``(T, N, N)``, or an operator (materialised once).
        obs_noise: $R$, additive in observation space. Shape ``(M, M)``,
            ``(T, M, M)``, or an operator.
        observations: Observed data, shape ``(T, M)``.
        init_mean: Mean of the prior on x₀, shape ``(N,)``.
        init_cov: Covariance of the prior on x₀, shape ``(N, N)``.
            The filter predicts before each update, so
            ``observations[0]`` observes ``x₁ = dynamics(x₀) + q``, not
            x₀. ``dynamics`` is time-invariant, so to score a first
            observation against the prior itself, update the prior with
            `gaussx.nonlinear_kalman_update` and filter the remaining
            observations from there.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``, which is derivative-free
            and exact for affine maps. Must supply a cross-covariance.

            The ``alpha`` matters: `gaussx.UnscentedIntegrator`'s own
            default of ``1e-3`` places the sigma points ~1e-3 from the mean
            and recovers the moments by cancellation, which costs roughly
            seven digits. That is invisible in float64 but ruinous in
            float32 — JAX's default — where it misplaces the
            log-likelihood of a *linear* problem by over one nat. Pass
            ``alpha=1e-3`` explicitly only if you want the classic scaled
            transform and are running in x64.
        mask: Optional observation mask, with the same semantics as
            `gaussx.kalman_filter` — ``(T,)`` gates whole steps, ``(T, M)``
            gates individual channels. Masked entries of ``observations``
            are never read, so they may be ``NaN``.
        joseph: Use the Joseph-form covariance update. Defaults to
            ``True``; see Notes.
        solver: Optional solver strategy for the innovation solve. When
            ``None``, the innovation is factorised once by Cholesky.
        validate: Check at every step that the predicted, innovation and
            updated covariances are positive (semi-)definite, raising an
            ``EquinoxRuntimeError`` otherwise. Only a quadrature rule with
            negative weights can violate this, so it defaults to
            ``not integrator.guarantees_psd(N)``: on for the scaled
            unscented transform with small ``alpha``, the degree-5 cubature
            rule above ``N = 4`` and any custom integrator; off (and free)
            for the default ``UnscentedIntegrator(alpha=1.0)`` and the
            cubature, Gauss-Hermite, Taylor and Monte Carlo rules. Pass
            ``True`` to force the checks.

    Returns:
        A `gaussx.FilterState`, identical in shape to
        `gaussx.kalman_filter`'s output.

    Raises:
        TypeError: If ``integrator`` does not supply a cross-covariance
            (raised at trace time).
        ValueError: If ``mask`` or the noise covariances are misshapen.
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    T, M = observations.shape

    Q_seq = _broadcast_noise(process_noise, T, init_mean.shape[-1], "process_noise")
    R_seq = _broadcast_noise(obs_noise, T, M, "obs_noise")
    mask_seq = _normalise_mask(mask, T, M)
    channel_mask = mask_seq.ndim == 2

    def step(carry, inputs):
        mean, cov, ll = carry
        Q_t, R_t, y_t, mask_t = inputs

        # The loop is exactly `predict` then `update`; both are public, so
        # a caller who wants a different loop can use them directly.
        mean_pred, cov_pred = nonlinear_kalman_predict(
            dynamics, mean, cov, Q_t, integrator=integrator, validate=validate
        )

        def _update(_):
            return nonlinear_kalman_update(
                obs_fn,
                mean_pred,
                cov_pred,
                y_t,
                R_t,
                integrator=integrator,
                mask=mask_t if channel_mask else None,
                joseph=joseph,
                solver=solver,
                validate=validate,
            )

        def _skip(_):
            # Gated-off step: keep the prediction and contribute no
            # likelihood. Filtered == predicted here, which is also what
            # makes the smoother's gain degenerate harmlessly at this step.
            return mean_pred, cov_pred, jnp.zeros((), dtype=cov_pred.dtype)

        if channel_mask:
            # No lax.cond needed: an all-False row already reduces the
            # update to the identity via the substitutions inside
            # `nonlinear_kalman_update`, so this path is branch-free.
            mean_new, cov_new, ll_inc = _update(None)
        else:
            # Gate the whole step so the predict-only branch evaluates
            # neither the update arithmetic nor its gradients.
            mean_new, cov_new, ll_inc = jax.lax.cond(
                mask_t, _update, _skip, operand=None
            )

        carry_new = (mean_new, cov_new, ll + ll_inc)
        return carry_new, (mean_new, cov_new, mean_pred, cov_pred)

    init_carry = (init_mean, init_cov, jnp.zeros((), dtype=init_cov.dtype))
    final_carry, (f_means, f_covs, p_means, p_covs) = jax.lax.scan(
        step, init_carry, (Q_seq, R_seq, observations, mask_seq)
    )

    return FilterState(
        filtered_means=f_means,
        filtered_covs=f_covs,
        predicted_means=p_means,
        predicted_covs=p_covs,
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
    r"""Moment-matched nonlinear Rauch-Tung-Striebel smoother.

    The backward pass of `gaussx.nonlinear_kalman_filter`, sharing the same
    moment transform — the smoother gain uses the integrator's
    cross-covariance between $x_t$ and $x_{t+1}$:

    $$
    G_t = \mathrm{Cov}[x_t, f(x_t)]\, (P^-_{t+1})^{-1},
    \qquad
    m^s_t = m_t + G_t (m^s_{t+1} - m^-_{t+1}),
    $$

    with the matching covariance recursion. As in the filter, no Jacobian
    is formed; for linear ``dynamics`` the gain reduces to
    $P_t A^\top (P^-_{t+1})^{-1}$ and the whole pass to
    `gaussx.rts_smoother`.

    Args:
        filter_state: Output of `gaussx.nonlinear_kalman_filter`. Pass the
            same ``dynamics`` and ``integrator`` used to produce it.
        dynamics: State transition ``(N,) -> (N,)``.
        process_noise: Deprecated and ignored -- the predicted covariances
            in ``filter_state`` already include it. Passing it warns; it will
            be removed in gaussx 0.7.0.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``; use the one the filter
            used.
        solver: Accepted for API symmetry with `gaussx.rts_smoother` and
            unused -- see `gaussx.nonlinear_rts_step`.
        validate: Check each smoothed covariance is PSD; defaults to
            ``not integrator.guarantees_psd(N)``, as in
            `gaussx.nonlinear_kalman_filter`.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    _warn_unused_process_noise("nonlinear_rts_smoother", process_noise)

    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    T = filter_state.filtered_means.shape[0]

    def step(carry, inputs):
        mean_smooth, cov_smooth = carry
        mean_filt, cov_filt, mean_pred, cov_pred = inputs

        mean_new, cov_new = nonlinear_rts_step(
            dynamics,
            mean_filt,
            cov_filt,
            mean_pred,
            cov_pred,
            mean_smooth,
            cov_smooth,
            integrator=integrator,
            solver=solver,
            validate=validate,
        )

        return (mean_new, cov_new), (mean_new, cov_new)

    # The backward pass is seeded at the final step, where smoothed and
    # filtered coincide because there is no future left to condition on.
    init_carry = (
        filter_state.filtered_means[T - 1],
        filter_state.filtered_covs[T - 1],
    )

    # Step t consumes the filtered belief at t and the *predicted* belief at
    # t+1, hence the offset slices; both are reversed so the scan runs
    # backwards through time.
    inputs = (
        filter_state.filtered_means[:-1][::-1],
        filter_state.filtered_covs[:-1][::-1],
        filter_state.predicted_means[1:][::-1],
        filter_state.predicted_covs[1:][::-1],
    )

    _, (s_means_rev, s_covs_rev) = jax.lax.scan(step, init_carry, inputs)

    # Undo the reversal and re-attach the final step, which the scan never
    # produced because it was the seed.
    s_means = jnp.concatenate(
        [s_means_rev[::-1], filter_state.filtered_means[T - 1 :]], axis=0
    )
    s_covs = jnp.concatenate(
        [s_covs_rev[::-1], filter_state.filtered_covs[T - 1 :]], axis=0
    )
    return s_means, s_covs
