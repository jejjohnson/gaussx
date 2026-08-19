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

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._distributions._joseph import joseph_update
from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._quadrature._integrator import AbstractIntegrator, moment_transform
from gaussx._quadrature._unscented import UnscentedIntegrator
from gaussx._ssm._kalman import FilterState
from gaussx._ssm._utils import _materialise
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


def _symmetric(matrix: Float[Array, "N N"]) -> lx.AbstractLinearOperator:
    """Wrap a dense covariance as a symmetric-tagged operator.

    Deliberately *not* ``positive_semidefinite_tag``: every covariance in
    this module is assembled by a quadrature rule, and a rule with negative
    weights can return an indefinite one. Symmetry is guaranteed (it is
    imposed); definiteness is not.
    """
    return lx.MatrixLinearOperator(matrix, lx.symmetric_tag)


def _broadcast_noise(
    noise: Float[Array, "*T D D"] | lx.AbstractLinearOperator,
    T: int,
    dim: int,
    name: str,
) -> Float[Array, "T D D"]:
    """Materialise a noise covariance and broadcast it along time.

    The trailing shape is checked against ``dim`` rather than only the
    rank: a ``(1, 1)`` or ``(D, 1)`` covariance would otherwise pass and
    then broadcast inside ``cov + noise``, adding a value across the whole
    covariance instead of raising.
    """
    dense = _materialise(noise)
    if dense.ndim == 2 and dense.shape == (dim, dim):
        return jnp.broadcast_to(dense, (T, dim, dim))
    if dense.ndim == 3 and dense.shape == (T, dim, dim):
        return dense
    msg = (
        f"{name} must have shape ({dim}, {dim}) or ({T}, {dim}, {dim}); "
        f"got {dense.shape}."
    )
    raise ValueError(msg)


def _normalise_mask(
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None,
    T: int,
    M: int,
) -> Bool[Array, " T"] | Bool[Array, "T M"]:
    """Validate and broadcast the observation mask, as `kalman_filter` does."""
    if mask is None:
        return jnp.ones((T,), dtype=bool)
    mask_seq = jnp.asarray(mask, dtype=bool)
    if mask_seq.ndim == 0:
        return jnp.broadcast_to(mask_seq, (T,))
    if mask_seq.ndim == 2:
        if mask_seq.shape != (T, M):
            msg = (
                f"mask must be a scalar or have shape ({T},) or ({T}, {M}); "
                f"got shape {mask_seq.shape}."
            )
            raise ValueError(msg)
        return mask_seq
    if mask_seq.shape != (T,):
        msg = (
            f"mask must be a scalar or have shape ({T},) or ({T}, {M}); "
            f"got shape {mask_seq.shape}."
        )
        raise ValueError(msg)
    return mask_seq


def masked_moment_inputs(
    obs_cov: Float[Array, "M M"],
    cross_cov: Float[Array, "N M"],
    obs_noise: Float[Array, "M M"],
    y: Float[Array, " M"],
    y_hat: Float[Array, " M"],
    mask: Bool[Array, " M"],
) -> tuple[
    Float[Array, "M M"],
    Float[Array, "N M"],
    Float[Array, "M M"],
    Float[Array, " M"],
    Float[Array, ""],
]:
    r"""Make masked observation channels inert in a moment-matched update.

    Marginalising channel $i$ out of a Kalman update is equivalent to
    keeping it but making it carry no information. For a linear filter that
    means zeroing row $i$ of $H$ and substituting a unit block into $R$. A
    moment-matched filter has no $H$ to zero, so the same substitution is
    applied to the **matched moments** instead:

    - zero the masked rows and columns of $\mathrm{Cov}[h(x)]$ (jointly,
      what zeroing a row of $H$ would do to $H P H^\top$),
    - zero the masked columns of $\mathrm{Cov}[x, h(x)]$ (likewise for
      $P H^\top$),
    - substitute a unit block into $R$,
    - zero the residual entry.

    The innovation then splits as

    $$
    S = \begin{bmatrix} S_{\mathrm{obs}} & 0 \\ 0 & I \end{bmatrix},
    $$

    so the gain $K = C S^{-1}$ has zero columns on the masked channels and
    they cannot move the state. The posterior is *exactly* the
    channel-deleted filter, with no branching — which is what lets the
    per-channel path run inside a `jax.lax.scan` without a `cond`.

    Exposed because the substitution is reusable: any moment-matched
    update — a custom filter loop, a smoother variant, an ensemble
    method — needs the same rewrite to handle partially observed vectors,
    and getting the joint row/column masking subtly wrong is easy.

    Note:
        Residuals are formed from separately-masked $y$ and $\hat y$
        rather than by masking their difference, so a masked $y$ entry may
        be ``NaN`` — the usual "not measured" encoding — without poisoning
        the reverse-mode gradient.

    Args:
        obs_cov: Matched $\mathrm{Cov}[h(x)]$, shape ``(M, M)``.
        cross_cov: Matched $\mathrm{Cov}[x, h(x)]$, shape ``(N, M)``.
        obs_noise: Observation noise $R$, shape ``(M, M)``.
        y: Observation vector, shape ``(M,)``. Masked entries are never
            read and may be ``NaN``.
        y_hat: Matched $\mathbb{E}[h(x)]$, shape ``(M,)``.
        mask: Per-channel mask, shape ``(M,)``. ``True`` keeps the channel.

    Returns:
        Tuple ``(obs_cov_eff, cross_cov_eff, obs_noise_eff, residual,
        n_missing)``. ``n_missing`` is the float count of masked channels;
        each contributed $-\tfrac{1}{2}\log 2\pi$ of dummy density to the
        full-vector log-likelihood, so adding
        ``0.5 * n_missing * log(2 pi)`` back recovers the exact marginal
        over the observed entries.
    """
    M = y.shape[-1]
    # keep[i, j] is True only where *both* channels survive, so masked
    # rows and columns are cleared together.
    keep = mask[:, None] & mask[None, :]

    # Zeroing row i of H would zero row i and column i of H P H^T; do that
    # directly to the matched Cov[h(x)].
    obs_cov_eff = jnp.where(keep, obs_cov, jnp.zeros_like(obs_cov))

    # ... and substitute a unit block into R on the masked channels, so
    # S = blockdiag(S_obs, I) rather than becoming singular.
    obs_noise_eff = jnp.where(keep, obs_noise, jnp.eye(M, dtype=obs_noise.dtype))

    # Column j of C = Cov[x, h(x)] is what channel j uses to move the
    # state; zero it and the gain's column j vanishes with it.
    cross_cov_eff = jnp.where(mask[None, :], cross_cov, jnp.zeros_like(cross_cov))

    # Mask y and y_hat *separately* rather than masking their difference:
    # a masked y entry is commonly NaN, and NaN in the discarded branch of
    # a where still poisons the reverse-mode gradient.
    residual = jnp.where(mask, y, jnp.zeros_like(y)) - jnp.where(
        mask, y_hat, jnp.zeros_like(y_hat)
    )

    n_missing = M - jnp.sum(mask.astype(y.dtype))
    return obs_cov_eff, cross_cov_eff, obs_noise_eff, residual, n_missing


def nonlinear_kalman_predict(
    dynamics: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    mean: Float[Array, " N"],
    cov: Float[Array, "N N"],
    process_noise: Float[Array, "N N"],
    *,
    integrator: AbstractIntegrator | None = None,
) -> tuple[Float[Array, " N"], Float[Array, "N N"]]:
    r"""One moment-matched predict step.

    $$
    m^- = \mathbb{E}[f(x)], \qquad P^- = \mathrm{Cov}[f(x)] + Q .
    $$

    Exposed alongside `gaussx.nonlinear_kalman_update` so a caller can
    drive their own loop -- an irregular time grid, a custom gating rule,
    a filter interleaved with something else -- without reimplementing the
    moment transform. `gaussx.nonlinear_kalman_filter` is exactly a
    `jax.lax.scan` over these two.

    Args:
        dynamics: State transition ``(N,) -> (N,)``. Deterministic.
        mean: Current mean, shape ``(N,)``.
        cov: Current covariance, shape ``(N, N)``.
        process_noise: $Q$, shape ``(N, N)``.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)`` — see
            `gaussx.nonlinear_kalman_filter` on why not ``alpha=1e-3``.

    Returns:
        Tuple ``(mean_pred, cov_pred)``.
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    # The process noise is additive and independent of x, so it enters only
    # as an additive term on the covariance -- the moment transform sees
    # the *deterministic* dynamics alone. No cross-covariance is needed
    # here (nothing is being conditioned on yet); the smoother re-runs the
    # same transform precisely to recover it.
    mean_pred, cov_dyn, _ = moment_transform(dynamics, mean, cov, integrator=integrator)
    return mean_pred, symmetrize(cov_dyn + process_noise)


def nonlinear_kalman_update(
    obs_fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    mean: Float[Array, " N"],
    cov: Float[Array, "N N"],
    observation: Float[Array, " M"],
    obs_noise: Float[Array, "M M"],
    *,
    integrator: AbstractIntegrator | None = None,
    mask: Bool[Array, " M"] | None = None,
    joseph: bool = True,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " N"], Float[Array, "N N"], Float[Array, ""]]:
    r"""One moment-matched update step.

    Moment-matches ``obs_fn`` at the predicted belief and runs the ordinary
    linear-Gaussian update against that matched joint:

    $$
    \hat y, S_{yy}, C = \mathcal{T}[h](m^-, P^-), \quad S = S_{yy} + R,
    \quad K = C S^{-1},
    $$

    $$
    m^+ = m^- + K(y - \hat y), \qquad
    \ell = -\tfrac{1}{2}\big(v^\top S^{-1} v + \log|S| + M \log 2\pi\big).
    $$

    The gain is built from the cross-covariance directly, so no Jacobian
    appears. See `gaussx.nonlinear_kalman_filter` for the meaning of
    ``joseph`` and the caveat on the returned log-likelihood.

    Args:
        obs_fn: Observation operator ``(N,) -> (M,)``.
        mean: Predicted mean $m^-$, shape ``(N,)``.
        cov: Predicted covariance $P^-$, shape ``(N, N)``.
        observation: Observed vector $y$, shape ``(M,)``.
        obs_noise: $R$, shape ``(M, M)``.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)`` — see
            `gaussx.nonlinear_kalman_filter` on why not ``alpha=1e-3``.
        mask: Optional per-channel mask, shape ``(M,)``. ``False`` entries
            are marginalised out exactly and may be ``NaN`` in
            ``observation``.
        joseph: Use the Joseph-form covariance update. Defaults to ``True``.
        solver: Optional solver strategy.

    Returns:
        Tuple ``(mean_upd, cov_upd, log_likelihood_increment)``. The
        increment is the exact marginal over the *observed* channels.
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    M = observation.shape[-1]

    y_hat, obs_cov, cross = moment_transform(obs_fn, mean, cov, integrator=integrator)

    if mask is None:
        obs_cov_e, cross_e, R_e = obs_cov, cross, obs_noise
        residual = observation - y_hat  # v = y - y_hat
        n_missing = jnp.zeros((), dtype=cov.dtype)
    else:
        # Rewrite the matched moments so masked channels carry no
        # information -- see `gaussx.masked_moment_inputs`.
        obs_cov_e, cross_e, R_e, residual, n_missing = masked_moment_inputs(
            obs_cov, cross, obs_noise, observation, y_hat, mask
        )

    # S = S_yy + R. Symmetrised because it is assembled from a weighted
    # outer-product sum, which drifts asymmetric.
    #
    # Tagged symmetric rather than positive-semidefinite: S_yy comes from a
    # quadrature rule, and rules with negative weights (the scaled
    # unscented transform, the degree-5 cubature rule above N=4) can return
    # an indefinite Cov[h(x)]. Claiming PSD would route the solve to a
    # Cholesky path that returns NaN on such a matrix instead of a solver
    # that copes.
    innovation = symmetrize(obs_cov_e + R_e)
    innovation_op = _symmetric(innovation)

    # K = C S^-1. solve_rows solves S x = c for each *row* of C, i.e. it
    # forms C S^-1 without inverting S.
    gain = solve_rows(innovation_op, cross_e, solver=solver)  # (N, M)

    # m+ = m- + K v
    mean_upd = mean + gain @ residual

    if joseph:
        # Joseph form: P+ = (I - K H)P-(I - K H)^T + K R K^T.
        #
        # That needs an H, which a moment-matched filter does not have. The
        # right stand-in is the statistical linearisation of h under the
        # predicted belief (gaussx#161 section 3.3.5): writing
        # h(x) ~ A x + b + eps, the regression gain is
        #
        #     A = C^T (P-)^-1
        #
        # which is exactly what statistical_linear_regression returns, and
        # exactly H when h is linear. So this reduces to the textbook
        # Joseph update in the linear case, while staying PSD for the
        # merely-approximate gain otherwise.
        #
        # Relative to the standard form the two differ by K Omega K^T, with
        # Omega = S_yy - A P- A^T the linearisation residual: PSD, and zero
        # for affine h. Hence switching this default cannot perturb the
        # linear reduction.
        obs_eff = solve_rows(_symmetric(cov), cross_e.T, solver=solver)

        # The noise of that regression is R + Omega, *not* R: linearising
        # h leaves a residual eps ~ N(0, Omega) on top of the measurement
        # noise, and Joseph form must be given the noise of the model whose
        # H it is using. Omega = S_yy - H_eff P- H_eff^T, so
        #
        #     R + Omega = S - H_eff C
        #
        # which is free here -- both factors are already formed.
        #
        # Passing R alone would return the matched-joint posterior minus
        # K Omega K^T, i.e. systematically overconfident on nonlinear maps.
        # With the residual included the two covariance forms agree to
        # 2.8e-17, so Joseph is a numerically safer route to the *same*
        # answer rather than a different one.
        effective_noise = symmetrize(innovation - obs_eff @ cross_e)
        cov_upd = joseph_update(cov, gain, obs_eff, effective_noise)
    else:
        # P+ = P- - K S K^T. Correct and cheaper, but its PSD-ness relies
        # on the moment triple being mutually consistent (Omega >= 0),
        # which a negative-weight rule can violate.
        cov_upd = symmetrize(cov - gain @ innovation @ gain.T)

    # ll += -0.5 (v^T S^-1 v + log|S| + M log 2pi).
    #
    # An approximation, not the exact marginal: S is the *matched*
    # innovation covariance. Exact when the maps are affine.
    solved = dispatch_solve(innovation_op, residual, solver)
    logdet = dispatch_logdet(innovation_op, solver)
    # Each masked channel contributed a dummy unit block to S, worth
    # -0.5 log(2 pi) of the full-vector density. Adding it back makes the
    # result the exact marginal over the observed entries, and independent
    # of the dummy block's variance.
    ll_inc = (
        -0.5 * (residual @ solved + logdet + M * _LOG_2PI) + 0.5 * n_missing * _LOG_2PI
    )
    return mean_upd, cov_upd, ll_inc


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
        solver: Optional solver strategy.

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

    # G = Sigma_xx+ (P-_{t+1})^-1, again via a row-wise solve rather than
    # an inverse. For linear f this is P_t A^T (P-_{t+1})^-1, the textbook
    # RTS gain.
    gain = solve_rows(_symmetric(cov_predicted), cross, solver=solver)  # (N, N)

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
    | `gaussx.FifthOrderCubatureIntegrator` | cubature Kalman filter (CKF) |
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
        init_mean: Initial state mean, shape ``(N,)``.
        init_cov: Initial state covariance, shape ``(N, N)``.
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
            ``None``, uses structural dispatch.

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
            dynamics, mean, cov, Q_t, integrator=integrator
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
        process_noise: Accepted for API symmetry with
            `gaussx.rts_smoother` and unused — the predicted covariances in
            ``filter_state`` already include it.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``; use the one the filter
            used.
        solver: Optional solver strategy. When ``None``, uses structural
            dispatch.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)``.
    """
    del process_noise  # predicted covariances already include it

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
