"""Moment-matched predict and update steps of the nonlinear Kalman filter.

`gaussx.nonlinear_kalman_predict` and `gaussx.nonlinear_kalman_update` with
their validity and masking helpers. The filter and smoother loops built on
them, and the design they follow, are in `gaussx._ssm._nonlinear_filter`.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._distributions._joseph import joseph_update
from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._quadrature._integrator import AbstractIntegrator, moment_transform
from gaussx._quadrature._unscented import UnscentedIntegrator
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


def _check_noise_shape(
    noise: Float[Array, "D D"],
    dim: int,
    name: str,
) -> Float[Array, "D D"]:
    """Reject a covariance whose shape would broadcast instead of failing.

    A ``(1, 1)`` or ``(D, 1)`` covariance passes a rank-only check and is
    then broadcast across every entry of the matrix it is added to, which
    corrupts the result silently rather than raising.
    """
    if jnp.shape(noise) != (dim, dim):
        msg = f"{name} must have shape ({dim}, {dim}); got {jnp.shape(noise)}."
        raise ValueError(msg)
    return noise


def _reject_indefinite(
    cov: Float[Array, "N N"],
    message: str,
) -> Float[Array, "N N"]:
    """Reject a covariance that is not positive semi-definite.

    The threshold is scaled to the size of the covariance rather than being
    exactly zero. A legitimately rank-deficient covariance -- a
    deterministic state, zero process noise -- is singular by construction,
    and rounding puts its null directions a few ulps either side of zero; a
    strict test would reject those. Genuine indefiniteness from an
    inconsistent moment triple is orders of magnitude away from this bound.

    The full spectrum is checked rather than the diagonal, because an
    indefinite covariance can have entirely positive variances -- e.g.
    ``[[0.36, -0.64], [-0.64, 0.36]]``, whose smallest eigenvalue is -0.28.
    """
    dimension = cov.shape[-1]
    tolerance = dimension * jnp.finfo(cov.dtype).eps * jnp.maximum(jnp.trace(cov), 1.0)
    return eqx.error_if(cov, jnp.linalg.eigvalsh(cov).min() < -tolerance, message)


def _resolve_validate(
    validate: bool | None, integrator: AbstractIntegrator, dim: int
) -> bool:
    """``validate`` as given, or whether ``integrator`` can go indefinite.

    The definiteness checks cost an eigendecomposition each and can only
    fire for a rule without a PSD guarantee, so by default they are skipped
    for the built-in positive-weight rules (gh-331) and kept for everything
    else, custom rules included.
    """
    if validate is None:
        return not integrator.guarantees_psd(dim)
    return validate


def _solve_psd_or_lstsq(
    P: Float[Array, "N N"],
    B: Float[Array, "N K"],
) -> Float[Array, "N K"]:
    """``P⁻¹ B`` for PSD ``P``: Cholesky, or minimum-norm least squares.

    ``P`` is legitimately singular for a deterministic initial state, zero
    process noise or dimension-reducing dynamics, where a Cholesky solve
    returns ``NaN`` although the minimum-norm solution is well defined.
    The least-squares (SVD) branch runs only when the Cholesky pivots say
    ``P`` is numerically singular, so the common case costs one ``potrf``
    instead of an SVD (gh-331). Each branch sees a safe stand-in for the
    other's input, so the branch not taken -- evaluated anyway under
    ``vmap`` -- cannot put ``NaN`` into the gradient.
    """
    n = P.shape[-1]
    chol = jnp.linalg.cholesky(P)
    pivots = jnp.diagonal(chol)
    scale = jnp.max(jnp.abs(jnp.diagonal(P)))
    nonsingular = jnp.all(jnp.isfinite(chol)) & (
        jnp.min(pivots) ** 2 > n * jnp.finfo(P.dtype).eps * scale
    )
    eye = jnp.eye(n, dtype=P.dtype)
    safe_chol = jnp.where(nonsingular, chol, eye)
    safe_P = jnp.where(nonsingular, eye, P)
    return jax.lax.cond(
        nonsingular,
        lambda: jsla.cho_solve((safe_chol, True), B),
        # rcond=0.0 keeps every representable mode: lstsq's default cutoff
        # scales with epsilon, which in float32 also discards small-but-real
        # covariance modes (for P = diag(1, 1e-8) it would zero that axis).
        lambda: jnp.linalg.lstsq(safe_P, B, rcond=0.0)[0],
    )


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
    validate: bool | None = None,
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
        validate: Check the predicted covariance is PSD. Defaults to
            ``not integrator.guarantees_psd(N)``; see
            `gaussx.nonlinear_kalman_filter`.

    Returns:
        Tuple ``(mean_pred, cov_pred)``.
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    process_noise = _check_noise_shape(process_noise, mean.shape[-1], "process_noise")

    # The process noise is additive and independent of x, so it enters only
    # as an additive term on the covariance -- the moment transform sees
    # the *deterministic* dynamics alone. No cross-covariance is needed
    # here (nothing is being conditioned on yet); the smoother re-runs the
    # same transform precisely to recover it.
    mean_pred, cov_dyn, _ = moment_transform(dynamics, mean, cov, integrator=integrator)
    cov_pred = symmetrize(cov_dyn + process_noise)
    if not _resolve_validate(validate, integrator, mean.shape[-1]):
        return mean_pred, cov_pred

    # Validate here as well as after the update. A negative-weight rule can
    # return an indefinite Cov[f(x)] that a small process noise does not
    # repair, and a step-level False mask would then expose it directly as
    # the filtered covariance. Nothing downstream would complain: the next
    # moment transform tags it PSD, and the dense square-root path clips
    # negative eigenvalues to zero, silently altering the belief rather
    # than reporting it.
    cov_pred = _reject_indefinite(
        cov_pred,
        "nonlinear_kalman_predict: the predicted covariance is not positive "
        "semi-definite. Cov[f(x)] came back indefinite, which a "
        "negative-weight quadrature rule can produce, and process_noise did "
        "not repair it. Use a positive-weight rule such as "
        "CubatureIntegrator or UnscentedIntegrator(alpha=1.0).",
    )
    return mean_pred, cov_pred


def _innovation_gain(
    innovation: Float[Array, "M M"],
    cross_e: Float[Array, "N M"],
    residual: Float[Array, " M"],
    solver: AbstractSolverStrategy | None,
    validate: bool,
) -> tuple[
    Float[Array, "M M"], Float[Array, "N M"], Float[Array, " M"], Float[Array, ""]
]:
    """Gain ``K = C S⁻¹``, ``S⁻¹ v`` and ``log|S|`` from one factorisation of S.

    Returns ``(innovation, gain, solved, logdet)``; ``innovation`` is passed
    back because the validity check wraps it in `equinox.error_if`.
    """
    # A quadrature rule with negative weights can return an indefinite
    # Cov[h(x)], and R may be too small to repair it. Neither the update
    # nor the likelihood is defined then -- the quadratic form can go
    # negative and the log-determinant becomes log|det S| -- so this is
    # rejected rather than allowed to produce a plausible-looking but
    # meaningless number. With a positive-weight rule S_yy is PSD, so S is
    # positive definite whenever R is and the check is skipped (gh-331).
    innovation_message = (
        "nonlinear_kalman_update: the innovation covariance S = Cov[h(x)] + R "
        "is not positive definite. A negative-weight quadrature rule (the "
        "scaled unscented transform, or the degree-5 cubature rule above "
        "N = 4) can return an indefinite Cov[h(x)]. Use a positive-weight "
        "rule such as CubatureIntegrator or UnscentedIntegrator(alpha=1.0), "
        "or increase obs_noise."
    )
    if solver is None:
        # One Cholesky of S serves the gain, the residual solve and the
        # log-determinant; a non-finite factor is the definiteness check.
        chol = jnp.linalg.cholesky(innovation)
        if validate:
            chol = eqx.error_if(
                chol, ~jnp.all(jnp.isfinite(jnp.diagonal(chol))), innovation_message
            )
        # K = C S^-1, i.e. S Kᵀ = Cᵀ.
        gain = jsla.cho_solve((chol, True), cross_e.T).T  # (N, M)
        solved = jsla.cho_solve((chol, True), residual)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
    else:
        if validate:
            innovation = eqx.error_if(
                innovation,
                jnp.linalg.eigvalsh(innovation).min() <= 0.0,
                innovation_message,
            )
        innovation_op = _symmetric(innovation)
        # K = C S^-1. solve_rows solves S x = c for each *row* of C.
        gain = solve_rows(innovation_op, cross_e, solver=solver)  # (N, M)
        solved = dispatch_solve(innovation_op, residual, solver)
        logdet = dispatch_logdet(innovation_op, solver)

    return innovation, gain, solved, logdet


def _updated_covariance(
    cov: Float[Array, "N N"],
    gain: Float[Array, "N M"],
    innovation: Float[Array, "M M"],
    cross_e: Float[Array, "N M"],
    joseph: bool,
) -> Float[Array, "N N"]:
    """The posterior covariance, in Joseph or standard form."""
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
        # H_eff = C^T (P^-)^-1. P^- is legitimately singular for a
        # deterministic initial state, zero process noise, or
        # dimension-reducing dynamics, and a well-posed solve returns NaN
        # on those even though the update itself is perfectly well defined
        # (R keeps S invertible). The pseudo-inverse then gives the
        # minimum-norm H_eff, the natural reading of the linearisation when
        # the belief is confined to a subspace; a Cholesky solve covers the
        # nonsingular case at a fraction of the cost.
        obs_eff = _solve_psd_or_lstsq(cov, cross_e).T

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

    return cov_upd


def _reject_inconsistent_posterior(cov_upd: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """Raise if the updated covariance is not PSD (an inconsistent joint)."""
    # A positive-definite S is not on its own enough: the *joint* over
    # (x, h(x)) must be consistent. An inconsistent triple -- Omega =
    # S_yy - H_eff P^- H_eff^T indefinite, which a negative-weight rule can
    # produce even where S is fine -- leaves the posterior indefinite.
    #
    # The full spectrum is checked, not just the diagonal: an indefinite
    # covariance can have entirely positive variances, e.g.
    # [[0.36, -0.64], [-0.64, 0.36]], whose smallest eigenvalue is -0.28.
    # A diagonal test would pass that and hand the next predict step a
    # covariance it will go on to treat as PSD.
    #
    # The threshold is scaled to the size of the covariance rather than
    # being exactly zero. A legitimately rank-deficient posterior -- a
    # deterministic state, zero process noise -- is singular by
    # construction, and rounding puts its null directions a few ulps either
    # side of zero; a strict test would reject those. Genuine
    # inconsistency is not marginal: the example above sits at -0.28
    # against a trace of 0.72, many orders above this bound.
    return _reject_indefinite(
        cov_upd,
        "nonlinear_kalman_update: the updated covariance is not positive "
        "semi-definite. The matched moments (Cov[h(x)], Cov[x, h(x)]) are "
        "not a consistent joint, which a negative-weight quadrature rule "
        "can produce. Use a positive-weight rule such as CubatureIntegrator "
        "or UnscentedIntegrator(alpha=1.0).",
    )


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
    validate: bool | None = None,
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
        solver: Optional solver strategy. With ``None`` the innovation is
            factorised once by Cholesky and shared by the gain, the
            residual solve and the log-determinant.
        validate: Check the innovation is positive definite and the
            updated covariance PSD. Defaults to
            ``not integrator.guarantees_psd(N)``; see
            `gaussx.nonlinear_kalman_filter`.

    Returns:
        Tuple ``(mean_upd, cov_upd, log_likelihood_increment)``. The
        increment is the exact marginal over the *observed* channels.
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)
    validate = _resolve_validate(validate, integrator, mean.shape[-1])

    M = observation.shape[-1]
    obs_noise = _check_noise_shape(obs_noise, M, "obs_noise")
    if mask is not None and jnp.shape(mask) != (M,):
        # A (1,) mask would broadcast across every channel: False would
        # suppress them all and return an identity update, True would enable
        # them all, in either case silently.
        msg = f"mask must have shape ({M},); got {jnp.shape(mask)}."
        raise ValueError(msg)

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
    innovation = symmetrize(obs_cov_e + R_e)

    innovation, gain, solved, logdet = _innovation_gain(
        innovation, cross_e, residual, solver, validate
    )

    # m+ = m- + K v
    mean_upd = mean + gain @ residual

    cov_upd = _updated_covariance(cov, gain, innovation, cross_e, joseph)

    ll_inc = _log_likelihood_increment(residual, solved, logdet, M, n_missing)
    if not validate:
        return mean_upd, cov_upd, ll_inc
    cov_upd = _reject_inconsistent_posterior(cov_upd)

    return mean_upd, cov_upd, ll_inc


def _log_likelihood_increment(
    residual: Float[Array, " M"],
    solved: Float[Array, " M"],
    logdet: Float[Array, ""],
    M: int,
    n_missing: Float[Array, ""],
) -> Float[Array, ""]:
    """``-0.5 (vᵀ S⁻¹ v + log|S| + M log 2π)``, over the observed channels.

    An approximation, not the exact marginal: ``S`` is the *matched*
    innovation covariance. Exact when the maps are affine. Each masked
    channel contributed a dummy unit block to ``S``, worth
    ``-0.5 log 2π`` of the full-vector density; adding it back makes the
    result the exact marginal over the observed entries, independent of
    the dummy block's variance.
    """
    return (
        -0.5 * (residual @ solved + logdet + M * _LOG_2PI) + 0.5 * n_missing * _LOG_2PI
    )
