"""Moment-matched predict and update steps of the nonlinear Kalman filter.

`gaussx.nonlinear_kalman_predict` and `gaussx.nonlinear_kalman_update`, with
their validity and masking helpers. The filter and smoother loops built on
them, and the design they follow, are in `gaussx._ssm._nonlinear_filter`.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._distributions._joseph import joseph_update
from gaussx._einx import einsum, rearrange
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

    Args:
        matrix: A symmetric dense covariance.

    Returns:
        ``matrix`` as a symmetric-tagged lineax operator.
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

    Args:
        noise: The covariance to check.
        dim: Its expected side length.
        name: Argument name for the error message.

    Returns:
        ``noise``, unchanged.

    Raises:
        ValueError: If ``noise`` is not ``(dim, dim)``.
    """
    if jnp.shape(noise) != (dim, dim):
        msg = f"{name} must have shape ({dim}, {dim}); got {jnp.shape(noise)}."
        raise ValueError(msg)
    return noise


def _reject_indefinite(
    cov: Float[Array, "N N"],
    message: str,
) -> Float[Array, "N N"]:
    """Reject a covariance Σ whose smallest eigenvalue is below −D ϵ max(tr Σ, 1).

    The threshold is scaled to the size of the covariance rather than being
    exactly zero. A legitimately rank-deficient covariance -- a
    deterministic state, zero model noise -- is singular by construction,
    and rounding puts its null directions a few ulps either side of zero; a
    strict test would reject those. Genuine indefiniteness from an
    inconsistent moment triple is orders of magnitude away from this bound.

    The full spectrum is checked rather than the diagonal, because an
    indefinite covariance can have entirely positive variances -- e.g.
    ``[[0.36, -0.64], [-0.64, 0.36]]``, whose smallest eigenvalue is -0.28.

    Args:
        cov: The covariance Σ to check, of side D.
        message: Error message if it is indefinite.

    Returns:
        ``cov``, wrapped in `equinox.error_if`.
    """
    dimension = cov.shape[-1]
    # D ϵ
    scaled_eps = dimension * jnp.finfo(cov.dtype).eps
    # D ϵ max(tr Σ, 1)
    tolerance = scaled_eps * jnp.maximum(jnp.trace(cov), 1.0)
    # min eig(Σ)
    smallest_eigenvalue = jnp.linalg.eigvalsh(cov).min()
    return eqx.error_if(cov, smallest_eigenvalue < -tolerance, message)


def _resolve_validate(
    validate: bool | None, integrator: AbstractIntegrator, dim: int
) -> bool:
    """``validate`` as given, or whether ``integrator`` can go indefinite.

    The definiteness checks cost an eigendecomposition each and can only
    fire for a rule without a PSD guarantee, so by default they are skipped
    for the built-in positive-weight rules (gh-331) and kept for everything
    else, custom rules included.

    Args:
        validate: The caller's choice, or ``None`` for the default.
        integrator: The moment-matching rule.
        dim: State length D_x.

    Returns:
        Whether to run the definiteness checks.
    """
    if validate is None:
        return not integrator.guarantees_psd(dim)
    return validate


def _solve_psd_or_lstsq(
    covariance: Float[Array, "N N"],
    rhs: Float[Array, "N K"],
) -> Float[Array, "N K"]:
    """Σ⁻¹ B for a PSD Σ: by Cholesky, or by minimum-norm least squares.

    Σ is legitimately singular for a deterministic initial state, zero
    model noise or dimension-reducing dynamics, where a Cholesky solve
    returns ``NaN`` although the minimum-norm solution is well defined.
    The least-squares (SVD) branch runs only when the Cholesky pivots say
    Σ is numerically singular, so the common case costs one ``potrf``
    instead of an SVD (gh-331). Each branch sees a safe stand-in for the
    other's input, so the branch not taken -- evaluated anyway under
    ``vmap`` -- cannot put ``NaN`` into the gradient.

    Args:
        covariance: The PSD matrix Σ.
        rhs: The right-hand sides B, one per column.

    Returns:
        Σ⁻¹ B, or the minimum-norm least-squares solution when Σ is
        singular.
    """
    size = covariance.shape[-1]
    # Σ = L Lᵀ
    cholesky_factor = jnp.linalg.cholesky(covariance)
    pivots = jnp.diagonal(cholesky_factor)
    scale = jnp.max(jnp.abs(jnp.diagonal(covariance)))
    # min Lᵢᵢ² > D ϵ maxᵢ |Σᵢᵢ|
    smallest_pivot_squared = jnp.min(pivots) ** 2
    threshold = size * jnp.finfo(covariance.dtype).eps * scale
    is_finite = jnp.all(jnp.isfinite(cholesky_factor))
    nonsingular = is_finite & (smallest_pivot_squared > threshold)
    eye = jnp.eye(size, dtype=covariance.dtype)
    safe_cholesky_factor = jnp.where(nonsingular, cholesky_factor, eye)
    safe_covariance = jnp.where(nonsingular, eye, covariance)
    return jax.lax.cond(
        nonsingular,
        # Σ⁻¹ B = L⁻ᵀ L⁻¹ B
        lambda: jsla.cho_solve((safe_cholesky_factor, True), rhs),
        # rcond=0.0 keeps every representable mode: lstsq's default cutoff
        # scales with epsilon, which in float32 also discards small-but-real
        # covariance modes (for Σ = diag(1, 1e-8) it would zero that axis).
        lambda: jnp.linalg.lstsq(safe_covariance, rhs, rcond=0.0)[0],
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

    Args:
        noise: One covariance for all steps, a stack over time, or an
            operator.
        T: Number of steps.
        dim: Side length of each covariance.
        name: Argument name for the error message.

    Returns:
        A dense stack of covariances, one per step.

    Raises:
        ValueError: If ``noise`` is neither ``(dim, dim)`` nor
            ``(T, dim, dim)``.
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
    """Validate and broadcast the observation mask, as `kalman_filter` does.

    Args:
        mask: ``None``, a scalar, a per-step or a per-channel mask.
        T: Number of steps.
        M: Observation length D_y.

    Returns:
        A per-step mask, or the per-channel mask unchanged.

    Raises:
        ValueError: If ``mask`` has any other shape.
    """
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
    """Make masked observation channels inert in a moment-matched update.

        Σ_y = [[Σ_y,obs, 0], [0, I]],    G = Σ_xy Σ_y⁻¹ = [G_obs, 0]

    Marginalising channel i out of a Kalman update is equivalent to keeping
    it but making it carry no information. For a linear filter that means
    zeroing row i of H and substituting a unit block into Σ_ε. A
    moment-matched filter has no H to zero, so the same substitution is
    applied to the matched moments instead. The observation covariance
    Σ_y = Σ_h + Σ_ε then splits into the observed block and an identity, so
    the gain has zero columns on the masked channels and they cannot move
    the state. The posterior is exactly the channel-deleted filter, with no
    branching, which lets the per-channel path run inside a `jax.lax.scan`
    without a ``cond``.

    Exposed because the substitution is reusable: any moment-matched update
    (a custom filter loop, a smoother variant, an ensemble method) needs the
    same rewrite to handle partially observed vectors, and getting the joint
    row and column masking subtly wrong is easy.

    Algorithm:
        1. Zero the masked rows and columns of Σ_h = Cov[h(x)] jointly, which
           is what zeroing a row of H does to H Σ_x Hᵀ.
        2. Substitute a unit block into Σ_ε on the masked channels.
        3. Zero the masked columns of Σ_xy, as zeroing a row of H does to
           Σ_x Hᵀ.
        4. Form the residual r = y − μ_y from separately masked y and μ_y,
           so a masked y entry may be ``NaN`` (the usual "not measured"
           encoding) without poisoning the reverse-mode gradient.

    Args:
        obs_cov: Matched covariance Σ_h = Cov[h(x)] of the observation map.
        cross_cov: Matched cross-covariance Σ_xy of x and h(x).
        obs_noise: Observation-error covariance Σ_ε.
        y: Observation y. Masked entries are never read and may be ``NaN``.
        y_hat: Matched predicted observation μ_y = E[h(x)].
        mask: Per-channel mask; ``True`` keeps the channel.

    Returns:
        Tuple ``(obs_cov_eff, cross_cov_eff, obs_noise_eff, residual,
        n_missing)``: the masked Σ_h, Σ_xy and Σ_ε, the residual r and the
        float count N_missing of masked channels. Each masked channel
        contributes −½ log 2π of dummy density to the full-vector
        log-likelihood, so adding ½ N_missing log 2π back recovers the exact
        marginal over the observed entries.

    Examples:
        Mask the second of two channels:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> obs_cov, cross_cov = jnp.ones((2, 2)), jnp.ones((3, 2))
        >>> obs_noise = 0.5 * jnp.eye(2)
        >>> y, y_hat = jnp.array([1.0, jnp.nan]), jnp.zeros(2)
        >>> mask = jnp.array([True, False])
        >>> masked = gaussx.masked_moment_inputs(
        ...     obs_cov, cross_cov, obs_noise, y, y_hat, mask
        ... )
        >>> obs_cov_eff, cross_cov_eff, obs_noise_eff, residual, n_missing = masked
        >>> obs_cov_eff.tolist(), obs_noise_eff.tolist()
        ([[1.0, 0.0], [0.0, 0.0]], [[0.5, 0.0], [0.0, 1.0]])
        >>> residual.tolist(), float(n_missing)
        ([1.0, 0.0], 1.0)
    """
    obs_size = y.shape[-1]
    # keep[i, j] is True only where both channels survive, so masked rows
    # and columns are cleared together.
    keep = einx.logical_and("row, col -> row col", mask, mask)

    # Zeroing row i of H would zero row i and column i of H Σ_x Hᵀ; do that
    # directly to the matched Σ_h.
    obs_cov_eff = jnp.where(keep, obs_cov, jnp.zeros_like(obs_cov))

    # Substitute a unit block into Σ_ε on the masked channels, so
    # Σ_y = blockdiag(Σ_y,obs, I) rather than becoming singular.
    unit_noise = jnp.eye(obs_size, dtype=obs_noise.dtype)
    obs_noise_eff = jnp.where(keep, obs_noise, unit_noise)

    # Column j of Σ_xy is what channel j uses to move the state; zero it and
    # the gain's column j vanishes with it.
    cross_cov_eff = einx.where(
        "obs, state obs, state obs -> state obs",
        mask,
        cross_cov,
        jnp.zeros_like(cross_cov),
    )

    # Mask y and μ_y separately rather than masking their difference: a
    # masked y entry is commonly NaN, and NaN in the discarded branch of a
    # where still poisons the reverse-mode gradient.
    masked_obs = jnp.where(mask, y, jnp.zeros_like(y))
    masked_predicted_obs = jnp.where(mask, y_hat, jnp.zeros_like(y_hat))
    # r = y − μ_y on the observed channels, 0 elsewhere
    residual = masked_obs - masked_predicted_obs

    # N_missing = D_y − N_observed
    n_observed = jnp.sum(mask.astype(y.dtype))
    n_missing = obs_size - n_observed
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
    """One moment-matched predict step through the dynamics f.

        μ_x,t|t−1 = E[f(x_t−1)]
        Σ_x,t|t−1 = Cov[f(x_t−1)] + Σ_η,t

    with x_t−1 ~ 𝒩(μ_x,t−1|t−1, Σ_x,t−1|t−1) and the expectations computed
    by ``integrator``'s moment transform. Exposed alongside
    `gaussx.nonlinear_kalman_update` so a caller can drive their own loop
    (an irregular time grid, a custom gating rule, a filter interleaved
    with something else) without reimplementing the moment transform.
    `gaussx.nonlinear_kalman_filter` is exactly a `jax.lax.scan` over these
    two.

    Algorithm:
        1. Push 𝒩(μ_x,t−1|t−1, Σ_x,t−1|t−1) through f with the integrator's
           moment transform, for E[f(x_t−1)] and Cov[f(x_t−1)].
        2. Add Σ_η,t and symmetrise.
        3. Optionally reject an indefinite Σ_x,t|t−1.

    Args:
        dynamics: Deterministic dynamics f, mapping a state to the next.
        mean: Filtered mean μ_x,t−1|t−1.
        cov: Filtered covariance Σ_x,t−1|t−1.
        process_noise: Additive model-error covariance Σ_η,t.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``; see
            `gaussx.nonlinear_kalman_filter` on why not ``alpha=1e-3``.
        validate: Check that Σ_x,t|t−1 is PSD. Defaults to
            ``not integrator.guarantees_psd(D_x)``; see
            `gaussx.nonlinear_kalman_filter`.

    Returns:
        Tuple ``(mean_pred, cov_pred)`` holding μ_x,t|t−1 and Σ_x,t|t−1.

    Raises:
        ValueError: If ``process_noise`` is not ``(D_x, D_x)``.
        EquinoxRuntimeError: If validation is on and Σ_x,t|t−1 is not PSD.

    Examples:
        Linear dynamics reproduce the Kalman predict step:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transition = jnp.array([[1.0, 0.1], [0.0, 1.0]])
        >>> mean, cov = gaussx.nonlinear_kalman_predict(
        ...     lambda x: transition @ x, jnp.ones(2), jnp.eye(2), 0.1 * jnp.eye(2)
        ... )
        >>> mean.shape, cov.shape
        ((2,), (2, 2))
        >>> exact = transition @ transition.T + 0.1 * jnp.eye(2)
        >>> bool(jnp.allclose(cov, exact, atol=1e-6))
        True
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)

    process_noise = _check_noise_shape(process_noise, mean.shape[-1], "process_noise")

    # The model noise is additive and independent of x, so it enters only
    # as an additive term on the covariance: the moment transform sees the
    # deterministic dynamics alone. No cross-covariance is needed here
    # (nothing is being conditioned on yet); the smoother re-runs the same
    # transform precisely to recover it.
    # μ_x,t|t−1 = E[f(x_t−1)], Cov[f(x_t−1)]
    mean_pred, dynamics_covariance, _ = moment_transform(
        dynamics, mean, cov, integrator=integrator
    )
    # Σ_x,t|t−1 = Cov[f(x_t−1)] + Σ_η,t
    cov_pred = dynamics_covariance + process_noise
    cov_pred = symmetrize(cov_pred)
    if not _resolve_validate(validate, integrator, mean.shape[-1]):
        return mean_pred, cov_pred

    # Validate here as well as after the update. A negative-weight rule can
    # return an indefinite Cov[f(x)] that a small Σ_η does not repair, and a
    # step-level False mask would then expose it directly as the filtered
    # covariance. Nothing downstream would complain: the next moment
    # transform tags it PSD, and the dense square-root path clips negative
    # eigenvalues to zero, silently altering the belief rather than
    # reporting it.
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
    obs_covariance: Float[Array, "M M"],
    cross_covariance: Float[Array, "N M"],
    residual: Float[Array, " M"],
    solver: AbstractSolverStrategy | None,
    validate: bool,
) -> tuple[
    Float[Array, "M M"], Float[Array, "N M"], Float[Array, " M"], Float[Array, ""]
]:
    """Gain G_t = Σ_xy,t Σ_y,t⁻¹, Σ_y,t⁻¹ r_t and log det Σ_y,t from one factor.

    Args:
        obs_covariance: Observation covariance Σ_y,t = Σ_h,t + Σ_ε,t.
        cross_covariance: Cross-covariance Σ_xy,t.
        residual: Residual r_t = y_t − μ_y,t.
        solver: Solver strategy for Σ_y,t; ``None`` factorises it once by
            Cholesky.
        validate: Reject a Σ_y,t that is not positive definite.

    Returns:
        Tuple ``(obs_covariance, gain, whitened_residual, logdet)``.
        ``obs_covariance`` is passed back because the validity check wraps
        it in `equinox.error_if`.
    """
    # A quadrature rule with negative weights can return an indefinite
    # Σ_h,t, and Σ_ε,t may be too small to repair it. Neither the update
    # nor the likelihood is defined then (the quadratic form can go
    # negative and the log-determinant becomes log |det Σ_y,t|), so this is
    # rejected rather than allowed to produce a plausible-looking but
    # meaningless number. With a positive-weight rule Σ_h,t is PSD, so
    # Σ_y,t is positive definite whenever Σ_ε,t is and the check is skipped
    # (gh-331).
    innovation_message = (
        "nonlinear_kalman_update: the innovation covariance S = Cov[h(x)] + R "
        "is not positive definite. A negative-weight quadrature rule (the "
        "scaled unscented transform, or the degree-5 cubature rule above "
        "N = 4) can return an indefinite Cov[h(x)]. Use a positive-weight "
        "rule such as CubatureIntegrator or UnscentedIntegrator(alpha=1.0), "
        "or increase obs_noise."
    )
    if solver is None:
        # One Cholesky of Σ_y,t serves the gain, the residual solve and the
        # log-determinant; a non-finite factor is the definiteness check.
        # Σ_y,t = L Lᵀ
        cholesky_factor = jnp.linalg.cholesky(obs_covariance)
        if validate:
            pivots_finite = jnp.all(jnp.isfinite(jnp.diagonal(cholesky_factor)))
            cholesky_factor = eqx.error_if(
                cholesky_factor, ~pivots_finite, innovation_message
            )
        # Σ_xyᵀ
        cross_covariance_transpose = rearrange(
            cross_covariance, "state obs -> obs state"
        )
        # G_tᵀ = Σ_y,t⁻¹ Σ_xy,tᵀ
        gain_transpose = jsla.cho_solve(
            (cholesky_factor, True), cross_covariance_transpose
        )
        # G_t = Σ_xy,t Σ_y,t⁻¹
        gain = rearrange(gain_transpose, "obs state -> state obs")
        # Σ_y,t⁻¹ r_t
        whitened_residual = jsla.cho_solve((cholesky_factor, True), residual)
        # log det Σ_y,t = 2 ∑ᵢ log Lᵢᵢ
        log_pivots = jnp.log(jnp.diagonal(cholesky_factor))
        logdet = 2.0 * jnp.sum(log_pivots)
    else:
        if validate:
            smallest_eigenvalue = jnp.linalg.eigvalsh(obs_covariance).min()
            obs_covariance = eqx.error_if(
                obs_covariance,
                smallest_eigenvalue <= 0.0,
                innovation_message,
            )
        obs_covariance_op = _symmetric(obs_covariance)
        # G_t = Σ_xy,t Σ_y,t⁻¹, row by row since Σ_y,t is symmetric
        gain = solve_rows(obs_covariance_op, cross_covariance, solver=solver)
        # Σ_y,t⁻¹ r_t
        whitened_residual = dispatch_solve(obs_covariance_op, residual, solver)
        # log det Σ_y,t
        logdet = dispatch_logdet(obs_covariance_op, solver)

    return obs_covariance, gain, whitened_residual, logdet


def _updated_covariance(
    predicted_covariance: Float[Array, "N N"],
    gain: Float[Array, "N M"],
    obs_covariance: Float[Array, "M M"],
    cross_covariance: Float[Array, "N M"],
    joseph: bool,
) -> Float[Array, "N N"]:
    """Updated covariance Σ_x,t|t, in Joseph or standard form.

        standard: Σ_x,t|t = Σ_x,t|t−1 − G_t Σ_y,t G_tᵀ
        Joseph:   Σ_x,t|t = (I − G_t H_t) Σ_x,t|t−1 (I − G_t H_t)ᵀ
                            + G_t (Σ_ε,t + Σ_lin,t) G_tᵀ

    with the statistical linearisation H_t = Σ_xy,tᵀ Σ_x,t|t−1⁻¹ and its
    residual covariance Σ_lin,t = Σ_h,t − H_t Σ_x,t|t−1 H_tᵀ, so that
    Σ_ε,t + Σ_lin,t = Σ_y,t − H_t Σ_xy,t.

    Args:
        predicted_covariance: Predicted covariance Σ_x,t|t−1.
        gain: Gain G_t.
        obs_covariance: Observation covariance Σ_y,t.
        cross_covariance: Cross-covariance Σ_xy,t.
        joseph: Use the Joseph form.

    Returns:
        The updated covariance Σ_x,t|t.
    """
    if joseph:
        # Joseph form needs an H_t, which a moment-matched filter does not
        # have. The right stand-in is the statistical linearisation of h
        # under the predicted belief (gaussx#161 section 3.3.5): writing
        # h(x) ≈ H_t x + b + e, the regression gain is
        # H_t = Σ_xy,tᵀ Σ_x,t|t−1⁻¹, which is exactly what
        # statistical_linear_regression returns as ``A``, and exactly H_t
        # when h is linear. So this reduces to the textbook Joseph update in
        # the linear case, while staying PSD for the merely-approximate gain
        # otherwise. Relative to the standard form the two differ by
        # G_t Σ_lin,t G_tᵀ, PSD and zero for affine h, so switching this
        # default cannot perturb the linear reduction.
        #
        # Σ_x,t|t−1 is legitimately singular for a deterministic initial
        # state, zero model noise or dimension-reducing dynamics, and a
        # well-posed solve returns NaN on those even though the update
        # itself is well defined (Σ_ε,t keeps Σ_y,t invertible). The
        # pseudo-inverse then gives the minimum-norm H_t, the natural
        # reading of the linearisation when the belief is confined to a
        # subspace; a Cholesky solve covers the nonsingular case at a
        # fraction of the cost.
        # H_tᵀ = Σ_x,t|t−1⁻¹ Σ_xy,t
        obs_operator_transpose = _solve_psd_or_lstsq(
            predicted_covariance, cross_covariance
        )
        # H_t = Σ_xy,tᵀ Σ_x,t|t−1⁻¹
        obs_operator = rearrange(obs_operator_transpose, "state obs -> obs state")

        # The noise of that regression is Σ_ε,t + Σ_lin,t, not Σ_ε,t:
        # linearising h leaves a residual e ~ 𝒩(0, Σ_lin,t) on top of the
        # measurement noise, and the Joseph form must be given the noise of
        # the model whose H_t it is using. Passing Σ_ε,t alone would return
        # the matched-joint posterior minus G_t Σ_lin,t G_tᵀ, systematically
        # overconfident on nonlinear maps. With the residual included the
        # two covariance forms agree to 2.8e-17, so Joseph is a numerically
        # safer route to the same answer rather than a different one.
        # H_t Σ_xy,t
        explained_covariance = einsum(
            obs_operator, cross_covariance, "obs state, state col -> obs col"
        )
        # Σ_ε,t + Σ_lin,t = Σ_y,t − H_t Σ_xy,t
        effective_noise = obs_covariance - explained_covariance
        effective_noise = symmetrize(effective_noise)
        # Σ_x,t|t = (I − G_t H_t) Σ_x,t|t−1 (I − G_t H_t)ᵀ
        #     + G_t (Σ_ε,t + Σ_lin,t) G_tᵀ
        updated_covariance = joseph_update(
            predicted_covariance, gain, obs_operator, effective_noise
        )
    else:
        # Correct and cheaper, but its PSD-ness relies on the moment triple
        # being mutually consistent (Σ_lin,t ⪰ 0), which a negative-weight
        # rule can violate.
        # G_t Σ_y,t
        gain_obs_covariance = einsum(
            gain, obs_covariance, "state obs, obs col -> state col"
        )
        # G_t Σ_y,t G_tᵀ
        correction = einsum(
            gain_obs_covariance, gain, "state col, other col -> state other"
        )
        # Σ_x,t|t = Σ_x,t|t−1 − G_t Σ_y,t G_tᵀ
        updated_covariance = predicted_covariance - correction
        updated_covariance = symmetrize(updated_covariance)

    return updated_covariance


def _reject_inconsistent_posterior(
    updated_covariance: Float[Array, "N N"],
) -> Float[Array, "N N"]:
    """Raise if Σ_x,t|t is not PSD, which means an inconsistent joint.

    A positive-definite Σ_y,t is not on its own enough: the joint over
    (x_t, h(x_t)) must be consistent. An inconsistent triple, with Σ_lin,t
    indefinite, which a negative-weight rule can produce even where Σ_y,t
    is fine, leaves the posterior indefinite. `_reject_indefinite` explains
    why the full spectrum is checked and why the threshold is not zero;
    genuine inconsistency is not marginal (its example sits at −0.28
    against a trace of 0.72, many orders above the bound).

    Args:
        updated_covariance: Updated covariance Σ_x,t|t.

    Returns:
        ``updated_covariance``, wrapped in `equinox.error_if`.
    """
    return _reject_indefinite(
        updated_covariance,
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
    """One moment-matched update step through the observation map h.

        G_t = Σ_xy,t Σ_y,t⁻¹,   Σ_y,t = Σ_h,t + Σ_ε,t
        μ_x,t|t = μ_x,t|t−1 + G_t (y_t − μ_y,t)

    where the integrator's moment transform at x_t ~ 𝒩(μ_x,t|t−1, Σ_x,t|t−1)
    gives the predicted observation μ_y,t = E[h(x_t)], its covariance
    Σ_h,t = Cov[h(x_t)] and the cross-covariance
    Σ_xy,t = E[(x_t − μ_x,t|t−1)(h(x_t) − μ_y,t)ᵀ]. The gain comes from
    Σ_xy,t directly, so no Jacobian J_h appears. See
    `gaussx.nonlinear_kalman_filter` for the meaning of ``joseph`` and the
    caveat on the returned log-likelihood.

    Algorithm:
        1. Moment-match h at the predicted belief for μ_y,t, Σ_h,t and
           Σ_xy,t; with a mask, make the masked channels inert
           (`gaussx.masked_moment_inputs`).
        2. Form Σ_y,t = Σ_h,t + Σ_ε,t and the residual r_t = y_t − μ_y,t.
        3. Factorise Σ_y,t once for G_t, Σ_y,t⁻¹ r_t and log det Σ_y,t.
        4. Update μ_x,t|t = μ_x,t|t−1 + G_t r_t, and Σ_x,t|t in Joseph or
           standard form.
        5. Score log p(y_t ∣ y_1:t−1)
           ≈ −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t + D_y log 2π),
           plus ½ log 2π per masked channel.

    Args:
        obs_fn: Observation map h, from a state to an observation.
        mean: Predicted mean μ_x,t|t−1.
        cov: Predicted covariance Σ_x,t|t−1.
        observation: Observation y_t.
        obs_noise: Observation-error covariance Σ_ε,t.
        integrator: Moment-matching rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``; see
            `gaussx.nonlinear_kalman_filter` on why not ``alpha=1e-3``.
        mask: Optional per-channel mask. ``False`` entries are marginalised
            out exactly and may be ``NaN`` in ``observation``.
        joseph: Use the Joseph-form covariance update. Defaults to ``True``.
        solver: Solver strategy for Σ_y,t. With ``None`` Σ_y,t is
            factorised once by Cholesky and shared by the gain, the residual
            solve and the log-determinant.
        validate: Check that Σ_y,t is positive definite and Σ_x,t|t is PSD.
            Defaults to ``not integrator.guarantees_psd(D_x)``; see
            `gaussx.nonlinear_kalman_filter`.

    Returns:
        Tuple ``(mean_upd, cov_upd, log_likelihood_increment)`` holding
        μ_x,t|t, Σ_x,t|t and log p(y_t ∣ y_1:t−1), the exact marginal over
        the observed channels for affine h.

    Raises:
        ValueError: If ``obs_noise`` is not ``(D_y, D_y)`` or ``mask`` is
            not ``(D_y,)``.
        EquinoxRuntimeError: If validation is on and Σ_y,t or Σ_x,t|t is
            indefinite.

    Examples:
        A linear observation reproduces the Kalman update:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> obs_operator = jnp.array([[1.0, 0.0]])
        >>> mean, cov, step = gaussx.nonlinear_kalman_update(
        ...     lambda x: obs_operator @ x,
        ...     jnp.zeros(2),
        ...     jnp.eye(2),
        ...     jnp.array([1.0]),
        ...     jnp.array([[1.0]]),
        ... )
        >>> bool(jnp.allclose(mean, jnp.array([0.5, 0.0]), atol=1e-6))
        True

        A fully masked observation leaves the belief unchanged:

        >>> mean, cov, step = gaussx.nonlinear_kalman_update(
        ...     lambda x: obs_operator @ x,
        ...     jnp.zeros(2),
        ...     jnp.eye(2),
        ...     jnp.array([jnp.nan]),
        ...     jnp.array([[1.0]]),
        ...     mask=jnp.array([False]),
        ... )
        >>> bool(jnp.allclose(cov, jnp.eye(2), atol=1e-6)), float(step)
        (True, 0.0)
    """
    if integrator is None:
        integrator = UnscentedIntegrator(alpha=1.0)
    validate = _resolve_validate(validate, integrator, mean.shape[-1])

    obs_size = observation.shape[-1]
    obs_noise = _check_noise_shape(obs_noise, obs_size, "obs_noise")
    if mask is not None and jnp.shape(mask) != (obs_size,):
        # A (1,) mask would broadcast across every channel: False would
        # suppress them all and return an identity update, True would enable
        # them all, in either case silently.
        msg = f"mask must have shape ({obs_size},); got {jnp.shape(mask)}."
        raise ValueError(msg)

    # μ_y,t = E[h(x_t)], Σ_h,t = Cov[h(x_t)], Σ_xy,t
    predicted_obs, matched_obs_covariance, matched_cross_covariance = moment_transform(
        obs_fn, mean, cov, integrator=integrator
    )

    if mask is None:
        obs_moment_covariance = matched_obs_covariance
        cross_covariance = matched_cross_covariance
        obs_noise_covariance = obs_noise
        # r_t = y_t − μ_y,t
        residual = observation - predicted_obs
        n_missing = jnp.zeros((), dtype=cov.dtype)
    else:
        # Rewrite the matched moments so masked channels carry no
        # information; see `gaussx.masked_moment_inputs`.
        (
            obs_moment_covariance,
            cross_covariance,
            obs_noise_covariance,
            residual,
            n_missing,
        ) = masked_moment_inputs(
            matched_obs_covariance,
            matched_cross_covariance,
            obs_noise,
            observation,
            predicted_obs,
            mask,
        )

    # Σ_y,t = Σ_h,t + Σ_ε,t, symmetrised because Σ_h,t is a weighted
    # outer-product sum, which drifts asymmetric
    obs_covariance = obs_moment_covariance + obs_noise_covariance
    obs_covariance = symmetrize(obs_covariance)

    # G_t = Σ_xy,t Σ_y,t⁻¹, Σ_y,t⁻¹ r_t, log det Σ_y,t
    obs_covariance, gain, whitened_residual, logdet = _innovation_gain(
        obs_covariance, cross_covariance, residual, solver, validate
    )

    # μ_x,t|t = μ_x,t|t−1 + G_t r_t
    mean_upd = mean + gain @ residual

    cov_upd = _updated_covariance(cov, gain, obs_covariance, cross_covariance, joseph)

    log_likelihood_step = _log_likelihood_increment(
        residual, whitened_residual, logdet, obs_size, n_missing
    )
    if not validate:
        return mean_upd, cov_upd, log_likelihood_step
    cov_upd = _reject_inconsistent_posterior(cov_upd)

    return mean_upd, cov_upd, log_likelihood_step


def _log_likelihood_increment(
    residual: Float[Array, " M"],
    whitened_residual: Float[Array, " M"],
    logdet: Float[Array, ""],
    obs_size: int,
    n_missing: Float[Array, ""],
) -> Float[Array, ""]:
    """Log-likelihood term of one step, over the observed channels.

        log p(y_t ∣ y_1:t−1) ≈ −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t
            + D_y log 2π) + ½ N_missing log 2π

    An approximation, not the exact marginal: Σ_y,t is the matched
    observation covariance. Exact when the maps are affine. Each masked
    channel contributed a dummy unit block to Σ_y,t, worth −½ log 2π of the
    full-vector density; adding it back makes the result the exact marginal
    over the observed entries, independent of the dummy block's variance.

    Args:
        residual: Residual r_t.
        whitened_residual: Σ_y,t⁻¹ r_t.
        logdet: log det Σ_y,t.
        obs_size: Observation length D_y, masked channels included.
        n_missing: Number of masked channels N_missing.

    Returns:
        The log-likelihood term log p(y_t ∣ y_1:t−1).
    """
    # r_tᵀ Σ_y,t⁻¹ r_t
    mahalanobis = residual @ whitened_residual
    # log p(y_t ∣ y_1:t−1) ≈ −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t
    #     + D_y log 2π) + ½ N_missing log 2π
    return (
        -0.5 * (mahalanobis + logdet + obs_size * _LOG_2PI) + 0.5 * n_missing * _LOG_2PI
    )
