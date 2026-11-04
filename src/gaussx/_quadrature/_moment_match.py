"""EP moment matching: tilted log-normaliser and its cavity-mean derivatives."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._inv import inv
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState


def _tilted_weights(
    log_lik_fn: Callable[[Float[Array, " D"]], Float[Array, ""]],
    chi: Float[Array, "P D"],
    w_m: Float[Array, " P"],
    power: float,
) -> tuple[Float[Array, " P"], Float[Array, ""]]:
    """Shifted tilted weights of a point-based rule.

        w̃ᵢ = wᵢ p(y ∣ fᵢ)ᵃ / exp(s),    s = maxᵢ a log p(y ∣ fᵢ)

    Shared by `moment_match` and `gaussx.ep_tilted_moments`. The shift s
    cancels in every normalised ratio and is differentiated as a constant.
    Weights keep the rule's sign.

    Args:
        log_lik_fn: Maps a latent f to the scalar log-likelihood log p(y ∣ f).
        chi: Points fᵢ of the rule.
        w_m: Mean weights wᵢ of the rule.
        power: Power-EP exponent a.

    Returns:
        Tuple ``(tilted_weights, shift)`` holding w̃ᵢ and s.

    Raises:
        ValueError: If ``log_lik_fn`` does not return a scalar per point.
    """
    log_likelihoods = jax.vmap(log_lik_fn)(chi)
    # a log p(y ∣ fᵢ)
    tempered = power * log_likelihoods
    if tempered.ndim != 1:
        msg = (
            f"log_lik_fn must return a scalar per point, got trailing "
            f"shape {tempered.shape[1:]}."
        )
        raise ValueError(msg)
    # s = maxᵢ a log p(y ∣ fᵢ)
    shift = jax.lax.stop_gradient(jnp.max(tempered))
    # p(y ∣ fᵢ)ᵃ / exp(s)
    shifted_likelihoods = jnp.exp(tempered - shift)
    # w̃ᵢ = wᵢ p(y ∣ fᵢ)ᵃ / exp(s)
    return w_m * shifted_likelihoods, shift


class MomentMatchResult(eqx.Module):
    """Tilted log-normaliser and its derivatives with respect to the cavity mean.

        log Z,    g = ∂ log Z / ∂μ_cav,    H = ∂² log Z / ∂μ_cav²

    Attributes:
        log_Z: Tilted log-normaliser log Z.
        d_log_Z: Gradient g.
        d2_log_Z: Hessian H.

    Examples:
        >>> import jax.numpy as jnp
        >>> result = MomentMatchResult(
        ...     log_Z=jnp.zeros(()), d_log_Z=jnp.zeros(2), d2_log_Z=jnp.eye(2)
        ... )
        >>> result.d2_log_Z.shape
        (2, 2)
    """

    log_Z: Float[Array, ""]
    d_log_Z: Float[Array, " D"]
    d2_log_Z: Float[Array, "D D"]


def moment_match(
    log_lik_fn: Callable[[Float[Array, " D"]], Float[Array, ""]],
    state: GaussianState,
    integrator: AbstractIntegrator,
    *,
    power: float = 1.0,
) -> MomentMatchResult:
    """EP moment matching: tilted log-normaliser and its first two derivatives.

        log Z = log ∫ p(y ∣ f)ᵃ 𝒩(f ∣ μ_cav, Σ_cav) df
        g = ∂ log Z / ∂μ_cav = Σ_cav⁻¹ 𝔼_tilt[f − μ_cav]
        H = ∂² log Z / ∂μ_cav² = Σ_cav⁻¹ (Σ_tilt − Σ_cav) Σ_cav⁻¹

    This is the EP quantity, *not* 𝔼_q[log p(y ∣ f)] (which is what
    variational inference needs — see `gaussx.expected_log_likelihood`).

    The tilted moments follow directly from the two derivatives:

        μ_tilt = μ_cav + Σ_cav g,    Σ_tilt = Σ_cav + Σ_cav H Σ_cav

    The Gaussian site is the natural-parameter *difference* between the
    tilted distribution and the cavity, which reduces to

        Λ_site = Σ_tilt⁻¹ − Σ_cav⁻¹ = −(I + H Σ_cav)⁻¹ H
        Λμ_site = Σ_tilt⁻¹ μ_tilt − Σ_cav⁻¹ μ_cav
                = (I + H Σ_cav)⁻¹ (g − H μ_cav)

    in the same ``nat2 = +Λ`` convention `gaussx.cavity_distribution` uses.
    That site approximates p(y ∣ f)ᵃ, so divide both naturals by a to recover
    the site for p(y ∣ f) itself.

    Do **not** feed (g, −H) in as the site: it drops the −H μ_cav term and
    the (I + H Σ_cav)⁻¹ factor. Nor is `gaussx.newton_update` the right
    conversion here — it yields (g − H μ_cav, −H), which is the local-Newton
    site, i.e. only the first-order approximation to the EP site above.

    All three outputs come from a **single** pass over the quadrature
    points: applying Stein's lemma to the derivatives of the Gaussian
    density turns them into reweightings of the same likelihood
    evaluations, so no autodiff runs through the integrator and no extra
    likelihood evaluations are needed.

    The sum over points is evaluated with the log-sum-exp shift, so
    likelihoods that underflow in the raw scale are handled. Note that
    quadrature weights may be negative (the scaled unscented transform and
    the degree-5 cubature rule both have negative weights in places), so
    Z is not guaranteed positive for a badly matched rule; prefer a rule
    whose points cover the likelihood's mass.

    Algorithm:
        1. Tilt and shift the rule's weights: w̃ᵢ.
        2. Normalise: log Z = s + log ∑ᵢ w̃ᵢ.
        3. Precision-weight the points: uᵢ = Σ_cav⁻¹ (fᵢ − μ_cav).
        4. Reweight for g and H.

    Args:
        log_lik_fn: Maps a latent f to the scalar log-likelihood
            log p(y ∣ f).
        state: Cavity distribution 𝒩(μ_cav, Σ_cav). Its covariance must be
            symmetric.
        integrator: Point-based rule supplying the points and weights,
            e.g. `gaussx.FifthOrderCubatureIntegrator`.
        power: Power-EP exponent a. ``1.0`` is standard EP; a < 1 corresponds
            to fractional EP / alpha-divergence minimisation. Documented for
            a in (0, 1].

    Returns:
        `MomentMatchResult` with ``log_Z``, ``d_log_Z``, ``d2_log_Z``.

    Raises:
        NotImplementedError: If ``integrator`` is not point-based.
        ValueError: If ``log_lik_fn`` does not return a scalar per point.

    Examples:
        A flat likelihood has Z = 1 and no curvature:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.MatrixLinearOperator(
        ...     jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> result = moment_match(
        ...     lambda f: 0.0 * jnp.sum(f), state, gaussx.CubatureIntegrator()
        ... )
        >>> bool(jnp.allclose(result.log_Z, 0.0, atol=1e-5))
        True
        >>> result.d2_log_Z.shape
        (2, 2)
    """
    points, mean_weights, _ = integrator.points_and_weights(state)
    # w̃ᵢ = wᵢ p(y ∣ fᵢ)ᵃ / exp(s), signed
    tilted_weights, shift = _tilted_weights(log_lik_fn, points, mean_weights, power)
    # ∑ᵢ w̃ᵢ
    shifted_normaliser = jnp.sum(tilted_weights)

    # log Z = s + log ∑ᵢ w̃ᵢ
    log_normaliser = shift + jnp.log(shifted_normaliser)

    # jnp, not einx, for the broadcasts and reductions below: einx cannot
    # resolve an empty latent (D = 0), which returns empty derivatives here today.
    # fᵢ − μ_cav
    anomalies = points - state.mean[None, :]
    # uᵢ = Σ_cav⁻¹ (fᵢ − μ_cav), row by row since Σ_cav is symmetric
    precision_weighted = solve_rows(state.cov, anomalies)

    # g = ∑ᵢ w̃ᵢ uᵢ / ∑ᵢ w̃ᵢ
    gradient = (
        jnp.sum(tilted_weights[:, None] * precision_weighted, axis=0)
        / shifted_normaliser
    )

    # ∑ᵢ w̃ᵢ uᵢ uᵢᵀ / ∑ᵢ w̃ᵢ
    second_moment = (
        jnp.sum(
            tilted_weights[:, None, None]
            * (precision_weighted[:, :, None] * precision_weighted[:, None, :]),
            axis=0,
        )
        / shifted_normaliser
    )
    # Σ_cav⁻¹
    cavity_precision = inv(state.cov).as_matrix()

    # H = ∑ᵢ w̃ᵢ uᵢ uᵢᵀ / ∑ᵢ w̃ᵢ − Σ_cav⁻¹ − g gᵀ
    hessian = symmetrize(
        second_moment - cavity_precision - jnp.outer(gradient, gradient)
    )

    return MomentMatchResult(log_Z=log_normaliser, d_log_Z=gradient, d2_log_Z=hessian)
