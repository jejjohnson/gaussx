"""EP tilted moments of a scalar site under any point-based rule."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._quadrature._gauss_hermite import GaussHermiteIntegrator
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._moment_match import _tilted_weights
from gaussx._quadrature._types import GaussianState


def ep_tilted_moments(
    log_lik_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    cav_mean: Float[Array, " *batch"],
    cav_var: Float[Array, " *batch"],
    *,
    order: int | None = None,
    integrator: AbstractIntegrator | None = None,
    power: float = 1.0,
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    """Moments of the scalar EP tilted distribution, elementwise over a batch.

        p_tilt(f) ∝ p(y ∣ f)ᵃ 𝒩(f ∣ μ_cav, σ²_cav)
        w̃ᵢ = wᵢ p(y ∣ fᵢ)ᵃ
        μ_tilt = ∑ᵢ w̃ᵢ fᵢ / ∑ᵢ w̃ᵢ
        σ²_tilt = ∑ᵢ w̃ᵢ (fᵢ − μ_tilt)² / ∑ᵢ w̃ᵢ

    with the points fᵢ and mean weights wᵢ of a point-based rule applied to
    the cavity. These are the same tilted moments as μ_cav + σ²_cav g and
    σ²_cav + σ²_cav H σ²_cav from `gaussx.moment_match` (Stein's lemma), and
    share its log-sum-exp shifted weights. The default rule is Gauss-Hermite
    of ``order`` points, the behaviour before ``integrator`` existed.

    A site whose cavity variance is not positive -- EP's usual failure mode,
    when the site holds more precision than the current posterior -- has no
    tilted distribution. Its cavity moments are returned unchanged, so the
    moment-matched site update 1/σ²_tilt − 1/σ²_cav is zero and nothing
    ``NaN`` reaches the site parameters; skip or damp such sites in the
    caller. Valid sites are unaffected, and gradients stay finite for both.

    Algorithm:
        1. Pass a site with σ²_cav ≤ 0 through as (μ_cav, σ²_cav).
        2. Take the points and weights of the rule on 𝒩(μ_cav, σ²_cav).
        3. Shift and tilt the weights: w̃ᵢ.
        4. Take the tilted moments, flooring σ²_tilt at machine epsilon
           times σ²_cav.

    Args:
        log_lik_fn: Maps a scalar latent f to the scalar log-likelihood
            log p(y ∣ f).
        cav_mean: Cavity means μ_cav.
        cav_var: Cavity variances σ²_cav. Non-positive entries are passed
            through as described above.
        order: Number of Gauss-Hermite points of the default rule. Default
            ``20``. Only valid when ``integrator`` is ``None``.
        integrator: Point-based rule (e.g. `gaussx.UnscentedIntegrator`,
            `gaussx.CubatureIntegrator`) applied to each 1-D cavity. Default
            ``GaussHermiteIntegrator(order)``.
        power: Power-EP exponent a; ``1.0`` is standard EP.

    Returns:
        Tuple ``(tilted_mean, tilted_var)`` holding μ_tilt and σ²_tilt.

    Raises:
        ValueError: If both ``integrator`` and ``order`` are given.
        NotImplementedError: If ``integrator`` is not point-based.

    Examples:
        A flat likelihood leaves the cavity unchanged:

        >>> import jax.numpy as jnp
        >>> tilted_mean, tilted_var = ep_tilted_moments(
        ...     lambda f: 0.0 * f, jnp.array([0.5, 1.0]), jnp.array([1.0, 2.0])
        ... )
        >>> bool(jnp.allclose(tilted_mean, jnp.array([0.5, 1.0])))
        True

        An invalid site passes through:

        >>> tilted_mean, tilted_var = ep_tilted_moments(
        ...     lambda f: -(f**2), jnp.array([0.5]), jnp.array([-1.0])
        ... )
        >>> bool(jnp.allclose(tilted_var, jnp.array([-1.0])))
        True
    """
    if integrator is None:
        integrator = GaussHermiteIntegrator(order=20 if order is None else order)
    elif order is not None:
        msg = "Pass either `integrator` or `order` (Gauss-Hermite), not both."
        raise ValueError(msg)

    # Cavity (and so the rule's points) in at least float32, without
    # promoting a float32 cavity to float64 under x64. promote_types (not
    # result_type with a float32 operand) so a weakly typed float64 cavity,
    # e.g. jnp.array(0.3), stays float64.
    dtype = jnp.promote_types(jnp.result_type(cav_mean, cav_var), jnp.float32)
    cav_mean = jnp.asarray(cav_mean, dtype=dtype)
    cav_var = jnp.asarray(cav_var, dtype=dtype)

    def _quadrature_moments(
        cavity_mean: Float[Array, ""], cavity_variance: Float[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        state = GaussianState(
            mean=jnp.atleast_1d(cavity_mean),
            cov=lx.DiagonalLinearOperator(jnp.atleast_1d(cavity_variance)),
        )
        points, mean_weights, _ = integrator.points_and_weights(state)
        # w̃ᵢ = wᵢ p(y ∣ fᵢ)ᵃ / exp(shift)
        tilted_weights, _ = _tilted_weights(
            lambda x: log_lik_fn(x[0]), points, mean_weights, power
        )
        nodes = points[:, 0]
        # ∑ᵢ w̃ᵢ
        normaliser = jnp.sum(tilted_weights)
        # μ_tilt = ∑ᵢ w̃ᵢ fᵢ / ∑ᵢ w̃ᵢ
        tilted_mean = jnp.sum(tilted_weights * nodes) / normaliser
        # σ²_tilt = ∑ᵢ w̃ᵢ (fᵢ − μ_tilt)² / ∑ᵢ w̃ᵢ
        tilted_variance = (
            jnp.sum(tilted_weights * (nodes - tilted_mean) ** 2) / normaliser
        )
        # Floor relative to the cavity, not in absolute units: a tilted
        # variance below eps * cavity variance is quadrature round-off, and
        # an absolute floor would inflate small-scale problems.
        tilted_variance = jnp.maximum(
            tilted_variance, jnp.finfo(tilted_variance.dtype).eps * cavity_variance
        )
        return tilted_mean, tilted_variance

    def _compute_moments(
        cavity_mean: Float[Array, ""], cavity_variance: Float[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        # Double where: the quadrature never sees a negative variance, so
        # neither the value nor the gradient of an invalid site is NaN.
        valid = cavity_variance > 0
        safe_variance = jnp.where(valid, cavity_variance, 1.0)
        tilted_mean, tilted_variance = _quadrature_moments(cavity_mean, safe_variance)
        return (
            jnp.where(valid, tilted_mean, cavity_mean),
            jnp.where(valid, tilted_variance, cavity_variance),
        )

    return jnp.vectorize(_compute_moments)(cav_mean, cav_var)
