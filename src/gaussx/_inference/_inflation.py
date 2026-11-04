"""Ensemble inflation: multiplicative, RTPP and RTPS."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def inflate_multiplicative(
    ensemble: Float[Array, "J N"],
    factor: float | Float[Array, ""],
) -> Float[Array, "J N"]:
    """Multiplicative ensemble inflation about the sample mean.

        x⁽ᵉ⁾ ← x̄ + ζ (x⁽ᵉ⁾ − x̄)

    Restores the spread lost to sampling error or model collapse by scaling
    the anomalies.

    Algorithm:
        1. Take the sample mean x̄ over the members.
        2. Scale each anomaly x⁽ᵉ⁾ − x̄ by ζ and add x̄ back.

    Args:
        ensemble: Ensemble members x⁽ᵉ⁾, one row per member.
        factor: Inflation factor ζ ≥ 1, e.g. 1.02 to 1.10.

    Returns:
        The inflated ensemble, with the same sample mean.

    Examples:
        Doubling the spread of two scalar members:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> ensemble = jnp.array([[0.0], [2.0]])
        >>> gaussx.inflate_multiplicative(ensemble, 2.0).tolist()
        [[-1.0], [3.0]]

        A factor of one changes nothing:

        >>> inflated = gaussx.inflate_multiplicative(ensemble, 1.0)
        >>> bool(jnp.allclose(inflated, ensemble))
        True
    """
    # jnp, not einx: an empty |ℰ| or D_x returns an empty ensemble here today,
    # and einx raises AxisSizeError on a zero-length axis.
    # x̄ = (1 / |ℰ|) ∑ₑ x⁽ᵉ⁾
    mean = jnp.mean(ensemble, axis=0, keepdims=True)
    # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
    anomalies = ensemble - mean
    # ζ x̃⁽ᵉ⁾
    scaled_anomalies = factor * anomalies
    # x⁽ᵉ⁾ ← x̄ + ζ x̃⁽ᵉ⁾
    return mean + scaled_anomalies


def inflate_rtpp(
    posterior: Float[Array, "J N"],
    prior: Float[Array, "J N"],
    alpha: float | Float[Array, ""],
) -> Float[Array, "J N"]:
    """Relaxation to prior perturbations (RTPP).

        x̃⁽ᵉ⁾_t|t ← (1 − α) x̃⁽ᵉ⁾_t|t + α x̃⁽ᵉ⁾_t|t−1

    Relaxes the posterior anomalies toward the prior ones while keeping the
    posterior mean. Each anomaly is taken about its own ensemble's sample mean.

    Algorithm:
        1. Take the anomalies of each ensemble about its own sample mean.
        2. Blend them with weights 1 − α and α.
        3. Add the blend to the posterior sample mean.

    Args:
        posterior: Analysis ensemble x⁽ᵉ⁾_t|t.
        prior: Forecast ensemble x⁽ᵉ⁾_t|t−1.
        alpha: Relaxation weight α ∈ [0, 1].

    Returns:
        The relaxed analysis ensemble, with the posterior's sample mean.

    Examples:
        Halfway back to a wider prior:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> posterior = jnp.array([[0.0], [2.0]])
        >>> prior = jnp.array([[-3.0], [3.0]])
        >>> gaussx.inflate_rtpp(posterior, prior, 0.5).tolist()
        [[-1.0], [3.0]]

        A weight of zero keeps the posterior:

        >>> relaxed = gaussx.inflate_rtpp(posterior, prior, 0.0)
        >>> bool(jnp.allclose(relaxed, posterior))
        True

    References:
        Zhang et al. (2004).
    """
    # jnp, not einx: an empty |ℰ| or D_x returns an empty ensemble here today,
    # and einx raises AxisSizeError on a zero-length axis.
    # x̄_t|t = (1 / |ℰ|) ∑ₑ x⁽ᵉ⁾_t|t
    posterior_mean = jnp.mean(posterior, axis=0, keepdims=True)
    # x̃⁽ᵉ⁾_t|t = x⁽ᵉ⁾_t|t − x̄_t|t
    posterior_anomalies = posterior - posterior_mean
    # x̃⁽ᵉ⁾_t|t−1 = x⁽ᵉ⁾_t|t−1 − x̄_t|t−1
    prior_anomalies = prior - jnp.mean(prior, axis=0, keepdims=True)
    # (1 − α) x̃⁽ᵉ⁾_t|t
    kept_anomalies = (1.0 - alpha) * posterior_anomalies
    # x̄_t|t + (1 − α) x̃⁽ᵉ⁾_t|t
    relaxed = posterior_mean + kept_anomalies
    # x⁽ᵉ⁾ ← x̄_t|t + (1 − α) x̃⁽ᵉ⁾_t|t + α x̃⁽ᵉ⁾_t|t−1
    return relaxed + alpha * prior_anomalies


def inflate_rtps(
    posterior: Float[Array, "J N"],
    prior: Float[Array, "J N"],
    beta: float | Float[Array, ""],
    eps: float = 1e-12,
) -> Float[Array, "J N"]:
    """Relaxation to prior spread (RTPS).

        x̃⁽ᵉ⁾_t|t ← x̃⁽ᵉ⁾_t|t ((1 − β) + β σ_x,t|t−1 / σ_x,t|t)

    Scales each posterior anomaly, per coordinate, so the analysis spread
    relaxes back toward the prior spread. σ_x is the per-coordinate ensemble
    standard deviation with the 1 / |ℰ| divisor.

    Algorithm:
        1. Take the posterior sample mean and anomalies.
        2. Take the per-coordinate spread of both ensembles.
        3. Form the scale (1 − β) + β σ_x,t|t−1 / (σ_x,t|t + floor).
        4. Scale the anomalies and add the posterior sample mean back.

    Args:
        posterior: Analysis ensemble x⁽ᵉ⁾_t|t.
        prior: Forecast ensemble x⁽ᵉ⁾_t|t−1.
        beta: Relaxation weight β ∈ [0, 1].
        eps: Floor added to σ_x,t|t to avoid division by zero.

    Returns:
        The spread-restored analysis ensemble, with the posterior's sample
        mean.

    Examples:
        Halfway back to a spread three times wider:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> posterior = jnp.array([[-1.0], [1.0]])
        >>> prior = jnp.array([[-3.0], [3.0]])
        >>> relaxed = gaussx.inflate_rtps(posterior, prior, 0.5)
        >>> bool(jnp.allclose(relaxed, jnp.array([[-2.0], [2.0]])))
        True

        A weight of zero keeps the posterior:

        >>> relaxed = gaussx.inflate_rtps(posterior, prior, 0.0)
        >>> bool(jnp.allclose(relaxed, posterior))
        True

    References:
        Whitaker & Hamill (2012).
    """
    # jnp, not einx: an empty |ℰ| or D_x returns an empty ensemble here today,
    # and einx raises AxisSizeError on a zero-length axis.
    # x̄_t|t = (1 / |ℰ|) ∑ₑ x⁽ᵉ⁾_t|t
    posterior_mean = jnp.mean(posterior, axis=0, keepdims=True)
    # x̃⁽ᵉ⁾_t|t = x⁽ᵉ⁾_t|t − x̄_t|t
    posterior_anomalies = posterior - posterior_mean
    # σ_x,t|t
    posterior_spread = jnp.std(posterior, axis=0)
    # σ_x,t|t−1
    prior_spread = jnp.std(prior, axis=0)
    # (1 − β) + β σ_x,t|t−1 / (σ_x,t|t + floor)
    scale = (1.0 - beta) + beta * prior_spread / (posterior_spread + eps)
    # x̃⁽ᵉ⁾_t|t ((1 − β) + β σ_x,t|t−1 / (σ_x,t|t + floor))
    scaled_anomalies = posterior_anomalies * scale[None, :]
    # x⁽ᵉ⁾ ← x̄_t|t + scaled anomaly
    return posterior_mean + scaled_anomalies
