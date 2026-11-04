"""Ensemble covariance and cross-covariance statistics."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._low_rank_update import LowRankUpdate


def ensemble_covariance(
    particles: Float[Array, "J N"],
    *,
    bessel: bool = False,
) -> LowRankUpdate:
    """Empirical covariance from an ensemble as a low-rank operator.

        Σ̂_x = c X̃ᵀ X̃,   x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄,   x̄ = (1 / |ℰ|) ∑ₑ x⁽ᵉ⁾

    with c = 1 / |ℰ| when ``bessel=False`` (maximum likelihood) and
    c = 1 / (|ℰ| − 1) when ``bessel=True`` (unbiased, the ensemble Kalman
    convention). The result has rank ≤ |ℰ| − 1 and never materialises the
    (D_x, D_x) matrix, which pays off when |ℰ| ≪ D_x.

    Algorithm:
        1. Check that the ensemble is large enough for the divisor.
        2. Centre the members on their sample mean x̄ to get the anomalies X̃.
        3. Scale them into the low-rank factor U = √c X̃ᵀ.
        4. Wrap U over a zero diagonal base, so that Σ̂_x = U Uᵀ.

    Args:
        particles: Ensemble members x⁽ᵉ⁾, one row per member.
        bessel: If ``True``, use the divisor |ℰ| − 1 from the ensemble Kalman
            literature. This lower-level helper defaults to ``False`` for
            backwards compatibility; `ensemble_kalman_gain` defaults to
            ``True`` for the EnKF convention.

    Returns:
        A `LowRankUpdate` holding Σ̂_x, with a zero base and an |ℰ|-column
        low-rank factor.

    Raises:
        ValueError: If the ensemble is empty, or if ``bessel`` is set with
            |ℰ| < 2.

    Note:
        The ``bessel`` default differs across this module: these two
        covariance helpers default to ``bessel=False`` (1 / |ℰ|), while
        `ensemble_kalman_gain`, `localized_kalman_gain`, `enkf_analysis`,
        `eki_step` and `discrepancy_step_size` default to ``True``
        (1 / (|ℰ| − 1)) and `etkf_transform` is 1 / (|ℰ| − 1) throughout. A
        gain assembled by hand from these helpers with their defaults
        therefore differs from `ensemble_kalman_gain`'s; pass ``bessel=True``
        to match it.

    Examples:
        Two members in two dimensions, with the unbiased divisor:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> particles = jnp.array([[1.0, 0.0], [-1.0, 0.0]])
        >>> covariance = gaussx.ensemble_covariance(particles, bessel=True)
        >>> expected = jnp.array([[2.0, 0.0], [0.0, 0.0]])
        >>> bool(jnp.allclose(covariance.as_matrix(), expected))
        True

        The default maximum-likelihood divisor halves it here:

        >>> halved = gaussx.ensemble_covariance(particles).as_matrix()
        >>> bool(jnp.allclose(halved, 0.5 * expected))
        True

        One member has no unbiased covariance:

        >>> gaussx.ensemble_covariance(
        ...     particles[:1], bessel=True
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Bessel correction requires J >= 2 particles
    """
    ensemble_size, state_size = particles.shape
    _check_ensemble_size(ensemble_size, bessel)
    # jnp, not einx: D_x = 0 returns an empty operator here today, and einx
    # raises AxisSizeError on a zero-length axis.
    # x̄ = (1 / |ℰ|) ∑ₑ x⁽ᵉ⁾
    mean = jnp.mean(particles, axis=0)
    # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
    anomalies = particles - mean[None, :]

    divisor = ensemble_size - 1 if bessel else ensemble_size
    # U = √c X̃ᵀ
    low_rank_factor = anomalies.T / jnp.sqrt(divisor)

    base = lx.DiagonalLinearOperator(jnp.zeros(state_size, dtype=particles.dtype))
    return LowRankUpdate(base, low_rank_factor)


def ensemble_cross_covariance(
    particles_theta: Float[Array, "J N"],
    particles_G: Float[Array, "J M"],
    *,
    bessel: bool = False,
) -> Float[Array, "N M"]:
    """Cross-covariance between two ensembles with the same members.

        Σ̂_xy = c ∑ₑ x̃⁽ᵉ⁾ h̃⁽ᵉ⁾ᵀ = c X̃ᵀ H̃

    with c = 1 / |ℰ| by default or c = 1 / (|ℰ| − 1) when ``bessel=True``. The
    tilde marks the anomaly from each ensemble's own sample mean.

    Algorithm:
        1. Check that the ensemble is large enough for the divisor.
        2. Centre each ensemble on its own sample mean.
        3. Contract the anomalies over the members and divide by the divisor.

    Args:
        particles_theta: First ensemble x⁽ᵉ⁾, one row per member.
        particles_G: Second ensemble h⁽ᵉ⁾, one row per member.
        bessel: If ``True``, use the divisor |ℰ| − 1 used by the ensemble
            Kalman recipes. This lower-level helper defaults to ``False`` for
            backwards compatibility; `ensemble_kalman_gain` defaults to
            ``True`` for the EnKF convention.

    Returns:
        The cross-covariance Σ̂_xy.

    Raises:
        ValueError: If the first ensemble is empty, or if ``bessel`` is set
            with |ℰ| < 2.

    Note:
        The ``bessel`` default differs across this module: these two
        covariance helpers default to ``bessel=False`` (1 / |ℰ|), while
        `ensemble_kalman_gain`, `localized_kalman_gain`, `enkf_analysis`,
        `eki_step` and `discrepancy_step_size` default to ``True``
        (1 / (|ℰ| − 1)) and `etkf_transform` is 1 / (|ℰ| − 1) throughout. A
        gain assembled by hand from these helpers with their defaults
        therefore differs from `ensemble_kalman_gain`'s; pass ``bessel=True``
        to match it.

    Examples:
        Perfectly correlated scalars, with the maximum-likelihood divisor:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> state = jnp.array([[1.0], [-1.0]])
        >>> gaussx.ensemble_cross_covariance(state, 2.0 * state).tolist()
        [[2.0]]

        The unbiased divisor doubles it for two members:

        >>> gaussx.ensemble_cross_covariance(
        ...     state, 2.0 * state, bessel=True
        ... ).tolist()
        [[4.0]]

        An ensemble against itself is `ensemble_covariance`:

        >>> particles = jnp.array([[1.0, 0.0], [-1.0, 2.0], [0.0, 1.0]])
        >>> cross = gaussx.ensemble_cross_covariance(particles, particles)
        >>> dense = gaussx.ensemble_covariance(particles).as_matrix()
        >>> bool(jnp.allclose(cross, dense))
        True
    """
    ensemble_size = particles_theta.shape[0]
    _check_ensemble_size(ensemble_size, bessel)
    # jnp, not einx: an empty D_x or D_y returns an empty matrix here today, and
    # a member mismatch reaches dot_general's error; einx raises AxisSizeError
    # for both.
    # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
    state_anomalies = particles_theta - jnp.mean(particles_theta, axis=0, keepdims=True)
    # h̃⁽ᵉ⁾ = h⁽ᵉ⁾ − h̄
    obs_anomalies = particles_G - jnp.mean(particles_G, axis=0, keepdims=True)
    divisor = ensemble_size - 1 if bessel else ensemble_size
    # X̃ᵀ H̃
    scatter = state_anomalies.T @ obs_anomalies
    # Σ̂_xy = c X̃ᵀ H̃
    return scatter / divisor


def _check_ensemble_size(ensemble_size: int, bessel: bool) -> None:
    """Check that an ensemble of |ℰ| members can supply the chosen divisor.

    Args:
        ensemble_size: Number of members |ℰ|.
        bessel: Whether the caller divides by |ℰ| − 1 rather than |ℰ|.

    Raises:
        ValueError: If |ℰ| < 1, or if ``bessel`` is set with |ℰ| < 2.
    """
    if ensemble_size < 1:
        raise ValueError(
            f"Ensemble must have at least one particle, got J={ensemble_size}."
        )
    if bessel and ensemble_size < 2:
        raise ValueError(
            "Bessel correction requires J >= 2 particles (divisor is J - 1); "
            f"got J={ensemble_size}. Pass bessel=False for a maximum-likelihood "
            "divisor."
        )
