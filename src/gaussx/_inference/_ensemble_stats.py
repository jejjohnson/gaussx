"""Ensemble covariance and cross-covariance statistics."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
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


def ledoit_wolf_shrinkage(
    particles: Float[Array, "J N"],
    *,
    bessel: bool = True,
) -> tuple[LowRankUpdate, Float[Array, ""]]:
    """Ledoit-Wolf shrinkage of an ensemble covariance, without forming it.

        Σ̂_λ = (1 − λ) Σ̂_x + λ μ I,   μ = tr(Σ̂_x) / D_x
        d² = ‖Σ̂_x − μ I‖_F²
        b² = (1 / |ℰ|²) ∑ₑ ‖x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ − Σ̂_x‖_F²
        λ = min(1, b² / d²)

    The estimator of Ledoit & Wolf (2004), shrinking toward a scaled identity.
    With the anomalies X̃ as rows and the (|ℰ|, |ℰ|) Gram matrix G = X̃ X̃ᵀ, and
    c the divisor of `ensemble_covariance`, everything comes from G at
    O(|ℰ|² D_x) and no (D_x, D_x) array is formed:

        tr(Σ̂_x) = c tr(G),   ‖Σ̂_x‖_F² = c² ‖G‖_F²
        d² = ‖Σ̂_x‖_F² − tr(Σ̂_x)² / D_x
        b² = (1 / |ℰ|²) (∑ₑ Gₑₑ² + (|ℰ| c² − 2 c) ‖G‖_F²)

    Ledoit and Wolf define Σ̂_x with c = 1 / |ℰ|. With ``bessel=True`` the same
    formulas are read with c = 1 / (|ℰ| − 1), a small difference.

    Algorithm:
        1. Check that the ensemble is large enough for the divisor.
        2. Form G from the anomalies.
        3. Read μ, d² and b² off G, and clip b² / d² to [0, 1].
        4. Return (1 − λ) U Uᵀ + λ μ I as a `gaussx.LowRankUpdate`.

    Args:
        particles: Ensemble in state space, one row per member.
        bessel: Use c = 1 / (|ℰ| − 1) rather than c = 1 / |ℰ|. Defaults to
            ``True``, the ensemble Kalman convention, unlike
            `ensemble_covariance`.

    Returns:
        Tuple ``(covariance, shrinkage)`` holding Σ̂_λ as a PSD-tagged
        `gaussx.LowRankUpdate` over a diagonal base, and λ. The update's weights
        carry 1 − λ, so its gradient stays finite where λ clips at 1.

    Raises:
        ValueError: If |ℰ| < 2 with ``bessel=True``, or |ℰ| < 1.

    Examples:
        Few members in a large state: the operator is structured, and it solves
        through Woodbury:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> particles = jr.normal(jr.key(0), (10, 200))
        >>> covariance, shrinkage = gaussx.ledoit_wolf_shrinkage(particles)
        >>> isinstance(covariance, gaussx.LowRankUpdate)
        True
        >>> bool(0.0 <= shrinkage <= 1.0)
        True
        >>> gaussx.solve(covariance, jnp.ones(200)).shape
        (200,)

        Unit-variance noise in a large state needs heavy shrinkage:

        >>> bool(shrinkage > 0.5)
        True

        A strongly correlated ensemble needs little:

        >>> shared = jr.normal(jr.key(1), (500, 1))
        >>> correlated = shared * jnp.ones((500, 5))
        >>> _, small = gaussx.ledoit_wolf_shrinkage(correlated)
        >>> bool(small < 0.1)
        True

    References:
        Ledoit & Wolf (2004), *J. Multivariate Anal.* 88, 365-411.
    """
    ensemble_size, state_size = particles.shape
    _check_ensemble_size(ensemble_size, bessel)
    covariance = ensemble_covariance(particles, bessel=bessel)
    # U = √c X̃ᵀ
    low_rank_factor = covariance.U
    # c
    divisor = ensemble_size - 1 if bessel else ensemble_size
    scale = jnp.asarray(1.0 / divisor, dtype=particles.dtype)
    # c G = Uᵀ U
    scaled_gram = einsum(
        low_rank_factor, low_rank_factor, "state member, state other -> member other"
    )
    # c Gₑₑ
    scaled_diagonal = jnp.diagonal(scaled_gram)
    # tr(Σ̂_x) = tr(Uᵀ U)
    trace = jnp.sum(scaled_diagonal)
    # μ = tr(Σ̂_x) / D_x
    target_scale = trace / state_size
    # ‖Σ̂_x‖_F² = ‖Uᵀ U‖_F²
    covariance_squared = jnp.sum(scaled_gram * scaled_gram)
    # d² = ‖Σ̂_x‖_F² − tr(Σ̂_x)² / D_x
    distance_squared = covariance_squared - trace * trace / state_size
    # ∑ₑ Gₑₑ²
    member_squared = jnp.sum(scaled_diagonal * scaled_diagonal) / (scale * scale)
    # (|ℰ| c² − 2 c) ‖G‖_F², with ‖G‖_F² = ‖Σ̂_x‖_F² / c²
    cross_squared = (ensemble_size - 2.0 / scale) * covariance_squared
    # b² = (1 / |ℰ|²) (∑ₑ Gₑₑ² + (|ℰ| c² − 2 c) ‖G‖_F²)
    deviation_squared = (member_squared + cross_squared) / ensemble_size**2
    # λ = 1 where b² ≥ d², decided first so that the division never meets d² = 0,
    # whose ∞ slope would give NaN gradients through the clipped branch.
    clipped = deviation_squared >= distance_squared
    safe_distance_squared = jnp.where(clipped, 1.0, distance_squared)
    # λ = b² / d², and b² is non-negative, which rounding could undershoot
    ratio = jnp.maximum(deviation_squared, 0.0) / safe_distance_squared
    shrinkage = jnp.where(clipped, 1.0, ratio)
    # λ μ I
    base = lx.DiagonalLinearOperator(
        shrinkage * target_scale * jnp.ones(state_size, dtype=particles.dtype)
    )
    # 1 − λ on every mode, so that no √(1 − λ) has an infinite slope at λ = 1
    weights = (1.0 - shrinkage) * jnp.ones(ensemble_size, dtype=particles.dtype)
    # Σ̂_λ = λ μ I + U diag(1 − λ) Uᵀ
    shrunk = LowRankUpdate(
        base, low_rank_factor, weights, tags=lx.positive_semidefinite_tag
    )
    return shrunk, shrinkage


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


def _check_obs_noise_shape(obs_size: int, obs_noise: lx.AbstractLinearOperator) -> None:
    """Check that Σ_ε is the (D_y, D_y) operator the observation ensemble needs.

    Without this check a wrong-sized operator broadcasts against the (D_y, D_y)
    sample covariance instead of raising: a (1, 1) Σ_ε against D_y = 3 adds the
    scalar to every entry and yields a plausible, wrong gain.

    Args:
        obs_size: Observation length D_y.
        obs_noise: Observation-error covariance Σ_ε.

    Raises:
        ValueError: If ``obs_noise`` is not (D_y, D_y).
    """
    if (obs_noise.in_size(), obs_noise.out_size()) != (obs_size, obs_size):
        raise ValueError(
            f"obs_noise must be ({obs_size}, {obs_size}) to match obs_particles, "
            f"got ({obs_noise.out_size()}, {obs_noise.in_size()})."
        )
