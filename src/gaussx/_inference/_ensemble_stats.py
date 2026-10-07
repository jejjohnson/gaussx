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
    r"""Empirical covariance from an ensemble as a low-rank operator.

    Returns ``C = c X'^T X'`` with ``c = 1 / J`` when ``bessel=False``
    (default, maximum likelihood) and ``c = 1 / (J - 1)`` when
    ``bessel=True`` (unbiased / ensemble Kalman filter convention).
    The result is a ``LowRankUpdate`` of rank ``<= J-1`` rather than
    materializing the full ``(N, N)`` matrix.  Efficient when
    ``J << N``.

    Args:
        particles: Ensemble of shape ``(J, N)``.
        bessel: If True, apply the ``1 / (J - 1)`` Bessel correction
            used throughout the ensemble Kalman filter literature. This
            lower-level helper defaults to False for backwards compatibility;
            `ensemble_kalman_gain` defaults to True for the EnKF
            convention.

    Returns:
        A ``LowRankUpdate`` operator representing the empirical
        covariance, with a zero base and ``J``-column low-rank factor.

    Note:
        The ``bessel`` default differs across this module: these two
        covariance helpers default to ``bessel=False`` (``1 / J``), while
        `ensemble_kalman_gain`, `localized_kalman_gain`, `enkf_analysis`,
        `eki_step` and `discrepancy_step_size` default to ``True``
        (``1 / (J - 1)``) and `etkf_transform` is ``1 / (J - 1)``
        throughout. A gain assembled by hand from these helpers with their
        defaults therefore differs from `ensemble_kalman_gain`'s; pass
        ``bessel=True`` to match it.
    """
    J, N = particles.shape
    _check_ensemble_size(J, bessel)
    mean = jnp.mean(particles, axis=0)
    deviations = particles - mean[None, :]  # (J, N)

    divisor = J - 1 if bessel else J
    U = deviations.T / jnp.sqrt(divisor)  # (N, J)

    base = lx.DiagonalLinearOperator(jnp.zeros(N, dtype=particles.dtype))
    return LowRankUpdate(base, U)


def ensemble_cross_covariance(
    particles_theta: Float[Array, "J N"],
    particles_G: Float[Array, "J M"],
    *,
    bessel: bool = False,
) -> Float[Array, "N M"]:
    r"""Cross-covariance between two ensemble sets.

    Computes ``C^{theta,G} = c sum_j (theta_j - bar)(G_j - bar)^T``
    with ``c = 1 / J`` by default or ``c = 1 / (J - 1)`` when
    ``bessel=True``.

    Args:
        particles_theta: First ensemble, shape ``(J, N)``.
        particles_G: Second ensemble, shape ``(J, M)``.
        bessel: If True, apply the ``1 / (J - 1)`` Bessel correction
            used by ensemble Kalman filter recipes. This lower-level helper
            defaults to False for backwards compatibility; `ensemble_kalman_gain`
            defaults to True for the EnKF convention.

    Returns:
        Cross-covariance array of shape ``(N, M)``.

    Note:
        The ``bessel`` default differs across this module: these two
        covariance helpers default to ``bessel=False`` (``1 / J``), while
        `ensemble_kalman_gain`, `localized_kalman_gain`, `enkf_analysis`,
        `eki_step` and `discrepancy_step_size` default to ``True``
        (``1 / (J - 1)``) and `etkf_transform` is ``1 / (J - 1)``
        throughout. A gain assembled by hand from these helpers with their
        defaults therefore differs from `ensemble_kalman_gain`'s; pass
        ``bessel=True`` to match it.
    """
    J = particles_theta.shape[0]
    _check_ensemble_size(J, bessel)
    dev_theta = particles_theta - jnp.mean(particles_theta, axis=0, keepdims=True)
    dev_G = particles_G - jnp.mean(particles_G, axis=0, keepdims=True)
    divisor = J - 1 if bessel else J
    return (dev_theta.T @ dev_G) / divisor


def _check_ensemble_size(J: int, bessel: bool) -> None:
    if J < 1:
        raise ValueError(f"Ensemble must have at least one particle, got J={J}.")
    if bessel and J < 2:
        raise ValueError(
            "Bessel correction requires J >= 2 particles (divisor is J - 1); "
            f"got J={J}. Pass bessel=False for a maximum-likelihood divisor."
        )
