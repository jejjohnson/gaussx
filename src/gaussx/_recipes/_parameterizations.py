"""Full 3-parameterization conversions for Gaussians.

Provides all six conversion directions between:

- **Mean/variance (Cholesky)**: ``(mu, S_sqrt)`` where ``S = S_sqrt @ S_sqrt^T``
- **Natural**: ``(eta1, eta2)`` where ``eta1 = Lambda @ mu``,
  ``eta2 = -0.5 * Lambda``
- **Expectation**: ``(m1, m2)`` where ``m1 = mu``,
  ``m2 = mu @ mu^T + Sigma``

Conversions follow the GPflow natgrad convention. All functions
operate on raw JAX arrays (not lineax operators).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def meanvar_to_natural(
    mu: Float[Array, "*batch N"],
    S_sqrt: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert mean/variance (Cholesky) to natural parameters.

    Given ``mu`` and lower-triangular ``S_sqrt`` such that
    ``Sigma = S_sqrt @ S_sqrt^T``:

    - ``eta1 = Sigma^{-1} mu``
    - ``eta2 = -0.5 * Sigma^{-1}``

    Uses the Cholesky factor directly to avoid forming ``Sigma``.

    Args:
        mu: Mean vector, shape ``(*batch, N)``.
        S_sqrt: Lower-triangular Cholesky factor, shape ``(*batch, N, N)``.

    Returns:
        Tuple ``(eta1, eta2)`` of natural parameters.
    """
    # Sigma^{-1} = S_sqrt^{-T} S_sqrt^{-1}
    # eta1 = Sigma^{-1} mu via cho_solve
    alpha = jnp.linalg.solve(S_sqrt, mu[..., None])[..., 0]  # S_sqrt^{-1} mu
    eta1 = jnp.linalg.solve(S_sqrt.mT, alpha[..., None])[..., 0]  # S_sqrt^{-T} alpha

    # eta2 = -0.5 * Sigma^{-1}
    S_sqrt_inv = jnp.linalg.inv(S_sqrt)
    Sigma_inv = S_sqrt_inv.mT @ S_sqrt_inv
    eta2 = -0.5 * Sigma_inv
    return eta1, eta2


def natural_to_meanvar(
    eta1: Float[Array, "*batch N"],
    eta2: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert natural parameters to mean/variance (Cholesky).

    Given ``eta1 = Lambda @ mu`` and ``eta2 = -0.5 * Lambda``:

    - ``Sigma = (-2 * eta2)^{-1}``
    - ``mu = Sigma @ eta1``
    - ``S_sqrt = cholesky(Sigma)``

    Args:
        eta1: Natural location parameter, shape ``(*batch, N)``.
        eta2: Natural quadratic parameter, shape ``(*batch, N, N)``.

    Returns:
        Tuple ``(mu, S_sqrt)`` where ``S_sqrt`` is the lower-triangular
        Cholesky factor of the covariance.
    """
    Sigma = jnp.linalg.inv(-2.0 * eta2)
    mu = (Sigma @ eta1[..., None])[..., 0]
    S_sqrt = jnp.linalg.cholesky(Sigma)
    return mu, S_sqrt


def meanvar_to_expectation(
    mu: Float[Array, "*batch N"],
    S_sqrt: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert mean/variance (Cholesky) to expectation parameters.

    Given ``mu`` and ``S_sqrt`` (lower-triangular Cholesky of ``Sigma``):

    - ``m1 = mu``
    - ``m2 = mu @ mu^T + Sigma = mu @ mu^T + S_sqrt @ S_sqrt^T``

    Args:
        mu: Mean vector, shape ``(*batch, N)``.
        S_sqrt: Lower-triangular Cholesky factor, shape ``(*batch, N, N)``.

    Returns:
        Tuple ``(m1, m2)`` of expectation parameters.
    """
    m1 = mu
    m2 = mu[..., None] * mu[..., None, :] + S_sqrt @ S_sqrt.mT
    return m1, m2


def expectation_to_meanvar(
    m1: Float[Array, "*batch N"],
    m2: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert expectation parameters to mean/variance (Cholesky).

    Given ``m1 = mu`` and ``m2 = mu @ mu^T + Sigma``:

    - ``mu = m1``
    - ``Sigma = m2 - m1 @ m1^T``
    - ``S_sqrt = cholesky(Sigma)``

    Args:
        m1: First moment (mean), shape ``(*batch, N)``.
        m2: Second moment, shape ``(*batch, N, N)``.

    Returns:
        Tuple ``(mu, S_sqrt)`` where ``S_sqrt`` is the lower-triangular
        Cholesky factor of the covariance.
    """
    mu = m1
    Sigma = m2 - m1[..., None] * m1[..., None, :]
    S_sqrt = jnp.linalg.cholesky(Sigma)
    return mu, S_sqrt


def natural_to_expectation(
    eta1: Float[Array, "*batch N"],
    eta2: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert natural parameters to expectation parameters.

    Given ``eta1 = Lambda @ mu`` and ``eta2 = -0.5 * Lambda``:

    - ``Sigma = (-2 * eta2)^{-1}``
    - ``mu = Sigma @ eta1``
    - ``m1 = mu``
    - ``m2 = mu @ mu^T + Sigma``

    Args:
        eta1: Natural location parameter, shape ``(*batch, N)``.
        eta2: Natural quadratic parameter, shape ``(*batch, N, N)``.

    Returns:
        Tuple ``(m1, m2)`` of expectation parameters.
    """
    Sigma = jnp.linalg.inv(-2.0 * eta2)
    mu = (Sigma @ eta1[..., None])[..., 0]
    m1 = mu
    m2 = mu[..., None] * mu[..., None, :] + Sigma
    return m1, m2


def expectation_to_natural(
    m1: Float[Array, "*batch N"],
    m2: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert expectation parameters to natural parameters.

    Given ``m1 = mu`` and ``m2 = mu @ mu^T + Sigma``:

    - ``Sigma = m2 - m1 @ m1^T``
    - ``eta1 = Sigma^{-1} @ m1``
    - ``eta2 = -0.5 * Sigma^{-1}``

    Args:
        m1: First moment (mean), shape ``(*batch, N)``.
        m2: Second moment, shape ``(*batch, N, N)``.

    Returns:
        Tuple ``(eta1, eta2)`` of natural parameters.
    """
    Sigma = m2 - m1[..., None] * m1[..., None, :]
    Sigma_inv = jnp.linalg.inv(Sigma)
    eta1 = (Sigma_inv @ m1[..., None])[..., 0]
    eta2 = -0.5 * Sigma_inv
    return eta1, eta2
