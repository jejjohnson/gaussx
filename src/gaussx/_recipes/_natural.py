"""Natural <-> expectation parameter conversions for Gaussians."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._primitives._inv import inv
from gaussx._primitives._solve import solve


def natural_to_expectation(
    eta1: jnp.ndarray,
    eta2: lx.AbstractLinearOperator,
) -> tuple[jnp.ndarray, lx.AbstractLinearOperator]:
    """Convert natural parameters to expectation parameters.

    Given natural parameters ``(eta1, eta2)`` where
    ``eta1 = Lambda @ mu`` and ``eta2 = -0.5 * Lambda``:

    - ``mu = solve(-2 * eta2, eta1)``
    - ``Sigma = inv(-2 * eta2)``

    Args:
        eta1: Natural location parameter, shape ``(N,)``.
        eta2: Natural precision-like operator, shape ``(N, N)``.

    Returns:
        Tuple ``(mu, Sigma)`` where mu is shape ``(N,)`` and
        Sigma is a linear operator.
    """
    neg2_eta2 = -2.0 * eta2
    mu = solve(neg2_eta2, eta1)
    Sigma = inv(neg2_eta2)
    return mu, Sigma


def expectation_to_natural(
    mu: jnp.ndarray,
    Sigma: lx.AbstractLinearOperator,
) -> tuple[jnp.ndarray, lx.AbstractLinearOperator]:
    """Convert expectation parameters to natural parameters.

    Given mean ``mu`` and covariance ``Sigma``:

    - ``eta1 = solve(Sigma, mu)``
    - ``eta2 = -0.5 * inv(Sigma)``

    Args:
        mu: Mean vector, shape ``(N,)``.
        Sigma: Covariance operator, shape ``(N, N)``.

    Returns:
        Tuple ``(eta1, eta2)`` where eta1 is shape ``(N,)`` and
        eta2 is a linear operator.
    """
    eta1 = solve(Sigma, mu)
    eta2 = -0.5 * inv(Sigma)
    return eta1, eta2
