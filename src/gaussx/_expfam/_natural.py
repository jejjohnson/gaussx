"""Natural <-> mean/covariance parameter conversions for Gaussians.

These functions convert between natural parameters ``(eta1, eta2)``
and mean/covariance ``(mu, Sigma)`` where ``Sigma`` is a lineax
operator. For moment-based expectation parameters ``(m1, m2)`` see
:mod:`gaussx._expfam._parameterizations`.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._primitives._inv import inv
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_solve


def natural_to_mean_cov(
    eta1: jnp.ndarray,
    eta2: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[jnp.ndarray, lx.AbstractLinearOperator]:
    """Convert natural parameters to mean/covariance.

    Given natural parameters ``(eta1, eta2)`` where
    ``eta1 = Lambda @ mu`` and ``eta2 = -0.5 * Lambda``:

    - ``mu = solve(-2 * eta2, eta1)``
    - ``Sigma = inv(-2 * eta2)``

    Args:
        eta1: Natural location parameter, shape ``(N,)``.
        eta2: Natural precision-like operator, shape ``(N, N)``.
        solver: Optional solver strategy. When ``None``, uses
            structural dispatch.

    Returns:
        Tuple ``(mu, Sigma)`` where mu is shape ``(N,)`` and
        Sigma is a linear operator.
    """
    neg2_eta2 = -2.0 * eta2
    mu = dispatch_solve(neg2_eta2, eta1, solver)
    Sigma = inv(neg2_eta2)
    return mu, Sigma


def mean_cov_to_natural(
    mu: jnp.ndarray,
    Sigma: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[jnp.ndarray, lx.AbstractLinearOperator]:
    """Convert mean/covariance to natural parameters.

    Given mean ``mu`` and covariance ``Sigma``:

    - ``eta1 = solve(Sigma, mu)``
    - ``eta2 = -0.5 * inv(Sigma)``

    Args:
        mu: Mean vector, shape ``(N,)``.
        Sigma: Covariance operator, shape ``(N, N)``.
        solver: Optional solver strategy. When ``None``, uses
            structural dispatch.

    Returns:
        Tuple ``(eta1, eta2)`` where eta1 is shape ``(N,)`` and
        eta2 is a linear operator.
    """
    eta1 = dispatch_solve(Sigma, mu, solver)
    eta2 = -0.5 * inv(Sigma)
    return eta1, eta2
