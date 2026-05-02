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

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._linalg import solve_columns
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._inv import inv
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_solve


def meanvar_to_natural(
    mu: Float[Array, "*batch N"],
    S_sqrt: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert mean/variance (Cholesky) to natural parameters.

    Given ``mu`` and lower-triangular ``S_sqrt`` such that
    ``Sigma = S_sqrt @ S_sqrt^T``:

    - ``eta1 = Sigma^{-1} mu``
    - ``eta2 = -0.5 * Sigma^{-1}``

    Uses the Cholesky factor directly via triangular solves; no solver
    parameter is exposed because the underlying systems are triangular
    rather than symmetric/PSD, and iterative strategies (CG, BBMM,
    PreconditionedCG, MINRES) are not valid here.

    Args:
        mu: Mean vector, shape ``(*batch, N)``.
        S_sqrt: Lower-triangular Cholesky factor, shape ``(*batch, N, N)``.

    Returns:
        Tuple ``(eta1, eta2)`` of natural parameters.
    """

    def _core(mu_s: Float[Array, " N"], s_sqrt_s: Float[Array, "N N"]):
        # eta1 = Sigma^{-1} mu = S_sqrt^{-T} S_sqrt^{-1} mu via cho_solve.
        eta1_s = jax.scipy.linalg.cho_solve((s_sqrt_s, True), mu_s)
        # eta2 = -0.5 * Sigma^{-1}, computed by a single matrix cho_solve.
        N = s_sqrt_s.shape[0]
        identity = jnp.eye(N, dtype=s_sqrt_s.dtype)
        Sigma_inv = jax.scipy.linalg.cho_solve((s_sqrt_s, True), identity)
        return eta1_s, -0.5 * Sigma_inv

    *batch, N = mu.shape
    if not batch:
        return _core(mu, S_sqrt)
    mu_flat = mu.reshape(-1, N)
    s_flat = S_sqrt.reshape(-1, N, N)
    eta1_flat, eta2_flat = jax.vmap(_core)(mu_flat, s_flat)
    return eta1_flat.reshape(mu.shape), eta2_flat.reshape(S_sqrt.shape)


def natural_to_meanvar(
    eta1: Float[Array, "*batch N"],
    eta2: Float[Array, "*batch N N"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert natural parameters to mean/variance (Cholesky).

    Given ``eta1 = Lambda @ mu`` and ``eta2 = -0.5 * Lambda``:

    - ``Sigma = (-2 * eta2)^{-1}``
    - ``mu = Sigma @ eta1``
    - ``S_sqrt = cholesky(Sigma)``

    Args:
        eta1: Natural location parameter, shape ``(*batch, N)``.
        eta2: Natural quadratic parameter, shape ``(*batch, N, N)``.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(mu, S_sqrt)`` where ``S_sqrt`` is the lower-triangular
        Cholesky factor of the covariance.
    """

    def _core(e1: Float[Array, " N"], e2: Float[Array, "N N"]):
        Lambda_op = lx.MatrixLinearOperator(-2.0 * e2, lx.positive_semidefinite_tag)
        mu_s = dispatch_solve(Lambda_op, e1, solver)
        Sigma = inv(Lambda_op).as_matrix()
        Sigma_op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        return mu_s, cholesky(Sigma_op).as_matrix()

    *batch, N = eta1.shape
    if not batch:
        return _core(eta1, eta2)
    eta1_flat = eta1.reshape(-1, N)
    eta2_flat = eta2.reshape(-1, N, N)
    mu_flat, s_flat = jax.vmap(_core)(eta1_flat, eta2_flat)
    return mu_flat.reshape(eta1.shape), s_flat.reshape(eta2.shape)


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

    No solver parameter is exposed because the only linear-algebra
    operation is Cholesky factorization, which is structurally fixed.

    Args:
        m1: First moment (mean), shape ``(*batch, N)``.
        m2: Second moment, shape ``(*batch, N, N)``.

    Returns:
        Tuple ``(mu, S_sqrt)`` where ``S_sqrt`` is the lower-triangular
        Cholesky factor of the covariance.
    """

    def _core(m1_s: Float[Array, " N"], m2_s: Float[Array, "N N"]):
        Sigma = m2_s - m1_s[:, None] * m1_s[None, :]
        Sigma_op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        return m1_s, cholesky(Sigma_op).as_matrix()

    *batch, N = m1.shape
    if not batch:
        return _core(m1, m2)
    m1_flat = m1.reshape(-1, N)
    m2_flat = m2.reshape(-1, N, N)
    mu_flat, s_flat = jax.vmap(_core)(m1_flat, m2_flat)
    return mu_flat.reshape(m1.shape), s_flat.reshape(m2.shape)


def natural_to_expectation(
    eta1: Float[Array, "*batch N"],
    eta2: Float[Array, "*batch N N"],
    *,
    solver: AbstractSolverStrategy | None = None,
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
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(m1, m2)`` of expectation parameters.
    """

    def _core(e1: Float[Array, " N"], e2: Float[Array, "N N"]):
        Lambda_op = lx.MatrixLinearOperator(-2.0 * e2, lx.positive_semidefinite_tag)
        mu_s = dispatch_solve(Lambda_op, e1, solver)
        Sigma = inv(Lambda_op).as_matrix()
        m2_s = mu_s[:, None] * mu_s[None, :] + Sigma
        return mu_s, m2_s

    *batch, N = eta1.shape
    if not batch:
        return _core(eta1, eta2)
    eta1_flat = eta1.reshape(-1, N)
    eta2_flat = eta2.reshape(-1, N, N)
    m1_flat, m2_flat = jax.vmap(_core)(eta1_flat, eta2_flat)
    return m1_flat.reshape(eta1.shape), m2_flat.reshape(eta2.shape)


def expectation_to_natural(
    m1: Float[Array, "*batch N"],
    m2: Float[Array, "*batch N N"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    r"""Convert expectation parameters to natural parameters.

    Given ``m1 = mu`` and ``m2 = mu @ mu^T + Sigma``:

    - ``Sigma = m2 - m1 @ m1^T``
    - ``eta1 = Sigma^{-1} @ m1``
    - ``eta2 = -0.5 * Sigma^{-1}``

    Args:
        m1: First moment (mean), shape ``(*batch, N)``.
        m2: Second moment, shape ``(*batch, N, N)``.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(eta1, eta2)`` of natural parameters.
    """

    def _core(m1_s: Float[Array, " N"], m2_s: Float[Array, "N N"]):
        Sigma = m2_s - m1_s[:, None] * m1_s[None, :]
        Sigma_op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        eta1_s = dispatch_solve(Sigma_op, m1_s, solver)
        N = m1_s.shape[0]
        identity = jnp.eye(N, dtype=m1_s.dtype)
        Sigma_inv = solve_columns(Sigma_op, identity, solver=solver)
        return eta1_s, -0.5 * Sigma_inv

    *batch, N = m1.shape
    if not batch:
        return _core(m1, m2)
    m1_flat = m1.reshape(-1, N)
    m2_flat = m2.reshape(-1, N, N)
    eta1_flat, eta2_flat = jax.vmap(_core)(m1_flat, m2_flat)
    return eta1_flat.reshape(m1.shape), eta2_flat.reshape(m2.shape)
