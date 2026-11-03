"""Conversions between the mean, natural and expectation parameters of a Gaussian.

A Gaussian 𝒩(μ, Σ) on x ∈ ℝ^{D_x}, with precision Λ = Σ⁻¹, has natural
parameters (Λ μ, −½ Λ) and expectation parameters
(𝔼[x], 𝔼[x xᵀ]) = (μ, Σ + μ μᵀ). L is the lower Cholesky factor of the
covariance, L Lᵀ = Σ. This module is the single home of these conversions,
in two forms:

- **Operator form**, ``(mu, Sigma)`` with Σ a `lineax.AbstractLinearOperator`:
  the solves and inverses dispatch on its structure (diagonal, Kronecker, …).
  See `natural_to_mean_cov` and `mean_cov_to_natural`.
- **Dense Cholesky form**, ``(mu, S_sqrt)`` with ``S_sqrt`` = L: all six
  directions between (μ, L), the natural and the expectation parameters, on
  plain arrays with any number of leading batch axes. See
  `mean_chol_to_natural`, `natural_to_mean_chol`, `mean_chol_to_expectation`,
  `expectation_to_mean_chol`, `natural_to_expectation` and
  `expectation_to_natural`.

The block-tridiagonal conversions of a Gauss-Markov chain are in
``_ssm/_ssm_natural.py`` and the per-site EP ones in ``_ssm/_site_natural.py``.
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


# ---------------------------------------------------------------------------
# Operator-based conversions (exploit lineax structural dispatch)
# ---------------------------------------------------------------------------


def natural_to_mean_cov(
    eta1: Float[Array, " N"],
    eta2: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    """Mean μ and covariance Σ from the natural parameters, in operator form.

        Λ = −2 (−½ Λ),   μ = Λ⁻¹ (Λ μ),   Σ = Λ⁻¹

    The solve and the lazy inverse dispatch on the structure of −½ Λ
    (diagonal, Kronecker, …). For dense arrays see `natural_to_mean_chol`;
    for a block-tridiagonal chain, `gaussx.naturals_to_ssm`.

    Args:
        eta1: Precision-weighted mean Λ μ.
        eta2: Scaled precision −½ Λ, as an operator.
        solver: Solver strategy for Λ; ``None`` uses structural dispatch.

    Returns:
        Tuple ``(mu, Sigma)`` holding the mean μ and the covariance Σ as a
        lazy operator.

    Examples:
        A diagonal precision Λ = diag(2, 4):

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> scaled_precision = lx.DiagonalLinearOperator(jnp.array([-1.0, -2.0]))
        >>> mean, covariance = gaussx.natural_to_mean_cov(
        ...     jnp.array([2.0, 4.0]), scaled_precision
        ... )
        >>> bool(jnp.allclose(mean, 1.0))
        True
        >>> bool(jnp.allclose(covariance.as_matrix(), jnp.diag(jnp.array([0.5, 0.25]))))
        True

        It undoes `mean_cov_to_natural`:

        >>> matrix = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        >>> dense = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> naturals = gaussx.mean_cov_to_natural(jnp.array([1.0, -1.0]), dense)
        >>> mean, covariance = gaussx.natural_to_mean_cov(*naturals)
        >>> bool(jnp.allclose(covariance.as_matrix(), matrix, atol=1e-5))
        True
    """
    # Λ = −2 (−½ Λ)
    precision = -2.0 * eta2
    # μ = Λ⁻¹ (Λ μ)
    mean = dispatch_solve(precision, eta1, solver)
    # Σ = Λ⁻¹, lazy
    covariance = inv(precision)
    return mean, covariance


def mean_cov_to_natural(
    mu: Float[Array, " N"],
    Sigma: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    """Natural parameters (Λ μ, −½ Λ) from the mean and covariance, in operator form.

        Λ μ = Σ⁻¹ μ,   −½ Λ = −½ Σ⁻¹

    The solve and the lazy inverse dispatch on the structure of Σ
    (diagonal, Kronecker, …). For dense arrays see `mean_chol_to_natural`;
    for a block-tridiagonal chain, `gaussx.ssm_to_naturals`.

    Args:
        mu: Mean μ.
        Sigma: Covariance Σ.
        solver: Solver strategy for Σ; ``None`` uses structural dispatch.

    Returns:
        Tuple ``(eta1, eta2)`` holding Λ μ and −½ Λ as a lazy operator.

    Examples:
        A diagonal covariance Σ = diag(2, 4):

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> eta1, eta2 = gaussx.mean_cov_to_natural(jnp.ones(2), covariance)
        >>> bool(jnp.allclose(eta1, jnp.array([0.5, 0.25])))
        True
        >>> bool(jnp.allclose(eta2.as_matrix(), jnp.diag(jnp.array([-0.25, -0.125]))))
        True
    """
    # Λ μ = Σ⁻¹ μ
    eta1 = dispatch_solve(Sigma, mu, solver)
    # Λ = Σ⁻¹, lazy
    precision = inv(Sigma)
    # −½ Λ
    eta2 = -0.5 * precision
    return eta1, eta2


# ---------------------------------------------------------------------------
# Dense-array conversions (Cholesky factor form, batch-compatible)
# ---------------------------------------------------------------------------


def mean_chol_to_natural(
    mu: Float[Array, "*batch N"],
    S_sqrt: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    """Natural parameters (Λ μ, −½ Λ) from the mean and the Cholesky factor of Σ.

        Λ μ = L⁻ᵀ L⁻¹ μ,   −½ Λ = −½ L⁻ᵀ L⁻¹,   L Lᵀ = Σ

    Both come from triangular solves against L. No solver is exposed: the
    systems are triangular, not symmetric positive definite, so the
    iterative strategies (CG, BBMM, preconditioned CG, MINRES) do not apply.

    Algorithm:
        1. Solve L Lᵀ (Λ μ) = μ by two triangular solves.
        2. Solve L Lᵀ Λ = I the same way and scale by −½.
        3. With batch axes, flatten them, vmap steps 1 and 2, and restore them.

    Args:
        mu: Mean μ, with any number of leading batch axes.
        S_sqrt: Lower Cholesky factor L of the covariance, L Lᵀ = Σ.

    Returns:
        Tuple ``(eta1, eta2)`` holding Λ μ and −½ Λ.

    Examples:
        A diagonal factor L = diag(1, 2), so Σ = diag(1, 4):

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> factor = jnp.diag(jnp.array([1.0, 2.0]))
        >>> eta1, eta2 = gaussx.mean_chol_to_natural(jnp.ones(2), factor)
        >>> bool(jnp.allclose(eta1, jnp.array([1.0, 0.25])))
        True
        >>> bool(jnp.allclose(eta2, jnp.diag(jnp.array([-0.5, -0.125]))))
        True

        Leading batch axes are kept:

        >>> factors = jnp.broadcast_to(jnp.eye(3), (4, 2, 3, 3))
        >>> eta1, eta2 = gaussx.mean_chol_to_natural(jnp.ones((4, 2, 3)), factors)
        >>> eta1.shape, eta2.shape
        ((4, 2, 3), (4, 2, 3, 3))
    """

    def _core(mean: Float[Array, " N"], factor: Float[Array, "N N"]):
        # Λ μ = Σ⁻¹ μ = L⁻ᵀ L⁻¹ μ
        precision_mean = jax.scipy.linalg.cho_solve((factor, True), mean)
        event_size = factor.shape[0]
        identity = jnp.eye(event_size, dtype=factor.dtype)
        # Λ = Σ⁻¹ = L⁻ᵀ L⁻¹ I, one matrix solve
        precision = jax.scipy.linalg.cho_solve((factor, True), identity)
        # −½ Λ
        scaled_precision = -0.5 * precision
        return precision_mean, scaled_precision

    *batch, event_size = mu.shape
    if not batch:
        return _core(mu, S_sqrt)
    # reshape, not einx: einx rejects the zero-length axis of an empty batch
    mean_flat = mu.reshape(-1, event_size)
    factor_flat = S_sqrt.reshape(-1, event_size, event_size)
    eta1_flat, eta2_flat = jax.vmap(_core)(mean_flat, factor_flat)
    return eta1_flat.reshape(mu.shape), eta2_flat.reshape(S_sqrt.shape)


def natural_to_mean_chol(
    eta1: Float[Array, "*batch N"],
    eta2: Float[Array, "*batch N N"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    """Mean μ and Cholesky factor of Σ from the natural parameters (Λ μ, −½ Λ).

        Λ = −2 (−½ Λ),   μ = Λ⁻¹ (Λ μ),   Σ = Λ⁻¹,   L Lᵀ = Σ

    Algorithm:
        1. Form Λ = −2 (−½ Λ) as a PSD-tagged operator.
        2. Solve Λ μ for μ.
        3. Invert Λ to the dense Σ and factor Σ = L Lᵀ.
        4. With batch axes, flatten them, vmap steps 1 to 3, and restore them.

    Args:
        eta1: Precision-weighted mean Λ μ, with any number of leading batch
            axes.
        eta2: Scaled precision −½ Λ.
        solver: Solver strategy for Λ; ``None`` uses structural dispatch.

    Returns:
        Tuple ``(mu, S_sqrt)`` holding the mean μ and the lower Cholesky
        factor L of the covariance.

    Examples:
        A diagonal precision Λ = diag(1, 4):

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> eta2 = jnp.diag(jnp.array([-0.5, -2.0]))
        >>> mean, factor = gaussx.natural_to_mean_chol(jnp.array([1.0, 4.0]), eta2)
        >>> bool(jnp.allclose(mean, 1.0))
        True
        >>> bool(jnp.allclose(factor, jnp.diag(jnp.array([1.0, 0.5]))))
        True

        It undoes `mean_chol_to_natural`:

        >>> factor = jnp.array([[1.0, 0.0], [0.5, 2.0]])
        >>> naturals = gaussx.mean_chol_to_natural(jnp.array([1.0, -1.0]), factor)
        >>> _, factor_back = gaussx.natural_to_mean_chol(*naturals)
        >>> bool(jnp.allclose(factor_back, factor, atol=1e-5))
        True
    """

    def _core(
        precision_mean: Float[Array, " N"], scaled_precision: Float[Array, "N N"]
    ):
        # Λ = −2 (−½ Λ)
        precision_matrix = -2.0 * scaled_precision
        precision = lx.MatrixLinearOperator(
            precision_matrix, lx.positive_semidefinite_tag
        )
        # μ = Λ⁻¹ (Λ μ)
        mean = dispatch_solve(precision, precision_mean, solver)
        # Σ = Λ⁻¹
        covariance_matrix = inv(precision).as_matrix()
        covariance = lx.MatrixLinearOperator(
            covariance_matrix, lx.positive_semidefinite_tag
        )
        # L Lᵀ = Σ
        factor = cholesky(covariance).as_matrix()
        return mean, factor

    *batch, event_size = eta1.shape
    if not batch:
        return _core(eta1, eta2)
    # reshape, not einx: einx rejects the zero-length axis of an empty batch
    eta1_flat = eta1.reshape(-1, event_size)
    eta2_flat = eta2.reshape(-1, event_size, event_size)
    mean_flat, factor_flat = jax.vmap(_core)(eta1_flat, eta2_flat)
    return mean_flat.reshape(eta1.shape), factor_flat.reshape(eta2.shape)


def mean_chol_to_expectation(
    mu: Float[Array, "*batch N"],
    S_sqrt: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    """Expectation parameters (𝔼[x], 𝔼[x xᵀ]) from the mean and Cholesky factor.

        𝔼[x] = μ,   𝔼[x xᵀ] = μ μᵀ + L Lᵀ = μ μᵀ + Σ

    Args:
        mu: Mean μ, with any number of leading batch axes.
        S_sqrt: Lower Cholesky factor L of the covariance, L Lᵀ = Σ.

    Returns:
        Tuple ``(m1, m2)`` holding the first moment 𝔼[x] and the second
        moment 𝔼[x xᵀ].

    Examples:
        A unit mean and Σ = I give 𝔼[x xᵀ] = I + 1 1ᵀ:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> m1, m2 = gaussx.mean_chol_to_expectation(jnp.ones(2), jnp.eye(2))
        >>> bool(jnp.allclose(m2, jnp.array([[2.0, 1.0], [1.0, 2.0]])))
        True

        Leading batch axes are kept:

        >>> factors = jnp.broadcast_to(jnp.eye(3), (4, 3, 3))
        >>> m1, m2 = gaussx.mean_chol_to_expectation(jnp.zeros((4, 3)), factors)
        >>> m1.shape, m2.shape
        ((4, 3), (4, 3, 3))
    """
    # 𝔼[x] = μ
    m1 = mu
    # jnp, not einx: einx rejects the zero-length axis of an empty batch
    # μ μᵀ
    mean_outer = mu[..., None] * mu[..., None, :]
    # Σ = L Lᵀ
    covariance = S_sqrt @ S_sqrt.mT
    # 𝔼[x xᵀ] = μ μᵀ + Σ
    m2 = mean_outer + covariance
    return m1, m2


def expectation_to_mean_chol(
    m1: Float[Array, "*batch N"],
    m2: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    """Mean μ and Cholesky factor of Σ from the expectation parameters.

        μ = 𝔼[x],   Σ = 𝔼[x xᵀ] − μ μᵀ,   L Lᵀ = Σ

    No solver is exposed: the only linear algebra is the Cholesky
    factorisation.

    Algorithm:
        1. Subtract the outer product of the mean from the second moment.
        2. Factor Σ = L Lᵀ.
        3. With batch axes, flatten them, vmap steps 1 and 2, and restore them.

    Args:
        m1: First moment 𝔼[x] = μ, with any number of leading batch axes.
        m2: Second moment 𝔼[x xᵀ] = Σ + μ μᵀ.

    Returns:
        Tuple ``(mu, S_sqrt)`` holding the mean μ and the lower Cholesky
        factor L of the covariance.

    Examples:
        It undoes `mean_chol_to_expectation`:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> factor = jnp.array([[1.0, 0.0], [0.5, 2.0]])
        >>> moments = gaussx.mean_chol_to_expectation(jnp.array([1.0, -1.0]), factor)
        >>> mean, factor_back = gaussx.expectation_to_mean_chol(*moments)
        >>> bool(jnp.allclose(factor_back, factor, atol=1e-5))
        True

        Leading batch axes are kept:

        >>> second = jnp.broadcast_to(jnp.eye(3), (4, 3, 3))
        >>> mean, factor = gaussx.expectation_to_mean_chol(jnp.zeros((4, 3)), second)
        >>> mean.shape, factor.shape
        ((4, 3), (4, 3, 3))
    """

    def _core(first_moment: Float[Array, " N"], second_moment: Float[Array, "N N"]):
        # jnp, not einx: einx rejects N = 0
        # μ μᵀ
        mean_outer = jnp.outer(first_moment, first_moment)
        # Σ = 𝔼[x xᵀ] − μ μᵀ
        covariance_matrix = second_moment - mean_outer
        covariance = lx.MatrixLinearOperator(
            covariance_matrix, lx.positive_semidefinite_tag
        )
        # L Lᵀ = Σ
        factor = cholesky(covariance).as_matrix()
        return first_moment, factor

    *batch, event_size = m1.shape
    if not batch:
        return _core(m1, m2)
    # reshape, not einx: einx rejects the zero-length axis of an empty batch
    m1_flat = m1.reshape(-1, event_size)
    m2_flat = m2.reshape(-1, event_size, event_size)
    mean_flat, factor_flat = jax.vmap(_core)(m1_flat, m2_flat)
    return mean_flat.reshape(m1.shape), factor_flat.reshape(m2.shape)


def natural_to_expectation(
    eta1: Float[Array, "*batch N"],
    eta2: Float[Array, "*batch N N"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    """Expectation parameters (𝔼[x], 𝔼[x xᵀ]) from the natural parameters.

        Λ = −2 (−½ Λ),   μ = Λ⁻¹ (Λ μ),   Σ = Λ⁻¹
        𝔼[x] = μ,   𝔼[x xᵀ] = μ μᵀ + Σ

    Algorithm:
        1. Form Λ = −2 (−½ Λ) as a PSD-tagged operator.
        2. Solve Λ μ for μ and invert Λ to the dense Σ.
        3. Add the outer product of the mean to Σ.
        4. With batch axes, flatten them, vmap steps 1 to 3, and restore them.

    Args:
        eta1: Precision-weighted mean Λ μ, with any number of leading batch
            axes.
        eta2: Scaled precision −½ Λ.
        solver: Solver strategy for Λ; ``None`` uses structural dispatch.

    Returns:
        Tuple ``(m1, m2)`` holding the first moment 𝔼[x] and the second
        moment 𝔼[x xᵀ].

    Examples:
        A unit mean and Λ = I give 𝔼[x xᵀ] = I + 1 1ᵀ:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> m1, m2 = gaussx.natural_to_expectation(jnp.ones(2), -0.5 * jnp.eye(2))
        >>> bool(jnp.allclose(m1, 1.0))
        True
        >>> bool(jnp.allclose(m2, jnp.array([[2.0, 1.0], [1.0, 2.0]])))
        True

        It undoes `expectation_to_natural`:

        >>> second = jnp.array([[2.0, 0.5], [0.5, 1.5]])
        >>> naturals = gaussx.expectation_to_natural(jnp.array([0.5, 0.0]), second)
        >>> _, second_back = gaussx.natural_to_expectation(*naturals)
        >>> bool(jnp.allclose(second_back, second, atol=1e-5))
        True
    """

    def _core(
        precision_mean: Float[Array, " N"], scaled_precision: Float[Array, "N N"]
    ):
        # Λ = −2 (−½ Λ)
        precision_matrix = -2.0 * scaled_precision
        precision = lx.MatrixLinearOperator(
            precision_matrix, lx.positive_semidefinite_tag
        )
        # μ = Λ⁻¹ (Λ μ)
        mean = dispatch_solve(precision, precision_mean, solver)
        # Σ = Λ⁻¹
        covariance = inv(precision).as_matrix()
        # jnp, not einx: einx rejects N = 0
        # μ μᵀ
        mean_outer = jnp.outer(mean, mean)
        # 𝔼[x xᵀ] = μ μᵀ + Σ
        second_moment = mean_outer + covariance
        return mean, second_moment

    *batch, event_size = eta1.shape
    if not batch:
        return _core(eta1, eta2)
    # reshape, not einx: einx rejects the zero-length axis of an empty batch
    eta1_flat = eta1.reshape(-1, event_size)
    eta2_flat = eta2.reshape(-1, event_size, event_size)
    m1_flat, m2_flat = jax.vmap(_core)(eta1_flat, eta2_flat)
    return m1_flat.reshape(eta1.shape), m2_flat.reshape(eta2.shape)


def expectation_to_natural(
    m1: Float[Array, "*batch N"],
    m2: Float[Array, "*batch N N"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    """Natural parameters (Λ μ, −½ Λ) from the expectation parameters.

        Σ = 𝔼[x xᵀ] − 𝔼[x] 𝔼[x]ᵀ,   Λ μ = Σ⁻¹ 𝔼[x],   −½ Λ = −½ Σ⁻¹

    Algorithm:
        1. Subtract the outer product of the mean from the second moment
           and tag Σ as PSD.
        2. Solve Σ for Λ μ, then against the identity for Λ = Σ⁻¹.
        3. Scale Λ by −½.
        4. With batch axes, flatten them, vmap steps 1 to 3, and restore them.

    Args:
        m1: First moment 𝔼[x] = μ, with any number of leading batch axes.
        m2: Second moment 𝔼[x xᵀ] = Σ + μ μᵀ.
        solver: Solver strategy for Σ; ``None`` uses structural dispatch.

    Returns:
        Tuple ``(eta1, eta2)`` holding Λ μ and −½ Λ.

    Examples:
        A unit mean with Σ = 2 I:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> second = 2.0 * jnp.eye(2) + jnp.ones((2, 2))
        >>> eta1, eta2 = gaussx.expectation_to_natural(jnp.ones(2), second)
        >>> bool(jnp.allclose(eta1, 0.5)), bool(jnp.allclose(eta2, -0.25 * jnp.eye(2)))
        (True, True)

        Leading batch axes are kept:

        >>> seconds = jnp.broadcast_to(jnp.eye(3), (4, 3, 3))
        >>> eta1, eta2 = gaussx.expectation_to_natural(jnp.zeros((4, 3)), seconds)
        >>> eta1.shape, eta2.shape
        ((4, 3), (4, 3, 3))
    """

    def _core(first_moment: Float[Array, " N"], second_moment: Float[Array, "N N"]):
        # jnp, not einx: einx rejects N = 0
        # μ μᵀ
        mean_outer = jnp.outer(first_moment, first_moment)
        # Σ = 𝔼[x xᵀ] − μ μᵀ
        covariance_matrix = second_moment - mean_outer
        covariance = lx.MatrixLinearOperator(
            covariance_matrix, lx.positive_semidefinite_tag
        )
        # Λ μ = Σ⁻¹ μ
        precision_mean = dispatch_solve(covariance, first_moment, solver)
        event_size = first_moment.shape[0]
        identity = jnp.eye(event_size, dtype=first_moment.dtype)
        # Λ = Σ⁻¹ I
        precision = solve_columns(covariance, identity, solver=solver)
        # −½ Λ
        scaled_precision = -0.5 * precision
        return precision_mean, scaled_precision

    *batch, event_size = m1.shape
    if not batch:
        return _core(m1, m2)
    # reshape, not einx: einx rejects the zero-length axis of an empty batch
    m1_flat = m1.reshape(-1, event_size)
    m2_flat = m2.reshape(-1, event_size, event_size)
    eta1_flat, eta2_flat = jax.vmap(_core)(m1_flat, m2_flat)
    return eta1_flat.reshape(m1.shape), eta2_flat.reshape(m2.shape)
