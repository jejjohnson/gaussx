"""Leave-one-out cross-validation of a GP via the bordered-system identity.

Σ_y is the N × N covariance of the training targets y (K_XX + σ_ε² I for a
GP) and α = Σ_y⁻¹ y the training solve, as in `gaussx.build_prediction_cache`.
Leaving out target n gives the predictive 𝒩(μ_LOO,n, σ²_LOO,n) from the rest.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._diag_inv import diag_inv
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve


class LOOResult(eqx.Module):
    """Result of `leave_one_out_cv`: the LOO log-likelihood and predictives.

    Attributes:
        loo_log_likelihood: LOO-CV log-likelihood
            ∑_n log 𝒩(y_n; μ_LOO,n, σ²_LOO,n).
        loo_means: LOO predictive means μ_LOO,n, one per training target.
        loo_variances: LOO predictive variances σ²_LOO,n, one per training
            target.
    """

    loo_log_likelihood: Float[Array, ""]
    loo_means: Float[Array, " N"]
    loo_variances: Float[Array, " N"]


def leave_one_out_cv(
    operator: lx.AbstractLinearOperator,
    y: Float[Array, " N"],
    *,
    solver: AbstractSolveStrategy | None = None,
    diag_inv_method: str = "solve",
    diag_inv_num_probes: int = 30,
    diag_inv_key: jax.Array | None = None,
) -> LOOResult:
    """LOO-CV via the bordered-system identity, without refitting N times.

        μ_LOO,n = y_n − α_n / [Σ_y⁻¹]_nn,   α = Σ_y⁻¹ y
        σ²_LOO,n = 1 / [Σ_y⁻¹]_nn
        LOO-CV = −½ ∑_n (log σ²_LOO,n + (y_n − μ_LOO,n)² / σ²_LOO,n + log 2π)

    The squared standardised residual simplifies to
    (y_n − μ_LOO,n)² / σ²_LOO,n = α_n² / [Σ_y⁻¹]_nn.

    Algorithm:
        1. Solve α = Σ_y⁻¹ y with ``solver`` or structural dispatch.
        2. Take diag(Σ_y⁻¹) with `diag_inv`.
        3. Form the LOO means and variances from α and diag(Σ_y⁻¹).
        4. Sum the per-target Gaussian log-densities.

    Args:
        operator: Training covariance Σ_y, noise included.
        y: Training targets y.
        solver: Optional solve strategy for α = Σ_y⁻¹ y and for the
            ``diag_inv`` computation. When ``None``, falls back to
            structural dispatch.
        diag_inv_method: Method passed to `diag_inv`. Defaults to
            ``"solve"`` (exact, deterministic, and honours ``solver``)
            rather than `diag_inv`'s own ``"auto"``, which switches to the
            stochastic Hutchinson estimator for unstructured ``N > 2048``.
            ``"cholesky"`` is equally exact and deterministic and is
            usually faster for a dense Σ_y (one factorisation instead
            of N solves); ``"auto"`` is preferable for structured
            operators with an exact fast path (`Kronecker`,
            `BlockTriDiag`, ...).
        diag_inv_num_probes: Number of Hutchinson probes when
            ``diag_inv_method="hutchinson"``.
        diag_inv_key: PRNG key for probe generation when
            ``diag_inv_method="hutchinson"``.

    Returns:
        A `LOOResult` holding the LOO log-likelihood and the LOO
        predictive means and variances.

    Examples:
        For a diagonal Σ_y the targets are independent, so leaving one out
        predicts 𝒩(0, [Σ_y]_nn):

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.MatrixLinearOperator(
        ...     jnp.diag(jnp.array([1.0, 2.0, 4.0])), lx.positive_semidefinite_tag
        ... )
        >>> y = jnp.array([0.5, -1.0, 2.0])
        >>> result = gaussx.leave_one_out_cv(covariance, y)
        >>> bool(jnp.allclose(result.loo_means, 0.0, atol=1e-6))
        True
        >>> bool(jnp.allclose(result.loo_variances, jnp.array([1.0, 2.0, 4.0])))
        True

        The Cholesky method agrees with the default:

        >>> covariance = lx.MatrixLinearOperator(
        ...     jnp.array([[2.0, 0.5, 0.1], [0.5, 2.0, 0.5], [0.1, 0.5, 2.0]]),
        ...     lx.positive_semidefinite_tag,
        ... )
        >>> exact = gaussx.leave_one_out_cv(covariance, y)
        >>> factored = gaussx.leave_one_out_cv(
        ...     covariance, y, diag_inv_method="cholesky"
        ... )
        >>> bool(jnp.allclose(exact.loo_log_likelihood, factored.loo_log_likelihood))
        True
    """
    # α = Σ_y⁻¹ y
    alpha = dispatch_solve(operator, y, solver)
    # [Σ_y⁻¹]_nn
    inverse_diagonal = diag_inv(
        operator,
        method=diag_inv_method,
        num_probes=diag_inv_num_probes,
        key=diag_inv_key,
        solver=solver,
    )

    # α_n / [Σ_y⁻¹]_nn
    standardised_alpha = alpha / inverse_diagonal
    # μ_LOO,n = y_n − α_n / [Σ_y⁻¹]_nn
    loo_means = y - standardised_alpha
    # σ²_LOO,n = 1 / [Σ_y⁻¹]_nn
    loo_variances = 1.0 / inverse_diagonal

    # log σ²_LOO,n
    log_variances = jnp.log(loo_variances)
    # α_n²
    alpha_squared = alpha**2
    # (y_n − μ_LOO,n)² / σ²_LOO,n = α_n² / [Σ_y⁻¹]_nn
    squared_standardised_residual = alpha_squared / inverse_diagonal
    # log σ²_LOO,n + (y_n − μ_LOO,n)² / σ²_LOO,n
    data_terms = log_variances + squared_standardised_residual
    # log 2π
    log_two_pi = jnp.log(2.0 * jnp.pi)
    # log σ²_LOO,n + (y_n − μ_LOO,n)² / σ²_LOO,n + log 2π
    terms = data_terms + log_two_pi
    # ∑_n (…)
    total = jnp.sum(terms)
    # LOO-CV = −½ ∑_n (…)
    loo_log_likelihood = -0.5 * total

    return LOOResult(
        loo_log_likelihood=loo_log_likelihood,
        loo_means=loo_means,
        loo_variances=loo_variances,
    )
