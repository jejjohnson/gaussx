"""Prediction cache: solve once, predict many.

Caches the training solve α = Σ_y⁻¹ y and, where it is cheap to keep, the
lower Cholesky factor L_y of Σ_y = L_y L_yᵀ, so that predictions at several
test sets reuse one factorisation for both the mean and the variance. Here
Σ_y is the N × N covariance of the training targets y (K_XX + σ_ε² I for a
GP), K_*X the M × N cross-covariance between the M test inputs and the
training inputs, and K_** the prior kernel matrix at the test inputs.
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import rearrange, reduce
from gaussx._linalg._linalg import solve_columns, solve_rows
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dense import DenseSolver
from gaussx._strategies._dispatch import dispatch_solve


# Operators whose `gaussx.cholesky` keeps their structure (dense, diagonal,
# per-factor Kronecker, per-block BlockDiag). Others (low-rank updates,
# Toeplitz, ...) keep their structural solve and get no cached factor.
_FACTORABLE = (
    lx.MatrixLinearOperator,
    lx.DiagonalLinearOperator,
    Kronecker,
    BlockDiag,
)


class PredictionCache(eqx.Module):
    """Cached training solve α = Σ_y⁻¹ y (and factor L_y) for amortized predictions.

    A cache is tied to the operator it was built from: rebuild it whenever
    the kernel hyperparameters (and so Σ_y) change.

    Attributes:
        alpha: Solved weights α = Σ_y⁻¹ y, one per training target.
        operator: The training covariance Σ_y the cache was built from, or
            ``None`` for a cache constructed from ``alpha`` alone.
        factor: Lower Cholesky factor L_y of Σ_y (structured where
            `gaussx.cholesky` preserves structure), or ``None`` when Σ_y is
            not factorised (iterative solvers, structured operators with
            their own solve).
    """

    alpha: Float[Array, " N"]
    operator: lx.AbstractLinearOperator | None = None
    factor: lx.AbstractLinearOperator | None = None


def build_prediction_cache(
    operator: lx.AbstractLinearOperator,
    y: Float[Array, " N"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> PredictionCache:
    """Solve Σ_y α = y and cache the result.

        α = Σ_y⁻¹ y = L_y⁻ᵀ L_y⁻¹ y,   Σ_y = L_y L_yᵀ

    For a PSD dense, diagonal, `Kronecker` or `BlockDiag` operator with
    ``solver`` ``None`` or `DenseSolver`, the factor L_y is computed once,
    α comes from two triangular solves with it, and L_y is stored so that
    `predict_variance` needs only triangular solves. Otherwise (iterative
    strategies, other structured operators) α comes from ``solver`` and
    ``factor`` is ``None``.

    Algorithm:
        1. If Σ_y is factorable, PSD and the solver is dense, factorise
           Σ_y = L_y L_yᵀ with `gaussx.cholesky`.
        2. Solve L_y w = y, then L_yᵀ α = w, and keep L_y.
        3. Otherwise solve Σ_y α = y with ``solver`` and keep no factor.

    Args:
        operator: Training covariance Σ_y.
        y: Training targets y.
        solver: Optional solve strategy. When ``None``, falls back
            to structural-dispatch `gaussx.solve`.

    Returns:
        A `PredictionCache` holding α, Σ_y and, where available, its
        Cholesky factor L_y.

    Examples:
        A dense PSD Σ_y keeps its Cholesky factor:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.MatrixLinearOperator(
        ...     jnp.diag(jnp.array([2.0, 4.0])), lx.positive_semidefinite_tag
        ... )
        >>> cache = gaussx.build_prediction_cache(covariance, jnp.array([2.0, 4.0]))
        >>> bool(jnp.allclose(cache.alpha, 1.0)), cache.factor is not None
        (True, True)

        An iterative solver gives the same weights without a factor:

        >>> iterative = gaussx.build_prediction_cache(
        ...     covariance, jnp.array([2.0, 4.0]), solver=gaussx.CGSolver()
        ... )
        >>> bool(jnp.allclose(iterative.alpha, 1.0)), iterative.factor is None
        (True, True)
    """
    if (
        isinstance(operator, _FACTORABLE)
        and lx.is_positive_semidefinite(operator)
        and (solver is None or isinstance(solver, DenseSolver))
    ):
        # Σ_y = L_y L_yᵀ
        factor = cholesky(operator)
        # L_yᵀ, taken before the first solve so the traced order (and with
        # it the order in which gradients accumulate) is unchanged
        factor_t = factor.transpose()
        # w = L_y⁻¹ y
        forward = solve(factor, y)
        # α = L_y⁻ᵀ w = Σ_y⁻¹ y
        alpha = solve(factor_t, forward)
        return PredictionCache(alpha=alpha, operator=operator, factor=factor)
    # α = Σ_y⁻¹ y
    alpha = dispatch_solve(operator, y, solver)
    return PredictionCache(alpha=alpha, operator=operator)


def predict_mean(
    cache: PredictionCache,
    K_cross: Float[Array, "Nt N"],
) -> Float[Array, " Nt"]:
    """Predictive mean of a GP at the test inputs, from a prediction cache.

        μ_* = K_*X Σ_y⁻¹ y = K_*X α

    Args:
        cache: Prediction cache from `build_prediction_cache`, holding α.
        K_cross: Cross-covariance ``K_*X`` between the test inputs and the
            training inputs.

    Returns:
        The predictive means ``μ_*``, one per test input.

    Examples:
        Weights α = (1, 1) and three test inputs:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> cache = gaussx.PredictionCache(alpha=jnp.ones(2))
        >>> K_cross = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> mean = gaussx.predict_mean(cache, K_cross)
        >>> bool(jnp.allclose(mean, jnp.array([1.0, 1.0, 2.0])))
        True
    """
    # μ_* = K_*X α
    return K_cross @ cache.alpha


def predict_variance(
    cache: PredictionCache | Float[Array, "Nt N"] | None = None,
    K_cross: Float[Array, "Nt N"] | Float[Array, " Nt"] | None = None,
    K_test_diag: Float[Array, " Nt"] | lx.AbstractLinearOperator | None = None,
    *,
    operator: lx.AbstractLinearOperator | None = None,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, " Nt"]:
    """Predictive variance of a GP at the test inputs, from a prediction cache.

        σ²_m = [K_**]_mm − [K_*X Σ_y⁻¹ K_X*]_mm
             = [K_**]_mm − ∑_n [V]_nm²,   V = L_y⁻¹ K_X*

    Call as ``predict_variance(cache, K_cross, K_test_diag)``, with the
    cache first as in `predict_mean`. With a cached factor L_y it takes
    triangular solves only, with no refactorisation of Σ_y. Without a
    factor it solves Σ_y v_m = k_mX for each test row k_mX of ``K_*X`` with
    ``solver`` and returns ``[K_**]_mm − k_mXᵀ v_m``.

    The old order ``predict_variance(K_cross, K_test_diag, operator)``
    (or ``operator=`` as a keyword) still works for one release, emits a
    `DeprecationWarning`, and refactorises Σ_y on every call.

    Algorithm:
        1. Route the deprecated call forms to one solve per test row
           against ``operator``.
        2. With a cached factor, whiten ``V = L_y⁻¹ K_X*`` and subtract the
           column sums of V² from the prior variances.
        3. Without one, solve Σ_y⁻¹ k_mX row by row with ``solver`` and
           subtract k_mXᵀ Σ_y⁻¹ k_mX.

    Args:
        cache: Prediction cache from `build_prediction_cache`.
        K_cross: Cross-covariance ``K_*X`` between the test inputs and the
            training inputs.
        K_test_diag: Prior variances ``[K_**]_mm`` at the test inputs, the
            diagonal of ``K_**``.
        operator: Deprecated; only for the old call form, where it is Σ_y.
        solver: Optional solve strategy for the no-factor path. When
            ``None``, falls back to structural-dispatch `gaussx.solve`.

    Returns:
        The predictive variances σ²_m, one per test input.

    Raises:
        TypeError: If the old call form has no operator, or ``operator`` is
            passed alongside a cache.
        ValueError: If the cache holds neither a factor nor an operator.

    Examples:
        Σ_y = 2 I and two test inputs, each seeing one training input:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> cache = gaussx.build_prediction_cache(covariance, jnp.ones(2))
        >>> K_cross = jnp.eye(2)
        >>> K_test_diag = jnp.full(2, 2.0)
        >>> variance = gaussx.predict_variance(cache, K_cross, K_test_diag)
        >>> bool(jnp.allclose(variance, 1.5))
        True

        An iterative cache solves per test row and agrees:

        >>> iterative = gaussx.build_prediction_cache(
        ...     covariance, jnp.ones(2), solver=gaussx.CGSolver()
        ... )
        >>> solved = gaussx.predict_variance(iterative, K_cross, K_test_diag)
        >>> bool(jnp.allclose(solved, variance, atol=1e-5))
        True

        A cache built from α alone cannot predict variances:

        >>> bare = gaussx.PredictionCache(alpha=jnp.ones(2))
        >>> gaussx.predict_variance(
        ...     bare, K_cross, K_test_diag
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: This PredictionCache holds only alpha; ...
    """

    if not isinstance(cache, PredictionCache):
        warn_deprecated(
            "predict_variance(K_cross, K_test_diag, operator) is deprecated "
            "and will be removed in gaussx 0.7.0; build a cache "
            "with build_prediction_cache(operator, y) and call "
            "predict_variance(cache, K_cross, K_test_diag), which reuses the "
            "cached factorisation."
        )
        if operator is None:
            # all positional: (K_cross, K_test_diag, operator)
            legacy = (cache, K_cross, K_test_diag)
        elif cache is not None:
            # (K_cross, [K_test_diag], operator=...)
            legacy = (cache, K_cross if K_test_diag is None else K_test_diag, operator)
        else:
            # all keywords
            legacy = (K_cross, K_test_diag, operator)
        legacy_cross, legacy_prior, legacy_operator = legacy
        if not isinstance(legacy_operator, lx.AbstractLinearOperator):
            msg = "predict_variance needs a PredictionCache or an operator."
            raise TypeError(msg)
        cross_covariance = cast(Array, legacy_cross)
        prior_variance = cast(Array, legacy_prior)
        # K_*X Σ_y⁻¹, one solve per test row
        solved_cross = solve_rows(legacy_operator, cross_covariance, solver=solver)
        # k_mX ⊙ Σ_y⁻¹ k_mX
        products = cross_covariance * solved_cross
        # [K_*X Σ_y⁻¹ K_X*]_mm
        explained = reduce(products, "test sample -> test", "sum")
        # σ²_m = [K_**]_mm − [K_*X Σ_y⁻¹ K_X*]_mm
        return prior_variance - explained

    if operator is not None:
        msg = "operator= is only for the deprecated call form; the cache holds K_y."
        raise TypeError(msg)
    cross_covariance = cast(Array, K_cross)
    prior_variance = cast(Array, K_test_diag)
    if cache.factor is not None:
        # K_X* = K_*Xᵀ
        cross_covariance_t = rearrange(cross_covariance, "test sample -> sample test")
        # V = L_y⁻¹ K_X*
        whitened_cross = solve_columns(cache.factor, cross_covariance_t)
        # V²
        whitened_cross_squared = whitened_cross**2
        # [K_*X Σ_y⁻¹ K_X*]_mm = ∑_n [V]_nm²
        explained = reduce(whitened_cross_squared, "sample test -> test", "sum")
        # σ²_m = [K_**]_mm − ∑_n [V]_nm²
        return prior_variance - explained
    if cache.operator is None:
        msg = (
            "This PredictionCache holds only alpha; build it with "
            "build_prediction_cache(operator, y) to predict variances."
        )
        raise ValueError(msg)
    # K_*X Σ_y⁻¹, one solve per test row
    solved_cross = solve_rows(cache.operator, cross_covariance, solver=solver)
    # k_mX ⊙ Σ_y⁻¹ k_mX
    products = cross_covariance * solved_cross
    # [K_*X Σ_y⁻¹ K_X*]_mm
    explained = reduce(products, "test sample -> test", "sum")
    # σ²_m = [K_**]_mm − [K_*X Σ_y⁻¹ K_X*]_mm
    return prior_variance - explained
