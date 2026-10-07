"""Prediction cache: solve once, predict many.

Caches the training solve ``alpha = K_y^{-1} y`` and, where it is cheap to
keep, the Cholesky factor of ``K_y``, so that predictions at multiple test
sets reuse the same factorisation for both the mean and the variance.
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import rearrange, reduce
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
    """Cached training solve (and factor) for amortized predictions.

    A cache is tied to the operator it was built from: rebuild it whenever
    the kernel hyperparameters (and so ``K_y``) change.

    Attributes:
        alpha: Solved weights ``K_y^{-1} y``, shape ``(N,)``.
        operator: The training covariance ``K_y`` the cache was built
            from, or ``None`` for a cache constructed from ``alpha`` alone.
        factor: Lower Cholesky factor ``L`` of ``K_y`` (structured where
            `gaussx.cholesky` preserves structure), or ``None`` when
            ``K_y`` is not factorised (iterative solvers, structured
            operators with their own solve).
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
    """Solve ``K_y alpha = y`` and cache the result.

    For a PSD dense, diagonal, `Kronecker` or `BlockDiag` operator with
    ``solver`` ``None`` or `DenseSolver`, computes ``L = cholesky(K_y)``
    once, solves ``alpha = L^{-T} L^{-1} y`` through it and stores ``L``
    so that `predict_variance` needs only triangular solves. Otherwise
    (iterative strategies, other structured operators) ``alpha`` comes
    from ``solver`` and ``factor`` is ``None``.

    Args:
        operator: Training covariance operator ``K_y``, shape ``(N, N)``.
        y: Training targets, shape ``(N,)``.
        solver: Optional solve strategy. When ``None``, falls back
            to structural-dispatch `gaussx.solve`.

    Returns:
        A `PredictionCache` holding ``alpha``, ``K_y`` and, where
        available, its Cholesky factor.
    """
    if (
        isinstance(operator, _FACTORABLE)
        and lx.is_positive_semidefinite(operator)
        and (solver is None or isinstance(solver, DenseSolver))
    ):
        L = cholesky(operator)
        alpha = solve(L.transpose(), solve(L, y))
        return PredictionCache(alpha=alpha, operator=operator, factor=L)
    alpha = dispatch_solve(operator, y, solver)
    return PredictionCache(alpha=alpha, operator=operator)


def predict_mean(
    cache: PredictionCache,
    K_cross: Float[Array, "Nt N"],
) -> Float[Array, " Nt"]:
    """Predictive mean: ``mu* = K_*f @ alpha``.

    Args:
        cache: Prediction cache from `build_prediction_cache`.
        K_cross: Cross-covariance matrix, shape ``(Nt, N)``.

    Returns:
        Predictive mean, shape ``(Nt,)``.
    """
    return K_cross @ cache.alpha


def predict_variance(
    cache: PredictionCache | Float[Array, "Nt N"] | None = None,
    K_cross: Float[Array, "Nt N"] | Float[Array, " Nt"] | None = None,
    K_test_diag: Float[Array, " Nt"] | lx.AbstractLinearOperator | None = None,
    *,
    operator: lx.AbstractLinearOperator | None = None,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, " Nt"]:
    r"""Predictive variance ``sigma^2_* = k_** - diag(K_*f K_y^{-1} K_f*)``.

    Call as ``predict_variance(cache, K_cross, K_test_diag)``, with the
    cache first as in `predict_mean`. With a cached factor ``L``:

    $$
    V = L^{-1} K_{*f}^\top, \qquad
    \sigma^2_{*,i} = k_{**,i} - \sum_n V_{ni}^2,
    $$

    triangular solves only (no refactorisation of ``K_y``). Without a
    factor, solves ``K_y v_i = K_cross[i, :]`` for each test row with
    ``solver`` and returns ``k_{**,i} - K_cross[i, :] v_i``.

    The old order ``predict_variance(K_cross, K_test_diag, operator)``
    (or ``operator=`` as a keyword) still works for one release, emits a
    `DeprecationWarning`, and refactorises ``K_y`` on every call.

    Args:
        cache: Prediction cache from `build_prediction_cache`.
        K_cross: Cross-covariance matrix, shape ``(Nt, N)``.
        K_test_diag: Prior variance at test points, shape ``(Nt,)``.
        operator: Deprecated; only for the old call form.
        solver: Optional solve strategy for the no-factor path. When
            ``None``, falls back to structural-dispatch `gaussx.solve`.

    Returns:
        Predictive variance, shape ``(Nt,)``.

    Raises:
        ValueError: If the cache holds neither a factor nor an operator.
    """
    from gaussx._linalg._linalg import solve_columns, solve_rows

    if not isinstance(cache, PredictionCache):
        warn_deprecated(
            "predict_variance(K_cross, K_test_diag, operator) is deprecated "
            "and will be removed in gaussx 0.7.0; build a cache "
            "with build_prediction_cache(operator, y) and call "
            "predict_variance(cache, K_cross, K_test_diag), which reuses the "
            "cached factorisation."
        )
        if operator is None:  # all positional: (K_cross, K_test_diag, operator)
            old = (cache, K_cross, K_test_diag)
        elif cache is not None:  # (K_cross, [K_test_diag], operator=...)
            old = (cache, K_cross if K_test_diag is None else K_test_diag, operator)
        else:  # all keywords
            old = (K_cross, K_test_diag, operator)
        if not isinstance(old[2], lx.AbstractLinearOperator):
            msg = "predict_variance needs a PredictionCache or an operator."
            raise TypeError(msg)
        Kc, kd = cast(Array, old[0]), cast(Array, old[1])
        V = solve_rows(old[2], Kc, solver=solver)
        return kd - reduce(Kc * V, "t n -> t", "sum")

    if operator is not None:
        msg = "operator= is only for the deprecated call form; the cache holds K_y."
        raise TypeError(msg)
    Kc, kd = cast(Array, K_cross), cast(Array, K_test_diag)
    if cache.factor is not None:
        V = solve_columns(cache.factor, rearrange(Kc, "t n -> n t"))
        return kd - reduce(V**2, "n t -> t", "sum")
    if cache.operator is None:
        msg = (
            "This PredictionCache holds only alpha; build it with "
            "build_prediction_cache(operator, y) to predict variances."
        )
        raise ValueError(msg)
    V = solve_rows(cache.operator, Kc, solver=solver)
    return kd - reduce(Kc * V, "t n -> t", "sum")
