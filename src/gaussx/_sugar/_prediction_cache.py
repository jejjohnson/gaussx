"""Prediction cache: solve once, predict many.

Caches the training solve ``alpha = K_y^{-1} y`` so that predictions at
multiple test sets reuse the same expensive linear solve.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve


class PredictionCache(eqx.Module):
    """Cached training solve for amortized predictions.

    Stores ``alpha = K_y^{-1} y`` so that downstream predictions only
    require a matrix-vector product rather than a fresh solve.

    Attributes:
        alpha: Solved weights ``K_y^{-1} y``, shape ``(N,)``.
    """

    alpha: jnp.ndarray


def build_prediction_cache(
    operator: lx.AbstractLinearOperator,
    y: jnp.ndarray,
    *,
    solver: AbstractSolveStrategy | None = None,
) -> PredictionCache:
    """Solve ``A alpha = y`` and cache the result.

    Args:
        operator: Training covariance operator ``K_y``, shape ``(N, N)``.
        y: Training targets, shape ``(N,)``.
        solver: Optional solve strategy. When ``None``, falls back
            to structural-dispatch :func:`gaussx.solve`.

    Returns:
        A :class:`PredictionCache` holding the solved weights.
    """
    alpha = dispatch_solve(operator, y, solver)
    return PredictionCache(alpha=alpha)


def predict_mean(
    cache: PredictionCache,
    K_cross: jnp.ndarray,
) -> jnp.ndarray:
    """Predictive mean: ``mu* = K_*f @ alpha``.

    Args:
        cache: Prediction cache from :func:`build_prediction_cache`.
        K_cross: Cross-covariance matrix, shape ``(Nt, N)``.

    Returns:
        Predictive mean, shape ``(Nt,)``.
    """
    return K_cross @ cache.alpha


def predict_variance(
    K_cross: jnp.ndarray,
    K_test_diag: jnp.ndarray,
    operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolveStrategy | None = None,
) -> jnp.ndarray:
    """Predictive variance: ``sigma^2* = k_** - diag(K_*f K_y^{-1} K_f*)``.

    For each test point *i*, solves ``K_y v_i = K_cross[i, :]`` and
    computes ``sigma^2_i = K_test_diag[i] - K_cross[i, :] @ v_i``.

    Args:
        K_cross: Cross-covariance matrix, shape ``(Nt, N)``.
        K_test_diag: Prior variance at test points, shape ``(Nt,)``.
        operator: Training covariance operator ``K_y``, shape ``(N, N)``.
        solver: Optional solve strategy. When ``None``, falls back
            to structural-dispatch :func:`gaussx.solve`.

    Returns:
        Predictive variance, shape ``(Nt,)``.
    """

    def _solve_row(row: jnp.ndarray) -> jnp.ndarray:
        return dispatch_solve(operator, row, solver)

    V = jax.vmap(_solve_row)(K_cross)
    return K_test_diag - jnp.sum(K_cross * V, axis=1)
