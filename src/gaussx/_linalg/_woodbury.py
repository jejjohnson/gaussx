"""Standalone Woodbury identity solve."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._primitives._solve import solve


def woodbury_solve(
    base: lx.AbstractLinearOperator,
    U: jnp.ndarray,
    D: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Standalone Woodbury identity solve: ``(L + U diag(D) U^T)^{-1} b``.

    Convenience function for cases where the user has the components
    but doesn't want to construct a ``LowRankUpdate`` operator.

    Uses the identity::

        (L + U D U^T)^{-1} b = L^{-1}b - L^{-1}U C^{-1} U^T L^{-1}b

    where ``C = D^{-1} + U^T L^{-1} U`` is the ``(k, k)`` capacitance
    matrix.

    Args:
        base: Base operator L, shape ``(N, N)``.
        U: Low-rank factor, shape ``(N, k)``.
        D: Diagonal scaling, shape ``(k,)``.
        b: Right-hand side, shape ``(N,)``.

    Returns:
        Solution x, shape ``(N,)``.
    """
    from gaussx._operators._low_rank_update import LowRankUpdate

    op = LowRankUpdate(base, U, D)
    return solve(op, b)
