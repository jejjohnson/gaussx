"""Dispatch helpers: route through solver strategy or fall back to primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._strategies._base import AbstractLogdetStrategy, AbstractSolveStrategy


def dispatch_solve(
    operator: lx.AbstractLinearOperator,
    vector: jnp.ndarray,
    solver: AbstractSolveStrategy | None = None,
) -> jnp.ndarray:
    """Solve ``A x = b`` via *solver* or structural-dispatch primitive.

    Args:
        operator: The linear operator A.
        vector: Right-hand side b.
        solver: Optional solve strategy. When ``None``, falls back
            to :func:`gaussx.solve` (structural dispatch).

    Returns:
        Solution x.
    """
    if solver is not None:
        return solver.solve(operator, vector)
    from gaussx._primitives._solve import solve

    return solve(operator, vector)


def dispatch_logdet(
    operator: lx.AbstractLinearOperator,
    solver: AbstractLogdetStrategy | None = None,
    *,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """Compute ``log |det(A)|`` via *solver* or structural-dispatch primitive.

    Args:
        operator: The linear operator A.
        solver: Optional logdet strategy. When ``None``, falls back
            to :func:`gaussx.logdet` (structural dispatch).
        key: Optional PRNG key for stochastic estimators.

    Returns:
        Scalar ``log |det(A)|``.
    """
    if solver is not None:
        return solver.logdet(operator, key=key)
    from gaussx._primitives._logdet import logdet

    return logdet(operator)
