"""Dispatch helpers: route through solver strategy or fall back to primitives."""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import AbstractLogdetStrategy, AbstractSolveStrategy


def dispatch_solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, " n"]:
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
) -> Float[Array, ""]:
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
