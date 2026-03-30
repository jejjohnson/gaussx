"""Dense solver strategy: Cholesky for PSD, LU otherwise."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._primitives._logdet import logdet as _logdet
from gaussx._primitives._solve import solve as _solve
from gaussx._strategies._base import AbstractSolverStrategy


class DenseSolver(AbstractSolverStrategy):
    """Dense solver strategy using gaussx structural dispatch.

    Delegates to ``gaussx.solve`` and ``gaussx.logdet`` which
    automatically select the best algorithm based on operator
    structure (Diagonal, BlockDiag, Kronecker, LowRankUpdate,
    or dense fallback via lineax).
    """

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        return _solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
    ) -> jnp.ndarray:
        return _logdet(operator)
