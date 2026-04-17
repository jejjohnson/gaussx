"""Dense solver strategy: Cholesky for PSD, LU otherwise."""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float

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
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        return _solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        return _logdet(operator)
