"""Composed solver strategy: mix-and-match solve and logdet."""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import (
    AbstractLogdetStrategy,
    AbstractSolverStrategy,
    AbstractSolveStrategy,
)


class ComposedSolver(AbstractSolverStrategy):
    """Mix-and-match solve and logdet from different strategies.

    This lets you pair, e.g., an exact dense solve with a stochastic
    log-determinant estimator, or an iterative CG solve with a
    closed-form Kronecker log-determinant.

    Accepts either fine-grained protocols (`AbstractSolveStrategy`,
    `AbstractLogdetStrategy`) or full solver strategies.

    Attributes:
        solve_strategy: Strategy whose ``.solve()`` method will be used.
        logdet_strategy: Strategy whose ``.logdet()`` method will be used.

    Examples:

        solver = ComposedSolver(
            solve_strategy=DenseSolver(),
            logdet_strategy=SLQLogdet(num_probes=50, lanczos_order=30),
        )
    """

    solve_strategy: AbstractSolveStrategy
    logdet_strategy: AbstractLogdetStrategy

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve ``A x = b`` by delegating to ``solve_strategy``.

        Args:
            operator: Linear operator ``A``.
            vector: Right-hand side ``b``, shape ``(n,)``.

        Returns:
            Solution ``x``, shape ``(n,)``.
        """
        return self.solve_strategy.solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Compute ``log|det(A)|`` by delegating to ``logdet_strategy``.

        Args:
            operator: Linear operator ``A``.
            key: Optional PRNG key forwarded to stochastic estimators.

        Returns:
            Scalar log-determinant.
        """
        return self.logdet_strategy.logdet(operator, key=key)
