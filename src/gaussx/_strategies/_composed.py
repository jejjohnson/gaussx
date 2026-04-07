"""Composed solver strategy: mix-and-match solve and logdet."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

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

    Accepts either fine-grained protocols (:class:`AbstractSolveStrategy`,
    :class:`AbstractLogdetStrategy`) or full solver strategies.

    Args:
        solve_strategy: Strategy whose ``.solve()`` method will be used.
        logdet_strategy: Strategy whose ``.logdet()`` method will be used.

    Example::

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
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve A x = b using the solve strategy."""
        return self.solve_strategy.solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """Compute log |det(A)| using the logdet strategy."""
        return self.logdet_strategy.logdet(operator, key=key)
