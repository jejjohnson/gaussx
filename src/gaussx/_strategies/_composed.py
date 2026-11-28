"""Composed solver strategy: x = A⁻¹ b and log |det A| from two strategies."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import (
    AbstractLogdetStrategy,
    AbstractSolverStrategy,
    AbstractSolveStrategy,
    _SolveStats,
)
from gaussx._strategies._bbmm import BBMMSolver
from gaussx._strategies._keyed import KeyedSolver
from gaussx._strategies._slq_logdet import SLQLogdet


if TYPE_CHECKING:
    from gaussx._preconditioners import AbstractPreconditioner


class ComposedSolver(AbstractSolverStrategy):
    """Mix-and-match a solve and a log-determinant from different strategies.

        x = A⁻¹ b   by ``solve_strategy``,   log |det A|   by ``logdet_strategy``

    This pairs, e.g., an exact dense solve with a stochastic log-determinant
    estimator, or an iterative CG solve with a closed-form Kronecker
    log-determinant. Either slot takes a fine-grained protocol
    (`AbstractSolveStrategy`, `AbstractLogdetStrategy`) or a full solver
    strategy.

    Attributes:
        solve_strategy: Strategy whose ``solve`` is used.
        logdet_strategy: Strategy whose ``logdet`` is used.

    Examples:
        A CG solve with an exact log-determinant:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> solver = gaussx.ComposedSolver(
        ...     solve_strategy=gaussx.CGSolver(),
        ...     logdet_strategy=gaussx.DenseLogdet(),
        ... )
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> b = jnp.array([1.0, 2.0])
        >>> bool(jnp.allclose(matrix @ solver.solve(A, b), b, atol=1e-3))
        True
        >>> round(float(solver.logdet(A)), 4)
        2.3979
    """

    solve_strategy: AbstractSolveStrategy
    logdet_strategy: AbstractLogdetStrategy

    def with_preconditioner(
        self, preconditioner: AbstractPreconditioner
    ) -> ComposedSolver:
        """A copy whose solve, and a BBMM log-determinant, use ``preconditioner``.

        The solve part must take it. The log-determinant part takes it too
        when it is a `gaussx.BBMMSolver` or a `gaussx.SLQLogdet` (or a
        `gaussx.KeyedSolver` around one), which use P; any other
        log-determinant is kept.

        Args:
            preconditioner: The preconditioner to attach.

        Returns:
            The `ComposedSolver` with ``preconditioner`` attached.

        Raises:
            ValueError: If the solve part cannot take it, or a part already
                carries one.
        """
        solve_strategy = self.solve_strategy.with_preconditioner(preconditioner)
        logdet_strategy = self.logdet_strategy
        if logdet_strategy is self.solve_strategy and isinstance(
            solve_strategy, AbstractLogdetStrategy
        ):
            # one strategy in both places stays one object
            logdet_strategy = solve_strategy
        elif logdet_takes_preconditioner(logdet_strategy):
            logdet_strategy = logdet_strategy.with_preconditioner(preconditioner)
        return dataclasses.replace(
            self, solve_strategy=solve_strategy, logdet_strategy=logdet_strategy
        )

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b with ``solve_strategy``.

            x = A⁻¹ b

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.
        """
        return self.solve_strategy.solve(operator, vector)

    def _solve_with_stats(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> _SolveStats:
        """`solve`, with the stats of ``solve_strategy``.

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            The solution and how the solve ended.
        """
        return self.solve_strategy._solve_with_stats(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Compute log |det A| with ``logdet_strategy``.

            log |det A| = ∑_d log |γ_d|

        with γ_d, d = 1, …, N, the eigenvalues of A.

        Args:
            operator: Linear operator A, N × N.
            key: Optional PRNG key, forwarded to a stochastic estimator.

        Returns:
            Scalar log |det A|, exact or estimated.
        """
        return self.logdet_strategy.logdet(operator, key=key)


def logdet_takes_preconditioner(strategy: AbstractLogdetStrategy) -> bool:
    """Whether ``strategy``'s log-determinant uses an attached preconditioner.

    Args:
        strategy: A log-determinant strategy.

    Returns:
        ``True`` for a `gaussx.BBMMSolver`, whose shared pass uses P, for a
        `gaussx.SLQLogdet`, which whitens A by P (gh-825), and for a
        `gaussx.KeyedSolver` around either.
    """
    if isinstance(strategy, KeyedSolver):
        return logdet_takes_preconditioner(strategy.strategy)
    return isinstance(strategy, BBMMSolver | SLQLogdet)
