"""Adapter from a lineax solver to a gaussx solve strategy (gh-376)."""

from __future__ import annotations

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import AbstractSolveStrategy


class LineaxSolver(AbstractSolveStrategy):
    """A lineax solver as a gaussx solve strategy.

    ``solver=`` means a gaussx strategy everywhere except the fallback of
    `gaussx.solve`, which historically takes a lineax solver. Both
    `gaussx.linear_solve` and `gaussx.dispatch_solve` wrap a bare
    ``lineax.AbstractLinearSolver`` in this class, so either kind works in
    either place.

    Attributes:
        solver: The lineax solver, e.g. ``lineax.CG(rtol=1e-6, atol=1e-6)``.
            Static configuration, like every strategy's.
    """

    solver: lx.AbstractLinearSolver = eqx.field(static=True)

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """``lineax.linear_solve(operator, vector, solver).value``.

        Args:
            operator: Linear operator ``A``.
            vector: Right-hand side ``b``, shape ``(n,)``.

        Returns:
            Solution ``x``, shape ``(n,)``.
        """
        return lx.linear_solve(operator, vector, self.solver).value


def as_solve_strategy(solver: object) -> AbstractSolveStrategy | None:
    """Normalise a ``solver=`` argument into a gaussx solve strategy.

    Args:
        solver: ``None``, a gaussx `AbstractSolveStrategy`, or a
            ``lineax.AbstractLinearSolver`` (wrapped in `LineaxSolver`).

    Returns:
        The strategy, or ``None``.

    Raises:
        TypeError: For anything else, naming both accepted types.
    """
    if solver is None or isinstance(solver, AbstractSolveStrategy):
        return solver
    if isinstance(solver, lx.AbstractLinearSolver):
        return LineaxSolver(solver)
    raise TypeError(
        "solver must be a gaussx.AbstractSolveStrategy (e.g. gaussx.CGSolver()) "
        "or a lineax.AbstractLinearSolver (e.g. lineax.CG(...)); got "
        f"{type(solver).__name__}."
    )
