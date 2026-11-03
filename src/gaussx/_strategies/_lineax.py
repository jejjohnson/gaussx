"""Adapter from a lineax solver to a gaussx solve strategy (gh-376).

A is the operator, b the right-hand side and x = A⁻¹ b the solution, computed
by whichever ``lineax.AbstractLinearSolver`` the caller chose.
"""

from __future__ import annotations

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import AbstractSolveStrategy


class LineaxSolver(AbstractSolveStrategy):
    """A lineax solver as a gaussx solve strategy.

        x = A⁻¹ b,   by ``lineax.linear_solve(A, b, solver)``

    ``solver=`` means a gaussx strategy everywhere except the fallback of
    `gaussx.solve`, which historically takes a lineax solver. Both
    `gaussx.linear_solve` and ``dispatch_solve`` wrap a bare
    ``lineax.AbstractLinearSolver`` in this class, so either kind works in
    either place.

    Attributes:
        solver: The lineax solver, e.g. ``lineax.CG(rtol=1e-6, atol=1e-6)``.
            Static configuration, like every strategy's.

    Examples:
        A Cholesky solve of a PSD matrix:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> b = jnp.array([1.0, 2.0])
        >>> x = gaussx.LineaxSolver(lx.Cholesky()).solve(A, b)
        >>> bool(jnp.allclose(matrix @ x, b, atol=1e-5))
        True
    """

    solver: lx.AbstractLinearSolver = eqx.field(static=True)

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b with the wrapped lineax solver.

            x = A⁻¹ b

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.
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

    Examples:
        >>> import lineax as lx
        >>> from gaussx._strategies._lineax import as_solve_strategy
        >>> type(as_solve_strategy(lx.CG(rtol=1e-6, atol=1e-6))).__name__
        'LineaxSolver'
        >>> as_solve_strategy(None) is None
        True
        >>> as_solve_strategy("cg")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: solver must be a gaussx.AbstractSolveStrategy ...
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
