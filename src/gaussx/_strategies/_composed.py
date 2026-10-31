"""Composed solver strategy: x = A⁻¹ b and log |det A| from two strategies."""

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
