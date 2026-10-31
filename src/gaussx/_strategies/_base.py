"""Abstract strategy protocols: a solve x = A⁻¹ b, a log-determinant log |det A|.

A is a square N × N operator, b ∈ ℝ^N the right-hand side and x ∈ ℝ^N the
solution. A strategy changes how these are computed (a factorisation, a
Krylov iteration, a stochastic estimate), never what they are.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float


class AbstractSolveStrategy(eqx.Module):
    """Protocol for a linear-solve strategy.

        x = A⁻¹ b

    A solve strategy fixes the algorithm for A x = b. Keeping it apart from
    the log-determinant lets `ComposedSolver` pair any solve with any
    log-determinant.

    Examples:
        Every full strategy is also a solve strategy:

        >>> import gaussx
        >>> isinstance(gaussx.CGSolver(), gaussx.AbstractSolveStrategy)
        True
    """

    @abc.abstractmethod
    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b.

            x = A⁻¹ b

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.
        """
        ...


class AbstractLogdetStrategy(eqx.Module):
    """Protocol for a log-determinant strategy.

        log |det A|

    A logdet strategy fixes the algorithm for log |det A|. Keeping it apart
    from the solve lets `ComposedSolver` pair any log-determinant with any
    solve. Every implementation takes an optional ``key`` for its stochastic
    estimators; deterministic strategies ignore it.

    Examples:
        The stochastic Lanczos estimator is a logdet-only strategy:

        >>> import gaussx
        >>> isinstance(gaussx.SLQLogdet(), gaussx.AbstractLogdetStrategy)
        True
        >>> isinstance(gaussx.SLQLogdet(), gaussx.AbstractSolveStrategy)
        False
    """

    @abc.abstractmethod
    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Compute log |det A|.

            log |det A| = ∑_d log |γ_d|

        with γ_d, d = 1, …, N, the eigenvalues of A.

        Args:
            operator: Linear operator A, N × N.
            key: Optional PRNG key for stochastic estimators.

        Returns:
            Scalar log |det A|, exact or estimated.
        """
        ...


class AbstractSolverStrategy(AbstractSolveStrategy, AbstractLogdetStrategy):
    """Protocol for a solver strategy that pairs a solve and a log-determinant.

        x = A⁻¹ b,   log |det A|

    A solver strategy fixes both algorithms, which decouples distributions
    and recipes from how their linear algebra is computed. Subclasses
    implement both `solve` and `logdet`; for independent control use
    `AbstractSolveStrategy` and `AbstractLogdetStrategy`, composed with
    `ComposedSolver`.

    Examples:
        A structural-dispatch strategy behind a solve and a log-determinant:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> solver = gaussx.DenseSolver()
        >>> isinstance(solver, gaussx.AbstractSolverStrategy)
        True
        >>> A = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> [round(float(v), 4) for v in solver.solve(A, jnp.ones(2))]
        [0.5, 0.25]
        >>> round(float(solver.logdet(A)), 4)
        2.0794
    """
