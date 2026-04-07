"""Abstract solver strategy protocols."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


class AbstractSolveStrategy(eqx.Module):
    """Protocol for linear solve strategies.

    A solve strategy encapsulates the choice of algorithm for
    solving ``A x = b``.  Separating solve from logdet lets
    users mix-and-match via :class:`ComposedSolver`.
    """

    @abc.abstractmethod
    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve A x = b."""
        ...


class AbstractLogdetStrategy(eqx.Module):
    """Protocol for log-determinant strategies.

    A logdet strategy encapsulates the choice of algorithm for
    computing ``log |det(A)|``.  Separating logdet from solve
    lets users mix-and-match via :class:`ComposedSolver`.

    All implementations accept an optional ``key`` parameter for
    stochastic methods.  Deterministic strategies ignore it.
    """

    @abc.abstractmethod
    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """Compute log |det(A)|.

        Args:
            operator: Linear operator.
            key: Optional PRNG key for stochastic estimators.
        """
        ...


class AbstractSolverStrategy(AbstractSolveStrategy, AbstractLogdetStrategy):
    """Protocol for solver strategies that pair solve + logdet.

    A solver strategy encapsulates the choice of algorithm for
    solving linear systems and computing log-determinants. This
    decouples distribution objects from solver implementation
    details.

    Subclasses must implement both :meth:`solve` and :meth:`logdet`.
    For independent control, see :class:`AbstractSolveStrategy` and
    :class:`AbstractLogdetStrategy`, composable via
    :class:`ComposedSolver`.
    """
