"""Abstract solver strategy protocol."""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import lineax as lx


class AbstractSolverStrategy(eqx.Module):
    """Protocol for solver strategies that pair solve + logdet.

    A solver strategy encapsulates the choice of algorithm for
    solving linear systems and computing log-determinants. This
    decouples distribution objects from solver implementation
    details.
    """

    @abc.abstractmethod
    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve A x = b."""
        ...

    @abc.abstractmethod
    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
    ) -> jnp.ndarray:
        """Compute log |det(A)|."""
        ...
