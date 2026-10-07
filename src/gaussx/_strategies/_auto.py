"""Automatic solver strategy: selects algorithm based on operator type + tags."""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners import AbstractPreconditioner
from gaussx._strategies._base import AbstractSolverStrategy


class AutoSolver(AbstractSolverStrategy):
    """Automatic solver selection based on operator type and size.

    Selection logic:

    - Structured, i.e. any operator with an exact structural solve and
      log-determinant (Diagonal,
      DiagonalizedOperator, BlockDiag, Kronecker, LowRankUpdate,
      KroneckerSum, SpectralFunction, BlockTriDiag and its bidiagonal
      factors, eigen-reducible sums of Kronecker products; also inside
      ``TaggedLinearOperator``, ``c * A``, ``A / c`` and ``-A``):
      DenseSolver, whose structural dispatch is exact and cheap
    - Small dense (N <= size_threshold): DenseSolver
    - Large PSD: CGSolver. Its ``logdet`` is a **stochastic**, fixed-seed
      SLQ estimate (the same probes on every call unless a ``key`` is
      passed, e.g. through `gaussx.KeyedSolver`); for an exact one use
      ``ComposedSolver(CGSolver(), DenseLogdet())``
    - Large general: DenseSolver (fallback)

    Attributes:
        size_threshold: Matrix dimension above which iterative
            solvers are preferred. Default: 1000.
        throw: Forwarded to the `CGSolver` built for large PSD operators:
            raise when CG does not converge (the default), or return the
            last iterate unchecked.
        preconditioner: Forwarded to the `CGSolver` built for large PSD
            operators; ignored when a direct solve is chosen (gh-390).
    """

    size_threshold: int = eqx.field(static=True, default=1000)
    throw: bool = eqx.field(static=True, default=True)
    preconditioner: AbstractPreconditioner | None = None

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b with automatically selected algorithm.

        Args:
            operator: The linear operator A.
            vector: The right-hand side b.

        Returns:
            The solution x.
        """
        return self._get_strategy(operator).solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Compute log |det(A)| with automatically selected algorithm.

        Args:
            operator: The linear operator A.
            key: Optional PRNG key (forwarded to stochastic strategies).

        Returns:
            Scalar log |det(A)|.
        """
        return self._get_strategy(operator).logdet(operator, key=key)

    def _get_strategy(
        self, operator: lx.AbstractLinearOperator
    ) -> AbstractSolverStrategy:
        """Select the best solver strategy for the given operator."""
        from gaussx._primitives._logdet import _has_structural_logdet
        from gaussx._strategies._cg import CGSolver
        from gaussx._strategies._dense import DenseSolver

        # Exact structural solve + logdet (gh-321). A sum of Kronecker
        # products only counts when the exact two-term reduction applies;
        # without it ``gaussx.solve`` would materialize, so it falls through
        # to the size/tag rules below and can still pick up CG.
        if _has_structural_logdet(operator):
            return DenseSolver()

        n = operator.in_size()
        if n <= self.size_threshold:
            return DenseSolver()

        # Large operators: use CG for PSD, DenseSolver otherwise
        if lx.is_positive_semidefinite(operator):
            return CGSolver(throw=self.throw, preconditioner=self.preconditioner)

        return DenseSolver()
