"""Automatic solver strategy: selects algorithm based on operator type + tags."""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import AbstractSolverStrategy


class AutoSolver(AbstractSolverStrategy):
    """Automatic solver selection based on operator type and size.

    Selection logic:

    - Structured (Diagonal, BlockDiag, Kronecker, LowRankUpdate):
      DenseSolver (structural dispatch handles efficiency)
    - Sums of Kronecker products with an exact reduction
      (``A₁ ⊗ B₁ + A₂ ⊗ B₂`` with one term positive definite):
      DenseSolver — the structural dispatch factorizes per factor
    - Small dense (N <= size_threshold): DenseSolver
    - Large PSD: CGSolver
    - Large general: DenseSolver (fallback)

    Attributes:
        size_threshold: Matrix dimension above which iterative
            solvers are preferred. Default: 1000.
    """

    size_threshold: int = 1000

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
        from gaussx._operators._block_diag import BlockDiag
        from gaussx._operators._kronecker import Kronecker
        from gaussx._operators._low_rank_update import LowRankUpdate
        from gaussx._operators._sum_kronecker import _is_eigen_reducible
        from gaussx._strategies._cg import CGSolver
        from gaussx._strategies._dense import DenseSolver

        if isinstance(
            operator, (lx.DiagonalLinearOperator, BlockDiag, Kronecker, LowRankUpdate)
        ):
            return DenseSolver()

        # A sum of Kronecker products only stays structured when the exact
        # two-term reduction applies; without it ``gaussx.solve`` would
        # materialize, so those operators fall through to the size/tag rules
        # below and can still pick up CG.
        if _is_eigen_reducible(operator):
            return DenseSolver()

        n = operator.in_size()
        if n <= self.size_threshold:
            return DenseSolver()

        # Large operators: use CG for PSD, DenseSolver otherwise
        if lx.is_positive_semidefinite(operator):
            return CGSolver()

        return DenseSolver()
