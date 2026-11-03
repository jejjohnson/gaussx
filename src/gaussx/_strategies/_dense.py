"""Dense solver strategy: structural dispatch, then Cholesky for PSD, LU otherwise.

A is the operator, b the right-hand side and x = A⁻¹ b the solution; the
strategy is exact, with no tolerance and no iteration.
"""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._logdet import logdet as _logdet
from gaussx._primitives._solve import solve as _solve
from gaussx._strategies._base import AbstractSolverStrategy


class DenseSolver(AbstractSolverStrategy):
    """Exact solver strategy through gaussx structural dispatch.

        x = A⁻¹ b,   log |det A|

    Delegates to `gaussx.solve` and `gaussx.logdet`, which pick the
    algorithm from the operator's structure (diagonal, `BlockDiag`,
    `Kronecker`, `LowRankUpdate`, …) and fall back to a dense lineax
    factorisation: Cholesky for a PSD-tagged A, LU otherwise.

    Examples:
        A Kronecker product is solved one factor at a time:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
        >>> A = gaussx.Kronecker(A_1, A_2)
        >>> solver = gaussx.DenseSolver()
        >>> [round(float(v), 4) for v in solver.solve(A, jnp.ones(4))]
        [0.25, 0.2, 0.125, 0.1]
        >>> round(float(solver.logdet(A)), 4)
        7.3778
    """

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b by structural dispatch (`gaussx.solve`).

            x = A⁻¹ b

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.
        """
        return _solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Compute log |det A| by structural dispatch (`gaussx.logdet`).

            log |det A| = ∑_d log |γ_d|

        with γ_d, d = 1, …, N, the eigenvalues of A.

        Args:
            operator: Linear operator A, N × N.
            key: Unused; present for protocol compatibility with stochastic
                estimators.

        Returns:
            Scalar log |det A|, exact.
        """
        return _logdet(operator)
