"""Automatic solver strategy: pick the algorithm from A's type, size and tags.

A is an N × N operator. Structured A takes exact structural dispatch, small A
a dense factorisation and large PSD A conjugate gradients.
"""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners import AbstractPreconditioner
from gaussx._primitives._logdet import _has_structural_logdet
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._dense import DenseSolver


class AutoSolver(AbstractSolverStrategy):
    """Automatic solver selection from the operator's type, size and tags.

        x = A⁻¹ b,   log |det A|,   by `DenseSolver` or `CGSolver`

    Selection logic:

    - Structured, i.e. any operator with an exact structural solve and
      log-determinant (Diagonal, DiagonalizedOperator, BlockDiag,
      Kronecker, LowRankUpdate, KroneckerSum, SpectralFunction,
      BlockTriDiag and its bidiagonal factors, eigen-reducible sums of
      Kronecker products; also inside ``TaggedLinearOperator``, ``c * A``,
      ``A / c`` and ``-A``): DenseSolver, whose structural dispatch is
      exact and cheap
    - Small dense (N ≤ ``size_threshold``): DenseSolver
    - Large PSD: CGSolver. Its ``logdet`` is a **stochastic**, fixed-seed
      SLQ estimate (the same probes on every call unless a ``key`` is
      passed, e.g. through `gaussx.KeyedSolver`); for an exact one use
      ``ComposedSolver(CGSolver(), DenseLogdet())``
    - Large general: DenseSolver (fallback)

    Attributes:
        size_threshold: Size N above which a PSD operator is solved
            iteratively. Default: 1000.
        throw: Forwarded to the `CGSolver` built for large PSD operators:
            raise when CG does not converge (the default), or return the
            last iterate unchecked.
        preconditioner: Forwarded to the `CGSolver` built for large PSD
            operators; ignored when a direct solve is chosen (gh-390).

    Examples:
        A small operator takes the exact dense path:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> b = jnp.array([1.0, 2.0])
        >>> x = gaussx.AutoSolver().solve(A, b)
        >>> bool(jnp.allclose(matrix @ x, b, atol=1e-5))
        True

        Above ``size_threshold`` a PSD operator goes to CG:

        >>> x = gaussx.AutoSolver(size_threshold=1).solve(A, b)
        >>> bool(jnp.allclose(matrix @ x, b, atol=1e-3))
        True
    """

    size_threshold: int = eqx.field(static=True, default=1000)
    throw: bool = eqx.field(static=True, default=True)
    preconditioner: AbstractPreconditioner | None = None

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b with the automatically selected strategy.

            x = A⁻¹ b

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.
        """
        return self._get_strategy(operator).solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Compute log |det A| with the automatically selected strategy.

            log |det A| = ∑_d log |γ_d|

        with γ_d, d = 1, …, N, the eigenvalues of A; a large PSD A gets the
        stochastic SLQ estimate of `CGSolver`.

        Args:
            operator: Linear operator A, N × N.
            key: Optional PRNG key, forwarded to a stochastic estimator.

        Returns:
            Scalar log |det A|, exact or estimated.
        """
        return self._get_strategy(operator).logdet(operator, key=key)

    def _get_strategy(
        self, operator: lx.AbstractLinearOperator
    ) -> AbstractSolverStrategy:
        """Select the strategy for ``operator``.

        Algorithm:
            1. An exact structural solve and log-determinant: `DenseSolver`.
            2. N ≤ ``size_threshold``: `DenseSolver`.
            3. A large PSD A: `CGSolver`.
            4. Otherwise: `DenseSolver`.

        Args:
            operator: Linear operator A, N × N.

        Returns:
            The strategy to use for A.
        """

        # Exact structural solve + logdet (gh-321). A sum of Kronecker
        # products only counts when the exact two-term reduction applies;
        # without it ``gaussx.solve`` would materialize, so it falls through
        # to the size/tag rules below and can still pick up CG.
        if _has_structural_logdet(operator):
            return DenseSolver()

        size = operator.in_size()
        if size <= self.size_threshold:
            return DenseSolver()

        # Large operators: use CG for PSD, DenseSolver otherwise
        if lx.is_positive_semidefinite(operator):
            return CGSolver(throw=self.throw, preconditioner=self.preconditioner)

        return DenseSolver()
