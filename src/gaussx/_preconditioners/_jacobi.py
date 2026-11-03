"""Jacobi (diagonal) preconditioner, M = diag(A).

A is the N × N system operator with diagonal entries A_nn, n = 1, …, N. The
Jacobi preconditioner applies M⁻¹ = diag(1 / |A_nn|) to each CG residual.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._primitives._diag import matrix_free_diag


class JacobiPreconditioner(AbstractPreconditioner):
    """Diagonal preconditioner M⁻¹ = diag(1 / |diag(A)|).

        M = diag(A),   [M⁻¹]_nn = 1 / |A_nn|   (0 where A_nn = 0)

    The cheapest preconditioner: it scales each coordinate by the reciprocal
    of the matching diagonal entry of A. Effective when A is diagonally
    dominant.

    **Cost.** Without an explicit ``diagonal`` it is read from the system
    operator at every solve. A stored matrix and the structured operators
    (`gaussx.diag`: `BlockDiag`, `Kronecker`, `LowRankUpdate`, ..., also
    inside sums and scalings) give it exactly for free. A matrix-free part,
    such as a bare `lineax.FunctionLinearOperator`, is never materialised,
    but its diagonal costs one matvec per row (N matvecs, O(N) memory),
    since Jacobi needs it exactly (gh-361). Pass ``diagonal=`` when it is
    known in closed form, e.g. a kernel's variance.

    A negative diagonal entry is inverted in absolute value, so the
    operator really is positive semi-definite, as CG requires (the usual
    choice for indefinite Jacobi). A zero entry is left unscaled (0), with
    a zero gradient (gh-402).

    Attributes:
        diagonal: The diagonal diag(A). When ``None``, it is extracted from
            the operator passed to `as_operator` (see **Cost**).

    Examples:
        From an explicit diagonal, with a negative and a zero entry:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> jacobi = gaussx.JacobiPreconditioner(
        ...     diagonal=jnp.array([2.0, -4.0, 0.0])
        ... )
        >>> [float(v) for v in jacobi.as_operator().mv(jnp.ones(3))]
        [0.5, 0.25, 0.0]

        From the system operator, inside CG:

        >>> import lineax as lx
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> solver = gaussx.CGSolver(preconditioner=gaussx.JacobiPreconditioner())
        >>> x = solver.solve(A, jnp.array([1.0, 2.0]))
        >>> bool(jnp.allclose(matrix @ x, jnp.array([1.0, 2.0]), atol=1e-3))
        True
    """

    diagonal: Float[Array, " n"] | None = None

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """Return M⁻¹ = diag(1 / |diag(A)|) as a PSD operator.

            [M⁻¹]_nn = 1 / |A_nn|   (0 where A_nn = 0)

        Args:
            operator: The system operator A; read for its diagonal when
                ``diagonal`` is ``None``.

        Returns:
            The diagonal operator M⁻¹, tagged positive semidefinite.

        Raises:
            ValueError: If neither ``diagonal`` nor ``operator`` is given.
        """
        diagonal = self.diagonal
        if diagonal is None:
            if operator is None:
                raise ValueError(
                    "JacobiPreconditioner needs either an explicit `diagonal` "
                    "or an operator to extract one from."
                )
            # A_nn, exact (one matvec per row for a matrix-free part)
            diagonal = matrix_free_diag(operator, estimate=False)
        # Double where: the untaken 1/0 branch would make the gradient NaN.
        nonzero = diagonal != 0.0
        safe_diagonal = jnp.where(nonzero, diagonal, 1.0)
        # [M⁻¹]_nn = 1 / |A_nn|, 0 where A_nn = 0
        inverse_diagonal = jnp.where(nonzero, 1.0 / jnp.abs(safe_diagonal), 0.0)
        return lx.TaggedLinearOperator(
            lx.DiagonalLinearOperator(inverse_diagonal), lx.positive_semidefinite_tag
        )
