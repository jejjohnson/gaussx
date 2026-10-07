"""Jacobi (diagonal) preconditioner."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._primitives._diag import matrix_free_diag


class JacobiPreconditioner(AbstractPreconditioner):
    """Diagonal preconditioner ``M^{-1} = diag(1 / |diag(A)|)``.

    The cheapest preconditioner: scales each coordinate by the reciprocal of
    the corresponding diagonal entry of ``A``. Effective when ``A`` is
    diagonally dominant.

    **Cost.** Without an explicit ``diagonal`` it is read from the system
    operator at every solve. A stored matrix and the structured operators
    (`gaussx.diag`: `BlockDiag`, `Kronecker`, `LowRankUpdate`, ..., also
    inside sums and scalings) give it exactly for free. A matrix-free part,
    such as a bare `lineax.FunctionLinearOperator`, is never materialised,
    but its diagonal costs one matvec per row (``n`` matvecs, O(n) memory),
    since Jacobi needs it exactly (gh-361). Pass ``diagonal=`` when it is
    known in closed form, e.g. a kernel's variance.

    A negative diagonal entry is inverted in absolute value, so the
    operator really is positive semi-definite, as CG requires (the usual
    choice for indefinite Jacobi). A zero entry is left unscaled (``0``),
    with a zero gradient (gh-402).

    Attributes:
        diagonal: The diagonal of ``A``. When ``None``, it is extracted from the
            operator passed to `as_operator` (see **Cost**).
    """

    diagonal: Float[Array, " n"] | None = None

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """Return ``diag(1 / d)`` as a PSD operator."""
        d = self.diagonal
        if d is None:
            if operator is None:
                raise ValueError(
                    "JacobiPreconditioner needs either an explicit `diagonal` "
                    "or an operator to extract one from."
                )

            d = matrix_free_diag(operator, estimate=False)
        # Double where: the untaken 1/0 branch would make the gradient NaN.
        nonzero = d != 0.0
        safe = jnp.where(nonzero, d, 1.0)
        inv = jnp.where(nonzero, 1.0 / jnp.abs(safe), 0.0)
        return lx.TaggedLinearOperator(
            lx.DiagonalLinearOperator(inv), lx.positive_semidefinite_tag
        )
