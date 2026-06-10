"""Pivoted partial-Cholesky preconditioner."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import matfree.low_rank
from jaxtyping import Array, Float

from gaussx._preconditioners._base import AbstractPreconditioner


class PartialCholeskyPreconditioner(AbstractPreconditioner):
    """Preconditioner from a pivoted partial Cholesky factor.

    Builds a rank-``k`` partial Cholesky factor ``L`` of the system operator via
    matfree, then applies ``(s I + L L^T)^{-1}`` through the Woodbury identity.
    For operators of the form ``K + sigma^2 I`` this dramatically reduces CG
    iteration counts.

    Attributes:
        rank: Rank of the partial Cholesky. ``<= 0`` disables preconditioning
            (:meth:`as_operator` returns ``None``).
        shift: Diagonal shift ``s`` for the preconditioner, typically the noise
            variance ``sigma^2``.
    """

    rank: int = 50
    shift: float = 1.0

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator | None:
        """Build the Woodbury preconditioner operator from *operator*."""
        if self.rank <= 0:
            return None
        if operator is None:
            raise ValueError(
                "PartialCholeskyPreconditioner.as_operator requires the system "
                "operator to build its factor."
            )

        n = operator.in_size()
        rank = min(self.rank, n)

        def mat_el(i, j):
            ej = jnp.zeros(n).at[j].set(1.0)
            return operator.mv(ej)[i]

        chol_fn = matfree.low_rank.cholesky_partial_pivot(mat_el, nrows=n, rank=rank)
        factor, info = chol_fn()
        precond_fn = matfree.low_rank.preconditioner(lambda: (factor, info))

        def precond_matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
            applied, _ = precond_fn(v, self.shift)
            return applied

        return lx.FunctionLinearOperator(
            precond_matvec,
            operator.out_structure(),
            lx.positive_semidefinite_tag,
        )
