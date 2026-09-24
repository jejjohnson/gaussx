"""Pivoted partial-Cholesky preconditioner."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._primitives._root import guarded_pivoted_cholesky


class PartialCholeskyPreconditioner(AbstractPreconditioner):
    """Preconditioner from a pivoted partial Cholesky factor.

    Builds a rank-``k`` pivoted partial Cholesky factor ``L`` of the system
    operator, then applies ``(s I + L L^T)^{-1}`` through the Woodbury identity.
    For operators of the form ``K + sigma^2 I`` this dramatically reduces CG
    iteration counts.

    The factor is guarded like `gaussx.root_decomposition`'s pivoted Cholesky:
    once ``rank`` exceeds the operator's numerical rank (small datasets,
    duplicated inputs, noiseless kernels), the surplus columns are exactly
    zero instead of NaN or inf, and the preconditioner degrades gracefully to
    the lower-rank one (gh-237).

    Attributes:
        rank: Rank of the partial Cholesky. ``<= 0`` disables preconditioning
            (`as_operator` returns ``None``).
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
        dtype = operator.in_structure().dtype

        def column(k):
            return operator.mv(jnp.zeros(n, dtype=dtype).at[k].set(1.0))

        factor = guarded_pivoted_cholesky(lx.diagonal(operator), column, rank)

        # Woodbury: (sI + L Lᵀ)⁻¹ v = (v - L (sI + Lᵀ L)⁻¹ Lᵀ v) / s. Zero
        # surplus columns only add s to the capacitance diagonal, so it stays
        # positive definite.
        shift = self.shift
        capacitance = jax.scipy.linalg.cho_factor(
            shift * jnp.eye(rank, dtype=dtype) + factor.T @ factor
        )

        def precond_matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
            correction = factor @ jax.scipy.linalg.cho_solve(capacitance, factor.T @ v)
            return (v - correction) / shift

        return lx.FunctionLinearOperator(
            precond_matvec,
            operator.out_structure(),
            lx.positive_semidefinite_tag,
        )
