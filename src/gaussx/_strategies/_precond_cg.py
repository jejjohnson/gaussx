"""Preconditioned CG solver with pivoted partial Cholesky preconditioner."""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners import PartialCholeskyPreconditioner
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._slq_logdet import SLQLogdet


class PreconditionedCGSolver(AbstractSolverStrategy):
    """CG solver with pivoted partial Cholesky preconditioner.

    Uses matfree's ``low_rank.cholesky_partial_pivot`` to build a
    rank-k preconditioner, then solves ``(sI + LL^T)^{-1} v`` via
    the Woodbury identity inside lineax CG.

    For operators of the form ``K + sigma^2 I``, preconditioning
    dramatically reduces the number of CG iterations.

    Attributes:
        preconditioner_rank: Rank of the partial Cholesky. Set to 0
            to disable preconditioning (falls back to plain CG).
        shift: Diagonal shift ``s`` for the preconditioner.
            Typically the noise variance ``sigma^2``.
        rtol: Relative tolerance for CG.
        atol: Absolute tolerance for CG.
        max_steps: Maximum CG iterations.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Lanczos iterations for SLQ logdet.
        seed: Seed for probe vector generation.
    """

    preconditioner_rank: int = 50
    shift: float = 1.0
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 1000
    num_probes: int = 20
    lanczos_order: int = 30
    seed: int = 0

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b via preconditioned CG.

        Args:
            operator: A PSD linear operator.
            vector: The right-hand side b.

        Returns:
            The solution x.
        """
        preconditioner = PartialCholeskyPreconditioner(
            rank=self.preconditioner_rank, shift=self.shift
        )
        return CGSolver(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            preconditioner=preconditioner,
        ).solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Stochastic log-determinant via Lanczos quadrature.

        Args:
            operator: A PSD linear operator.

        Returns:
            Scalar estimate of log |det(A)|.
        """
        return SLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
            seed=self.seed,
        ).logdet(operator, key=key)
