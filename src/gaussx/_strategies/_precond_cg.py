"""Preconditioned CG solver with pivoted partial Cholesky preconditioner."""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners import (
    AbstractPreconditioner,
    PartialCholeskyPreconditioner,
)
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._slq_logdet import SLQLogdet


class PreconditionedCGSolver(AbstractSolverStrategy):
    """CG solver with pivoted partial Cholesky preconditioner.

    Uses `gaussx.PartialCholeskyPreconditioner`'s guarded pivoted
    partial Cholesky to build a rank-k preconditioner, then solves
    ``(σ² I + F Fᵀ)⁻¹ v`` via the Woodbury identity inside lineax CG.

    For operators of the form ``K + σ² I``, preconditioning
    dramatically reduces the number of CG iterations.

    Pass a ``preconditioner`` built once with
    `gaussx.PartialCholeskyPreconditioner.from_operator` or
    `gaussx.NystromPreconditioner.from_operator` (on ``K``, with
    ``shift=σ²``) to reuse it across solves. Otherwise a rank
    ``preconditioner_rank`` factor of ``A − shift · I`` is rebuilt from the
    system operator ``A`` at every solve, so the noise is never counted
    twice (#345).

    Attributes:
        preconditioner_rank: Rank of the partial Cholesky built per solve.
            Set to 0 to disable preconditioning (falls back to plain CG).
            Ignored when ``preconditioner`` is given.
        shift: The noise variance ``σ²`` in the system ``A = K + σ² I``,
            for the preconditioner built per solve. Must not exceed the
            noise actually in ``A``. Ignored when ``preconditioner`` is given.
        rtol: Relative tolerance for CG. ``None``: ``1e-5`` in float64,
            ``1e-3`` in float32 (see `gaussx.CGSolver`).
        atol: Absolute tolerance for CG. ``None``: ``1e-5`` in every dtype.
        max_steps: Maximum CG iterations.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Lanczos iterations for SLQ logdet.
        seed: Seed for probe vector generation.
        preconditioner: A prebuilt preconditioner, used for every solve.
        throw: Raise when CG does not converge within ``max_steps``. With
            ``False`` the last iterate is returned unchecked (see
            `gaussx.CGSolver`).
    """

    preconditioner_rank: int = eqx.field(static=True, default=50)
    shift: float = eqx.field(static=True, default=1.0)
    rtol: float | None = eqx.field(static=True, default=None)
    atol: float | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=1000)
    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    seed: int = eqx.field(static=True, default=0)
    preconditioner: AbstractPreconditioner | None = None
    throw: bool = eqx.field(static=True, default=True)

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
        preconditioner = self.preconditioner
        if preconditioner is None:
            preconditioner = PartialCholeskyPreconditioner(
                rank=self.preconditioner_rank, shift=self.shift
            )
        return CGSolver(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            preconditioner=preconditioner,
            throw=self.throw,
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
