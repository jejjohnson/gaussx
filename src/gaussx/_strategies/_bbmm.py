"""BBMM solver strategy: batched CG + stochastic logdet (Gardner et al. 2018)."""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import SLQLogdet
from gaussx._strategies._tolerances import operator_dtype, resolve_tolerance


class BBMMSolver(AbstractSolverStrategy):
    """Black-Box Matrix-Matrix solver (Gardner et al. 2018).

    Simultaneously solves multiple RHS and computes logdet via
    modified batched CG (mBCG). Amortizes matvecs across solve
    and logdet.

    Solve: CG via lineax on each RHS column.
    Logdet: Stochastic Lanczos Quadrature via matfree.

    Probe vectors are generated at construction time from ``seed``
    and stored as frozen state. This makes ``logdet`` and
    ``solve_and_logdet`` deterministic functions of the operator —
    no PRNG key is needed at call time.

    Attributes:
        cg_max_iter: Maximum CG iterations.
        cg_tolerance: Relative and absolute tolerance for CG. ``None``: a
            relative ``1e-4`` in float64 and ``1e-3`` in float32 (gh-327),
            and an absolute ``1e-4`` in every dtype.
        lanczos_iter: Lanczos iterations for SLQ.
        num_probes: Number of probe vectors for Hutchinson.
        seed: Seed for probe vector generation.
        throw: Raise when CG does not converge within ``cg_max_iter``. With
            ``False`` the last iterate is returned unchecked (see
            `gaussx.CGSolver`).
    """

    cg_max_iter: int = eqx.field(static=True, default=1000)
    cg_tolerance: float | None = eqx.field(static=True, default=None)
    lanczos_iter: int = eqx.field(static=True, default=100)
    num_probes: int = eqx.field(static=True, default=10)
    seed: int = eqx.field(static=True, default=0)
    throw: bool = eqx.field(static=True, default=True)

    def _cg_tolerance(self, dtype) -> float:
        """``cg_tolerance``, or its default for *dtype*."""
        return resolve_tolerance(self.cg_tolerance, dtype, 1e-4)

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b via CG.

        Args:
            operator: A PSD linear operator.
            vector: The right-hand side b.

        Returns:
            The solution x.
        """
        rtol = self._cg_tolerance(operator_dtype(operator, vector))
        atol = 1e-4 if self.cg_tolerance is None else self.cg_tolerance
        solver = lx.CG(rtol=rtol, atol=atol, max_steps=self.cg_max_iter)
        return lx.linear_solve(operator, vector, solver, throw=self.throw).value

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Stochastic log-determinant via Lanczos quadrature.

        Probe vectors are generated deterministically from ``self.seed``.

        Args:
            operator: A PSD linear operator.

        Returns:
            Scalar estimate of log |det(A)|.
        """
        return SLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_iter,
            seed=self.seed,
        ).logdet(operator, key=key)

    def solve_and_logdet(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> tuple[Float[Array, " n"], Float[Array, ""]]:
        """Joint solve + logdet.

        Computes both solve(A, b) and logdet(A) sharing the
        operator's matvec calls where possible.

        Args:
            operator: A PSD linear operator.
            vector: The right-hand side b.

        Returns:
            Tuple of (solution, log_determinant).
        """
        return self.solve(operator, vector), self.logdet(operator)
