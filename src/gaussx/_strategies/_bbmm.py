"""BBMM solver strategy: batched CG + stochastic logdet (Gardner et al. 2018)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import SLQLogdet


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

    Args:
        cg_max_iter: Maximum CG iterations.
        cg_tolerance: Relative tolerance for CG.
        lanczos_iter: Lanczos iterations for SLQ.
        num_probes: Number of probe vectors for Hutchinson.
        jitter: Diagonal jitter for numerical stability.
        seed: Seed for probe vector generation.
    """

    cg_max_iter: int = 1000
    cg_tolerance: float = 1e-4
    lanczos_iter: int = 100
    num_probes: int = 10
    jitter: float = 1e-6
    seed: int = 0

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve A x = b via CG.

        Args:
            operator: A PSD linear operator.
            vector: The right-hand side b.

        Returns:
            The solution x.
        """
        solver = lx.CG(
            rtol=self.cg_tolerance,
            atol=self.cg_tolerance,
            max_steps=self.cg_max_iter,
        )
        return lx.linear_solve(operator, vector, solver).value

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
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
        vector: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
