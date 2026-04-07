"""CG solver strategy: iterative solve + stochastic logdet via matfree."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import SLQLogdet


class CGSolver(AbstractSolverStrategy):
    """Iterative CG solver with stochastic log-determinant.

    Uses lineax CG for the linear solve and matfree's stochastic
    Lanczos quadrature (SLQ) for the log-determinant. Suitable
    for large PSD operators where dense factorization is too
    expensive.

    Args:
        rtol: Relative tolerance for CG.
        atol: Absolute tolerance for CG.
        max_steps: Maximum CG iterations.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Order of the Lanczos decomposition for SLQ.
    """

    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 1000
    num_probes: int = 20
    lanczos_order: int = 30

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        solver = lx.CG(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps)
        return lx.linear_solve(operator, vector, solver).value

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """Stochastic log-determinant via Lanczos quadrature.

        Args:
            operator: A PSD linear operator.
            key: PRNG key for probe vector sampling. If None,
                uses ``jax.random.PRNGKey(0)``.

        Returns:
            Scalar estimate of log |det(A)|.
        """
        return SLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
        ).logdet(operator, key=key)
