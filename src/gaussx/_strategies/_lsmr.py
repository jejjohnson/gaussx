"""LSMR solver strategy for least-squares and regularized systems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.lstsq

from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import SLQLogdet


class LSMRSolver(AbstractSolverStrategy):
    """LSMR iterative least-squares solver (Fong & Saunders 2011).

    Matrix-free solver that only requires matvec and transpose-matvec.
    Supports Tikhonov regularization via ``damp`` parameter:
    minimizes ``||Ax - b||^2 + damp^2 ||x||^2``.

    Suitable for rectangular, ill-conditioned, or regularized systems.
    Has a custom VJP for memory-efficient backpropagation.

    Args:
        atol: Absolute tolerance.
        btol: Relative tolerance on the residual.
        ctol: Condition number tolerance.
        maxiter: Maximum iterations.
        damp: Tikhonov damping parameter.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Lanczos iterations for SLQ logdet.
        seed: Seed for probe vector generation.
    """

    atol: float = 1e-6
    btol: float = 1e-6
    ctol: float = 1e-6
    maxiter: int = 1000
    damp: float = 0.0
    num_probes: int = 20
    lanczos_order: int = 30
    seed: int = 0

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve A x = b via LSMR.

        Args:
            operator: A linear operator (may be rectangular).
            vector: The right-hand side b.

        Returns:
            The (least-squares) solution x.
        """
        lsmr_fn = matfree.lstsq.lsmr(
            atol=self.atol,
            btol=self.btol,
            ctol=self.ctol,
            maxiter=self.maxiter,
        )

        def vecmat(v):
            return operator.T.mv(v)

        result = lsmr_fn(vecmat, vector, damp=self.damp)
        return result[0]

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
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
