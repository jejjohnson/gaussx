"""CG solver strategy: iterative solve + stochastic logdet via matfree."""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners import AbstractPreconditioner
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import SLQLogdet


class CGSolver(AbstractSolverStrategy):
    """Iterative CG solver with stochastic log-determinant.

    Uses lineax CG for the linear solve and matfree's stochastic
    Lanczos quadrature (SLQ) for the log-determinant. Suitable
    for large PSD operators where dense factorization is too
    expensive.

    Attributes:
        rtol: Relative tolerance for CG.
        atol: Absolute tolerance for CG.
        max_steps: Maximum CG iterations.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Order of the Lanczos decomposition for SLQ.
        preconditioner: Optional preconditioner. When set, its approximate
            inverse is passed to lineax CG to accelerate convergence.
    """

    rtol: float = eqx.field(static=True, default=1e-5)
    atol: float = eqx.field(static=True, default=1e-5)
    max_steps: int = eqx.field(static=True, default=1000)
    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    preconditioner: AbstractPreconditioner | None = None

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve ``A x = b`` with conjugate gradients.

        Args:
            operator: A PSD linear operator ``A``.
            vector: Right-hand side ``b``, shape ``(n,)``.

        Returns:
            Solution ``x``, shape ``(n,)``.
        """
        solver = lx.CG(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps)
        options: dict[str, lx.AbstractLinearOperator] = {}
        if self.preconditioner is not None:
            precond_op = self.preconditioner.as_operator(operator)
            if precond_op is not None:
                # lineax treats `options` as non-differentiable and raises if a
                # tangent reaches it, but a preconditioner built from the
                # (traced) operator carries one. The CG solution does not
                # depend on M^{-1}, only the iteration count does, so stopping
                # its gradient is exact (as `_inv_quad_logdet` does).
                dynamic, static = eqx.partition(precond_op, eqx.is_array)
                options["preconditioner"] = eqx.combine(
                    jax.lax.stop_gradient(dynamic), static
                )
        return lx.linear_solve(operator, vector, solver, options=options).value

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
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
