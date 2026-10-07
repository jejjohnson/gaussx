"""CG solver strategy: iterative solve + stochastic logdet via matfree."""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners import AbstractPreconditioner
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import SLQLogdet
from gaussx._strategies._tolerances import operator_dtype, resolve_tolerance


class CGSolver(AbstractSolverStrategy):
    """Iterative CG solver with stochastic log-determinant.

    Uses lineax CG for the linear solve and matfree's stochastic
    Lanczos quadrature (SLQ) for the log-determinant. Suitable
    for large PSD operators where dense factorization is too
    expensive.

    The relative tolerance defaults to ``None``, which resolves from the
    operator's dtype at solve time: ``1e-5`` in float64 and ``1e-3`` in
    float32, where
    a relative residual of ``1e-5`` is out of reach once the condition number
    passes about ``1e3`` (gh-327). Set them explicitly to override.

    Attributes:
        rtol: Relative tolerance for CG. ``None``: ``1e-5`` in float64,
            ``1e-3`` in float32.
        atol: Absolute tolerance for CG. ``None``: ``1e-5`` in every dtype
            (only the relative tolerance is relaxed for float32).
        max_steps: Maximum CG iterations.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Order of the Lanczos decomposition for SLQ.
        preconditioner: Optional preconditioner. When set, its approximate
            inverse is passed to lineax CG to accelerate convergence.
        throw: Raise when CG does not converge within ``max_steps`` (the
            default). With ``False`` the last iterate is returned unchecked;
            an unconverged CG iterate can be far worse than zero, so only
            use it where the caller checks the result.
    """

    rtol: float | None = eqx.field(static=True, default=None)
    atol: float | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=1000)
    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    preconditioner: AbstractPreconditioner | None = None
    throw: bool = eqx.field(static=True, default=True)

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
        dtype = operator_dtype(operator, vector)
        solver = lx.CG(
            rtol=resolve_tolerance(self.rtol, dtype, 1e-5),
            atol=1e-5 if self.atol is None else self.atol,
            max_steps=self.max_steps,
        )
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
        return lx.linear_solve(
            operator, vector, solver, options=options, throw=self.throw
        ).value

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
