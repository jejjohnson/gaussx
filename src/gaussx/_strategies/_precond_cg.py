"""Preconditioned CG solver with pivoted partial Cholesky preconditioner."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.decomp
import matfree.funm
import matfree.low_rank
import matfree.stochtrace

from gaussx._strategies._base import AbstractSolverStrategy


class PreconditionedCGSolver(AbstractSolverStrategy):
    """CG solver with pivoted partial Cholesky preconditioner.

    Uses matfree's ``low_rank.cholesky_partial_pivot`` to build a
    rank-k preconditioner, then solves ``(sI + LL^T)^{-1} v`` via
    the Woodbury identity inside lineax CG.

    For operators of the form ``K + sigma^2 I``, preconditioning
    dramatically reduces the number of CG iterations.

    Args:
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
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve A x = b via preconditioned CG.

        Args:
            operator: A PSD linear operator.
            vector: The right-hand side b.

        Returns:
            The solution x.
        """
        # Build preconditioner from element access
        if self.preconditioner_rank > 0:
            n = operator.in_size()
            rank = min(self.preconditioner_rank, n)

            def mat_el(i, j):
                ei = jnp.zeros(n).at[i].set(1.0)
                ej = jnp.zeros(n).at[j].set(1.0)
                return ei @ operator.mv(ej)

            chol_fn = matfree.low_rank.cholesky_partial_pivot(
                mat_el, nrows=n, rank=rank
            )
            L, _info = chol_fn()

            precond_fn = matfree.low_rank.preconditioner(lambda: (L, _info))

            # Preconditioned solve: M^{-1} A x = M^{-1} b
            # where M = sI + LL^T
            def precond_matvec(v):
                Mv_inv, _ = precond_fn(v, self.shift)
                return Mv_inv

            # Apply preconditioner then CG on the preconditioned system
            # For simplicity, use lineax CG with the preconditioner as a
            # left-preconditioned solve
            b_precond = precond_matvec(vector)

            def precond_op_mv(v):
                return precond_matvec(operator.mv(v))

            precond_op = lx.FunctionLinearOperator(
                precond_op_mv, operator.out_structure()
            )
            solver = lx.CG(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps)
            return lx.linear_solve(precond_op, b_precond, solver).value

        # Fallback: plain CG
        solver = lx.CG(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps)
        return lx.linear_solve(operator, vector, solver).value

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
    ) -> jnp.ndarray:
        """Stochastic log-determinant via Lanczos quadrature.

        Args:
            operator: A PSD linear operator.

        Returns:
            Scalar estimate of log |det(A)|.
        """
        key = jax.random.PRNGKey(self.seed)
        n = operator.in_size()

        order = min(self.lanczos_order, n)
        tridiag = matfree.decomp.tridiag_sym(order, reortho="full")
        integrand = matfree.funm.integrand_funm_sym_logdet(tridiag)

        sample_shape = jnp.zeros(n)
        sampler = matfree.stochtrace.sampler_rademacher(
            sample_shape, num=self.num_probes
        )
        estimator = matfree.stochtrace.estimator(integrand, sampler)
        return estimator(operator.mv, key)
