"""Preconditioned CG strategy with a pivoted partial Cholesky preconditioner.

A = K + σ² I is an N × N symmetric positive-definite system with a PSD part K
and a noise variance σ². A rank-k pivoted partial Cholesky factor F of K
(N × k) gives the preconditioner M = σ² I + F Fᵀ ≈ A, applied as M⁻¹ by the
Woodbury identity inside conjugate gradients (CG). The log-determinant is the
stochastic Lanczos quadrature (SLQ) estimate over probes z^(i),
i = 1, …, N_probe.
"""

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
    """CG with a pivoted partial Cholesky preconditioner.

        M = σ² I + F Fᵀ ≈ A = K + σ² I,
        M⁻¹ v = σ⁻² (v − F (σ² I_k + Fᵀ F)⁻¹ Fᵀ v)

    `gaussx.PartialCholeskyPreconditioner`'s guarded pivoted partial
    Cholesky builds the rank-k factor F of K, and lineax CG applies M⁻¹
    by the Woodbury identity. For A = K + σ² I this cuts the number of CG
    steps sharply.

    Pass a ``preconditioner`` built once with
    `gaussx.PartialCholeskyPreconditioner.from_operator` or
    `gaussx.NystromPreconditioner.from_operator` (on K, with
    ``shift=σ²``) to reuse it across solves. Otherwise a rank
    ``preconditioner_rank`` factor of A − σ² I is rebuilt from the system
    operator A at every solve, so the noise is never counted twice (#345).

    Attributes:
        preconditioner_rank: Rank k of the partial Cholesky built per solve.
            Set to 0 to disable preconditioning (plain CG). Ignored when
            ``preconditioner`` is given.
        shift: The noise variance σ² in the system A = K + σ² I, for the
            preconditioner built per solve. Must not exceed the noise
            actually in A. Ignored when ``preconditioner`` is given.
        rtol: Relative tolerance tol_rel. ``None``: 1e-5 in float64, 1e-3
            in float32 (see `gaussx.CGSolver`).
        atol: Absolute tolerance tol_abs. ``None``: 1e-5 in float64; in
            float32, √eps maxₙ |b_n|, relative to the right-hand side (the
            solve runs on b / maxₙ |b_n|; gh-639).
        max_steps: Maximum number of CG steps j.
        num_probes: Number of SLQ probe vectors N_probe.
        lanczos_order: Number of Lanczos steps for the SLQ log-determinant.
        seed: Seed of the SLQ probes when `logdet` gets no ``key``.
        preconditioner: A prebuilt preconditioner, used for every solve.
        throw: Raise when CG does not converge within ``max_steps``. With
            ``False`` the last iterate is returned unchecked (see
            `gaussx.CGSolver`).

    Examples:
        A noisy RBF Gram matrix, with the preconditioner rebuilt per solve:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> inputs = jnp.linspace(0.0, 1.0, 20)
        >>> gaps = einx.subtract("row, col -> row col", inputs, inputs)
        >>> matrix = jnp.exp(-0.5 * gaps**2 / 0.1**2) + 0.1 * jnp.eye(20)
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> solver = gaussx.PreconditionedCGSolver(preconditioner_rank=5, shift=0.1)
        >>> x = solver.solve(A, jnp.ones(20))
        >>> bool(jnp.allclose(matrix @ x, jnp.ones(20), atol=1e-2))
        True
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
        """Solve A x = b by CG preconditioned with M⁻¹ ≈ A⁻¹.

            x = A⁻¹ b

        Algorithm:
            1. Take ``preconditioner``, or a rank-k partial Cholesky of
               A − σ² I with shift σ².
            2. Run `gaussx.CGSolver` with M⁻¹ and the same tolerances.

        Args:
            operator: Symmetric positive-definite operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.

        Raises:
            equinox.EquinoxRuntimeError: If CG has not converged after
                ``max_steps`` steps and ``throw`` is set.
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
        """Estimate log det A by stochastic Lanczos quadrature.

            log det A ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁

        with T^(i) the Lanczos tridiagonal of A started at z^(i). The
        preconditioner is not used here.

        Args:
            operator: Symmetric positive-definite operator A, N × N.
            key: PRNG key for the probes z^(i). ``None`` uses
                ``jax.random.PRNGKey(seed)``: the same probes on every call.

        Returns:
            Scalar estimate of log det A.
        """
        return SLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
            seed=self.seed,
        ).logdet(operator, key=key)
