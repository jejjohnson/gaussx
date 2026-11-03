"""Conjugate-gradient solver strategy: a CG solve and an SLQ log-determinant.

A is a square N × N symmetric positive-definite operator, b the right-hand
side and x the solution of A x = b. From x₀ = 0, CG builds iterates x_j,
residuals r_j = b − A x_j and search directions d_j, j = 0, 1, …, with step
sizes α_j and direction updates β_j; a preconditioner applies M⁻¹ ≈ A⁻¹ to
the residual, z_j = M⁻¹ r_j. The log-determinant is the stochastic Lanczos
quadrature (SLQ) estimate of `gaussx.SLQLogdet` over probes z^(i),
i = 1, …, N_probe.
"""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners import AbstractPreconditioner
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import SLQLogdet
from gaussx._strategies._tolerances import (
    operator_dtype,
    resolve_tolerance,
    rhs_scaling,
)


class CGSolver(AbstractSolverStrategy):
    """Conjugate-gradient solve with a stochastic log-determinant.

        α_j = r_jᵀ z_j / d_jᵀ A d_j,   x_j+1 = x_j + α_j d_j,
        r_j+1 = r_j − α_j A d_j,       β_j = r_j+1ᵀ z_j+1 / r_jᵀ z_j,
        d_j+1 = z_j+1 + β_j d_j,       z_j = M⁻¹ r_j

    stopping once every entry satisfies |r_j,n| ≤ tol_abs + tol_rel |b_n|.
    lineax CG runs the solve and matfree's stochastic Lanczos quadrature
    (SLQ) the log-determinant, for large PSD operators where a dense
    factorisation is too expensive.

    The relative tolerance tol_rel defaults to ``None``, which resolves from
    the operator's dtype at solve time: 1e-5 in float64 and 1e-3 in float32,
    where a relative residual of 1e-5 is out of reach once the condition
    number passes about 1e3 (gh-327). Set the tolerances explicitly to
    override.

    Attributes:
        rtol: Relative tolerance tol_rel. ``None``: 1e-5 in float64, 1e-3 in
            float32.
        atol: Absolute tolerance tol_abs. ``None``: 1e-5 in float64; in
            float32, √eps maxₙ |b_n|, relative to the right-hand side (the
            solve runs on b / maxₙ |b_n|; gh-639).
        max_steps: Maximum number of CG steps j.
        num_probes: Number of SLQ probe vectors N_probe.
        lanczos_order: Order of the Lanczos decomposition for SLQ.
        seed: Seed of the SLQ probes when `logdet` gets no ``key``. With no
            key every call reuses the same probes (common random numbers);
            see `gaussx.KeyedSolver`.
        preconditioner: Optional preconditioner. When set, its approximate
            inverse M⁻¹ ≈ A⁻¹ is passed to lineax CG to accelerate
            convergence.
        throw: Raise when CG does not converge within ``max_steps`` (the
            default). With ``False`` the last iterate is returned unchecked;
            an unconverged CG iterate can be far worse than zero, so only
            use it where the caller checks the result.

    Examples:
        A PSD system solved to the default tolerance:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> b = jnp.array([1.0, 2.0])
        >>> x = gaussx.CGSolver().solve(A, b)
        >>> bool(jnp.allclose(matrix @ x, b, atol=1e-3))
        True

        A Jacobi preconditioner gives the same solution:

        >>> solver = gaussx.CGSolver(preconditioner=gaussx.JacobiPreconditioner())
        >>> bool(jnp.allclose(solver.solve(A, b), x, atol=1e-3))
        True

        Running out of steps raises rather than returning a poor iterate:

        >>> tight = gaussx.CGSolver(max_steps=1, rtol=1e-12, atol=1e-12)
        >>> tight.solve(A, b)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        equinox.EquinoxRuntimeError: The maximum number of solver steps was reached.
    """

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
        """Solve A x = b with (preconditioned) conjugate gradients.

            x = A⁻¹ b

        Algorithm:
            1. Resolve tol_rel and tol_abs for the dtype; below float64 with
               the default ``atol``, rescale to A y = b / s with
               s = maxₙ |b_n|.
            2. Build M⁻¹ from ``preconditioner``, with its gradient stopped.
            3. Run lineax CG to |r_j,n| ≤ tol_abs + tol_rel |b_n|, raising
               at ``max_steps`` when ``throw`` is set.
            4. Undo the rescaling, x = s y.

        Args:
            operator: Symmetric positive-definite operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.

        Raises:
            equinox.EquinoxRuntimeError: If CG has not converged after
                ``max_steps`` steps and ``throw`` is set.
        """
        dtype = operator_dtype(operator, vector)
        # tol_abs and s = maxₙ |b_n| (s is None: no rescaling)
        atol, scale = rhs_scaling(self.atol, dtype, 1e-5, vector)
        # b / s
        rhs = vector if scale is None else vector / scale
        rtol = resolve_tolerance(self.rtol, dtype, 1e-5)
        solver = lx.CG(
            rtol=rtol,
            atol=atol,
            max_steps=self.max_steps,
        )
        options: dict[str, lx.AbstractLinearOperator] = {}
        if self.preconditioner is not None:
            precond_op = self.preconditioner.as_operator(operator)
            if precond_op is not None:
                # lineax treats `options` as non-differentiable and raises if a
                # tangent reaches it, but a preconditioner built from the
                # (traced) operator carries one. The CG solution does not
                # depend on M⁻¹, only the iteration count does, so stopping
                # its gradient is exact (as `_inv_quad_logdet` does).
                dynamic, static = eqx.partition(precond_op, eqx.is_array)
                options["preconditioner"] = eqx.combine(
                    jax.lax.stop_gradient(dynamic), static
                )
        # y = A⁻¹ (b / s)
        solution = lx.linear_solve(
            operator, rhs, solver, options=options, throw=self.throw
        ).value
        # x = s y
        return solution if scale is None else solution * scale

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Estimate log det A by stochastic Lanczos quadrature.

            log det A ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁

        with T^(i) the Lanczos tridiagonal of A started at z^(i).

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
