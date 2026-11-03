"""BBMM solver strategy: a CG solve and an SLQ logdet (Gardner et al. 2018).

A is a square N × N symmetric positive-definite operator, b the right-hand
side and x the solution of A x = b, computed by conjugate gradients (CG)
with residuals r_j = b − A x_j. The log-determinant is the stochastic
Lanczos quadrature (SLQ) estimate over probes z^(i), i = 1, …, N_probe, with
T^(i) the Lanczos tridiagonal of A started at z^(i). The shared modified
batched CG (mBCG) pass lives in `gaussx.inv_quad_logdet`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._renamed import UNSET, default, renamed
from gaussx._strategies._slq_logdet import SLQLogdet
from gaussx._strategies._tolerances import (
    operator_dtype,
    resolve_tolerance,
    rhs_scaling,
)


class BBMMSolver(AbstractSolverStrategy):
    """CG solve and SLQ logdet with BBMM's defaults (Gardner et al. 2018).

        x = A⁻¹ b   by CG,
        log det A ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁

    Solve: an ordinary lineax CG solve of each right-hand side.
    Logdet: stochastic Lanczos quadrature (SLQ) via matfree, with
    ``lanczos_order`` Lanczos steps on each of ``num_probes`` probes.

    The two are independent: `solve_and_logdet` is a convenience wrapper
    that calls `solve` and then `logdet`, sharing no matvecs (gh-396). The
    modified batched CG (mBCG) pass that *does* share them, running the
    right-hand sides and the probes through one block CG, is
    ``gaussx.inv_quad_logdet(operator, rhs, strategy=BBMMSolver(...))``. It
    is not automatically cheaper: it runs the probe columns to convergence
    rather than stopping at ``lanczos_order`` Lanczos steps, so it can apply
    the operator to more columns than a separate solve and SLQ do.

    Only the integer ``seed`` is stored. With no ``key``, `logdet` draws
    its probes from ``PRNGKey(seed)`` at every call, so it is a
    deterministic function of the operator: the same probes every time
    (common random numbers). Pass ``key`` or use `gaussx.KeyedSolver` to
    vary them.

    The options use the names shared by every iterative strategy (gh-405).
    The old keywords still work, with a deprecation warning:
    ``cg_tolerance=t`` sets ``rtol=atol=t``, ``cg_max_iter`` is
    ``max_steps`` and ``lanczos_iter`` is ``lanczos_order``.

    Attributes:
        rtol: Relative tolerance tol_rel. ``None``: 1e-4 in float64, 1e-3
            in float32 (gh-327).
        atol: Absolute tolerance tol_abs. ``None``: 1e-4 in float64; in
            float32, √eps maxₙ |b_n|, relative to the right-hand side (the
            solve runs on b / maxₙ |b_n|; gh-639).
        max_steps: Maximum number of CG steps j.
        lanczos_order: Number of Lanczos steps, the order of T^(i).
        num_probes: Number of probe vectors N_probe.
        seed: Seed of the probes z^(i) when `logdet` gets no ``key``.
        throw: Raise when CG does not converge within ``max_steps``. With
            ``False`` the last iterate is returned unchecked (see
            `gaussx.CGSolver`).

    Examples:
        A solve, and an SLQ estimate that is exact on a diagonal:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> b = jnp.array([1.0, 2.0])
        >>> solver = gaussx.BBMMSolver(lanczos_order=2, num_probes=4)
        >>> bool(jnp.allclose(matrix @ solver.solve(A, b), b, atol=1e-3))
        True
        >>> diagonal = lx.MatrixLinearOperator(
        ...     jnp.diag(jnp.array([1.0, 2.0])), lx.positive_semidefinite_tag
        ... )
        >>> round(float(solver.logdet(diagonal)), 3)
        0.693

        The deprecated keywords still work, with a warning:

        >>> import warnings
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     old = gaussx.BBMMSolver(cg_max_iter=50)
        >>> old.max_steps
        50

    References:
        Gardner, J. R., Pleiss, G., Weinberger, K. Q., Bindel, D., &
        Wilson, A. G. (2018). GPyTorch: Blackbox matrix-matrix Gaussian
        process inference with GPU acceleration. *Proc. NeurIPS*.
    """

    rtol: float | None = eqx.field(static=True, default=None)
    atol: float | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=1000)
    lanczos_order: int = eqx.field(static=True, default=100)
    num_probes: int = eqx.field(static=True, default=10)
    seed: int = eqx.field(static=True, default=0)
    throw: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        *,
        rtol: float | None = UNSET,
        atol: float | None = UNSET,
        max_steps: int = UNSET,
        lanczos_order: int = UNSET,
        num_probes: int = 10,
        seed: int = 0,
        throw: bool = True,
        cg_tolerance: float | None = UNSET,
        cg_max_iter: int = UNSET,
        lanczos_iter: int = UNSET,
    ) -> None:
        name = "BBMMSolver"
        rtol = renamed(name, "cg_tolerance", cg_tolerance, "rtol", rtol)
        if cg_tolerance is not UNSET:
            # cg_tolerance set both tolerances.
            if atol is not UNSET:
                raise TypeError(f"{name}: pass rtol= and atol=, not cg_tolerance=.")
            atol = cg_tolerance
        self.rtol = default(rtol, None)
        self.atol = default(atol, None)
        self.max_steps = default(
            renamed(name, "cg_max_iter", cg_max_iter, "max_steps", max_steps), 1000
        )
        self.lanczos_order = default(
            renamed(name, "lanczos_iter", lanczos_iter, "lanczos_order", lanczos_order),
            100,
        )
        self.num_probes = num_probes
        self.seed = seed
        self.throw = throw

    @property
    def cg_tolerance(self) -> float | None:
        """Deprecated: `rtol` (gh-405).

        Returns:
            ``rtol``, with a deprecation warning.
        """
        warn_deprecated(
            "BBMMSolver.cg_tolerance is deprecated and will be removed in gaussx "
            "0.7.0; use .rtol (gh-405)."
        )
        return self.rtol

    @property
    def cg_max_iter(self) -> int:
        """Deprecated: `max_steps` (gh-405).

        Returns:
            ``max_steps``, with a deprecation warning.
        """
        warn_deprecated(
            "BBMMSolver.cg_max_iter is deprecated and will be removed in gaussx "
            "0.7.0; use .max_steps (gh-405)."
        )
        return self.max_steps

    @property
    def lanczos_iter(self) -> int:
        """Deprecated: `lanczos_order` (gh-405).

        Returns:
            ``lanczos_order``, with a deprecation warning.
        """
        warn_deprecated(
            "BBMMSolver.lanczos_iter is deprecated and will be removed in gaussx "
            "0.7.0; use .lanczos_order (gh-405)."
        )
        return self.lanczos_order

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b by conjugate gradients.

            x = A⁻¹ b

        Algorithm:
            1. Resolve tol_rel and tol_abs for the dtype; below float64 with
               the default ``atol``, rescale to A y = b / s with
               s = maxₙ |b_n|.
            2. Run lineax CG to |r_j,n| ≤ tol_abs + tol_rel |b_n|, raising
               at ``max_steps`` when ``throw`` is set.
            3. Undo the rescaling, x = s y.

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
        atol, scale = rhs_scaling(self.atol, dtype, 1e-4, vector)
        # b / s
        rhs = vector if scale is None else vector / scale
        rtol = resolve_tolerance(self.rtol, dtype, 1e-4)
        solver = lx.CG(
            rtol=rtol,
            atol=atol,
            max_steps=self.max_steps,
        )
        # y = A⁻¹ (b / s)
        solution = lx.linear_solve(operator, rhs, solver, throw=self.throw).value
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

        With no ``key`` the probes z^(i) come from ``PRNGKey(seed)``, the
        same on every call.

        Args:
            operator: Symmetric positive-definite operator A, N × N.
            key: PRNG key for the probes z^(i). ``None`` uses
                ``jax.random.PRNGKey(seed)``.

        Returns:
            Scalar estimate of log det A.
        """
        return SLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
            seed=self.seed,
        ).logdet(operator, key=key)

    def solve_and_logdet(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> tuple[Float[Array, " n"], Float[Array, ""]]:
        """Solve A x = b and estimate log det A in one call.

            (x, log det A) = (A⁻¹ b, SLQ estimate)

        A convenience wrapper: it costs exactly `solve` plus `logdet`, with
        no shared matvecs. For the shared mBCG pass use
        ``gaussx.inv_quad_logdet(operator, rhs, strategy=self)``.

        Args:
            operator: Symmetric positive-definite operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Tuple ``(solution, logdet)`` holding x and the estimate of
            log det A.

        Raises:
            equinox.EquinoxRuntimeError: If CG has not converged after
                ``max_steps`` steps and ``throw`` is set.
        """
        return self.solve(operator, vector), self.logdet(operator)
