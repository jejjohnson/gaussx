"""BBMM solver strategy: batched CG + stochastic logdet (Gardner et al. 2018)."""

from __future__ import annotations

import equinox as eqx
import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._renamed import UNSET, default, renamed
from gaussx._strategies._slq_logdet import SLQLogdet
from gaussx._strategies._tolerances import operator_dtype, resolve_tolerance


class BBMMSolver(AbstractSolverStrategy):
    """CG solve and SLQ logdet with BBMM's defaults (Gardner et al. 2018).

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
        rtol: Relative tolerance for CG. ``None``: ``1e-4`` in float64,
            ``1e-3`` in float32 (gh-327).
        atol: Absolute tolerance for CG. ``None``: ``1e-4`` in every dtype.
        max_steps: Maximum CG iterations.
        lanczos_order: Lanczos iterations for SLQ.
        num_probes: Number of probe vectors for Hutchinson.
        seed: Seed for probe vector generation.
        throw: Raise when CG does not converge within ``max_steps``. With
            ``False`` the last iterate is returned unchecked (see
            `gaussx.CGSolver`).
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
        """Deprecated: `rtol` (gh-405)."""
        warn_deprecated("BBMMSolver.cg_tolerance is deprecated; use .rtol (gh-405).")
        return self.rtol

    @property
    def cg_max_iter(self) -> int:
        """Deprecated: `max_steps` (gh-405)."""
        warn_deprecated(
            "BBMMSolver.cg_max_iter is deprecated; use .max_steps (gh-405)."
        )
        return self.max_steps

    @property
    def lanczos_iter(self) -> int:
        """Deprecated: `lanczos_order` (gh-405)."""
        warn_deprecated(
            "BBMMSolver.lanczos_iter is deprecated; use .lanczos_order (gh-405)."
        )
        return self.lanczos_order

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
        dtype = operator_dtype(operator, vector)
        solver = lx.CG(
            rtol=resolve_tolerance(self.rtol, dtype, 1e-4),
            atol=1e-4 if self.atol is None else self.atol,
            max_steps=self.max_steps,
        )
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
            lanczos_order=self.lanczos_order,
            seed=self.seed,
        ).logdet(operator, key=key)

    def solve_and_logdet(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> tuple[Float[Array, " n"], Float[Array, ""]]:
        """``(solve(A, b), logdet(A))`` in one call.

        A convenience wrapper: it costs exactly `solve` plus `logdet`, with
        no shared matvecs. For the shared mBCG pass use
        ``gaussx.inv_quad_logdet(operator, rhs, strategy=self)``.

        Args:
            operator: A PSD linear operator.
            vector: The right-hand side b.

        Returns:
            Tuple of (solution, log_determinant).
        """
        return self.solve(operator, vector), self.logdet(operator)
