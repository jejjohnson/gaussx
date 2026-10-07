"""LSMR solver strategy for least-squares and regularized systems."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matfree.lstsq
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._renamed import UNSET, default, renamed
from gaussx._strategies._slq_logdet import SLQLogdet
from gaussx._strategies._tolerances import operator_dtype, resolve_tolerance


class LSMRSolver(AbstractSolverStrategy):
    """LSMR iterative least-squares solver (Fong & Saunders 2011).

    Matrix-free solver that only requires matvec and transpose-matvec.
    Supports Tikhonov regularization via ``damp`` parameter:
    minimizes ``||Ax - b||^2 + damp^2 ||x||^2``.

    Suitable for rectangular, ill-conditioned, or regularized systems.

    The undamped path delegates to `lineax.LSMR` (with lineax's
    implicit-differentiation rules). lineax's LSMR has no Tikhonov
    ``damp`` parameter, so damped solves use matfree's LSMR, which has
    a custom VJP for memory-efficient backpropagation.

    Attributes:
        atol: Absolute tolerance. ``None``: ``1e-6`` in float64, ``1e-3``
            in float32 (gh-327).
        btol: Relative tolerance on the residual. ``None``: as ``atol``.
        ctol: Condition number tolerance (the lineax path uses
            ``conlim = 1 / ctol``).
        max_steps: Maximum iterations (formerly ``maxiter``, a deprecated
            alias; gh-405).
        damp: Tikhonov damping parameter.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Lanczos iterations for SLQ logdet.
        seed: Seed for probe vector generation.
        throw: Raise when LSMR stops without converging (``max_steps``
            reached), on both the lineax and the damped matfree path. With
            ``False`` the last iterate is returned unchecked (gh-336).
    """

    atol: float | None = eqx.field(static=True, default=None)
    btol: float | None = eqx.field(static=True, default=None)
    ctol: float = eqx.field(static=True, default=1e-6)
    max_steps: int = eqx.field(static=True, default=1000)
    damp: float = eqx.field(static=True, default=0.0)
    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    seed: int = eqx.field(static=True, default=0)
    throw: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        *,
        atol: float | None = None,
        btol: float | None = None,
        ctol: float = 1e-6,
        max_steps: int = UNSET,
        damp: float = 0.0,
        num_probes: int = 20,
        lanczos_order: int = 30,
        seed: int = 0,
        throw: bool = True,
        maxiter: int = UNSET,
    ) -> None:
        self.atol = atol
        self.btol = btol
        self.ctol = ctol
        self.max_steps = default(
            renamed("LSMRSolver", "maxiter", maxiter, "max_steps", max_steps), 1000
        )
        self.damp = damp
        self.num_probes = num_probes
        self.lanczos_order = lanczos_order
        self.seed = seed
        self.throw = throw

    @property
    def maxiter(self) -> int:
        """Deprecated: `max_steps` (gh-405)."""
        warn_deprecated("LSMRSolver.maxiter is deprecated; use .max_steps (gh-405).")
        return self.max_steps

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " m"],
    ) -> Float[Array, " n"]:
        """Solve A x = b via LSMR.

        Args:
            operator: A linear operator (may be rectangular).
            vector: The right-hand side b.

        Returns:
            The (least-squares) solution x.
        """
        dtype = operator_dtype(operator, vector)
        atol = resolve_tolerance(self.atol, dtype, 1e-6)
        btol = resolve_tolerance(self.btol, dtype, 1e-6)
        if self.damp == 0.0:
            solver = lx.LSMR(
                rtol=btol,
                atol=atol,
                max_steps=self.max_steps,
                conlim=1.0 / self.ctol if self.ctol > 0 else 1e8,
            )
            return lx.linear_solve(operator, vector, solver, throw=self.throw).value

        # Tikhonov damping: not supported by lineax's LSMR, so the
        # damped path stays on matfree.
        lsmr_fn = matfree.lstsq.lsmr(
            atol=atol,
            btol=btol,
            ctol=self.ctol,
            maxiter=self.max_steps,
        )

        def vecmat(v):
            return operator.T.mv(v)

        x, stats = lsmr_fn(vecmat, vector, damp=self.damp)
        if self.throw:
            # matfree reports success but never raises; match lineax.
            x = eqx.error_if(
                x,
                jnp.logical_not(stats["success"]),
                "LSMR did not converge within `max_steps` steps. Increase "
                "`max_steps`, loosen `atol`/`btol`, or pass `throw=False`.",
            )
        return x

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Stochastic log-determinant via Lanczos quadrature.

        LSMR itself does not use this: it is SLQ on a *square* symmetric
        PSD operator, for the `AbstractSolverStrategy` interface.

        Args:
            operator: A square symmetric PSD linear operator.

        Returns:
            Scalar estimate of log |det(A)|.

        Raises:
            ValueError: If the operator is not square (gh-402).
        """
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "LSMRSolver.logdet needs a square operator; got shape "
                f"({operator.out_size()}, {operator.in_size()})."
            )
        return SLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
            seed=self.seed,
        ).logdet(operator, key=key)
