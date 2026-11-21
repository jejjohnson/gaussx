"""Abstract strategy protocols: a solve x = A⁻¹ b, a log-determinant log |det A|.

A is a square N × N operator, b ∈ ℝ^N the right-hand side and x ∈ ℝ^N the
solution. A strategy changes how these are computed (a factorisation, a
Krylov iteration, a stochastic estimate), never what they are.
"""

from __future__ import annotations

import abc
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, Int

from gaussx._strategies._tolerances import (
    operator_dtype,
    resolve_tolerance,
    rhs_scaling,
)


class _SolveStats(NamedTuple):
    """A solve's value with how it ended, for `gaussx.linear_solve_with_info`.

    Attributes:
        value: The solution x.
        num_steps: Iterations taken; 0 for a direct solve.
        success: The iteration's own verdict (always ``True`` for a direct
            solve).
        rtol: The relative tolerance the solve aimed at.
        atol: The absolute tolerance it aimed at, in the units of b.
    """

    value: Float[Array, " n"]
    num_steps: Int[Array, ""]
    success: Bool[Array, ""]
    rtol: float
    atol: float | Float[Array, ""]


def _default_tolerances(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
) -> tuple[float, float | Float[Array, ""]]:
    """The default tolerances for A and b: 1e-5 in float64, per dtype below.

    Args:
        operator: Linear operator A.
        vector: Right-hand side b.

    Returns:
        ``(rtol, atol)``, with ``atol`` in the units of b.
    """
    dtype = operator_dtype(operator, vector)
    atol, scale = rhs_scaling(None, dtype, 1e-5, vector)
    rtol = resolve_tolerance(None, dtype, 1e-5)
    return rtol, atol if scale is None else atol * scale


def _lineax_stats(
    solution: lx.Solution,
    value: Float[Array, " n"],
    rtol: float,
    atol: float | Float[Array, ""],
    scale: Float[Array, ""] | None,
) -> _SolveStats:
    """Read the step count and verdict of a lineax solve.

    Args:
        solution: The lineax solution.
        value: The solution x, after any rescaling is undone.
        rtol: The relative tolerance passed to lineax.
        atol: The absolute tolerance passed to lineax, for b / s.
        scale: The right-hand-side scale s, or ``None``.

    Returns:
        The stats, with ``atol`` in the units of b.
    """
    return _SolveStats(
        value,
        jnp.asarray(solution.stats.get("num_steps", 0)),
        solution.result == lx.RESULTS.successful,
        rtol,
        atol if scale is None else atol * scale,
    )


class AbstractSolveStrategy(eqx.Module):
    """Protocol for a linear-solve strategy.

        x = A⁻¹ b

    A solve strategy fixes the algorithm for A x = b. Keeping it apart from
    the log-determinant lets `ComposedSolver` pair any solve with any
    log-determinant.

    Examples:
        Every full strategy is also a solve strategy:

        >>> import gaussx
        >>> isinstance(gaussx.CGSolver(), gaussx.AbstractSolveStrategy)
        True
    """

    @abc.abstractmethod
    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b.

            x = A⁻¹ b

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.
        """
        ...

    def _solve_with_stats(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> _SolveStats:
        """`solve`, with its step count, verdict and tolerances.

        The default is a direct solve: no steps, success, and the default
        tolerances (1e-5 in float64). Iterative strategies override it.

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            The solution and how the solve ended.
        """
        rtol, atol = _default_tolerances(operator, vector)
        return _SolveStats(
            self.solve(operator, vector),
            jnp.zeros((), jnp.int32),
            jnp.ones((), bool),
            rtol,
            atol,
        )


class AbstractLogdetStrategy(eqx.Module):
    """Protocol for a log-determinant strategy.

        log |det A|

    A logdet strategy fixes the algorithm for log |det A|. Keeping it apart
    from the solve lets `ComposedSolver` pair any log-determinant with any
    solve. Every implementation takes an optional ``key`` for its stochastic
    estimators; deterministic strategies ignore it.

    Examples:
        The stochastic Lanczos estimator is a logdet-only strategy:

        >>> import gaussx
        >>> isinstance(gaussx.SLQLogdet(), gaussx.AbstractLogdetStrategy)
        True
        >>> isinstance(gaussx.SLQLogdet(), gaussx.AbstractSolveStrategy)
        False
    """

    @abc.abstractmethod
    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Compute log |det A|.

            log |det A| = ∑_d log |γ_d|

        with γ_d, d = 1, …, N, the eigenvalues of A.

        Args:
            operator: Linear operator A, N × N.
            key: Optional PRNG key for stochastic estimators.

        Returns:
            Scalar log |det A|, exact or estimated.
        """
        ...


class AbstractSolverStrategy(AbstractSolveStrategy, AbstractLogdetStrategy):
    """Protocol for a solver strategy that pairs a solve and a log-determinant.

        x = A⁻¹ b,   log |det A|

    A solver strategy fixes both algorithms, which decouples distributions
    and recipes from how their linear algebra is computed. Subclasses
    implement both `solve` and `logdet`; for independent control use
    `AbstractSolveStrategy` and `AbstractLogdetStrategy`, composed with
    `ComposedSolver`.

    Examples:
        A structural-dispatch strategy behind a solve and a log-determinant:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> solver = gaussx.DenseSolver()
        >>> isinstance(solver, gaussx.AbstractSolverStrategy)
        True
        >>> A = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> [round(float(v), 4) for v in solver.solve(A, jnp.ones(2))]
        [0.5, 0.25]
        >>> round(float(solver.logdet(A)), 4)
        2.0794
    """
