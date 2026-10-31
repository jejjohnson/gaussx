"""Route x = A⁻¹ b and log |det A| through a strategy, or structural dispatch.

Math code takes ``solver=None`` by default and calls these helpers: a strategy
decides the algorithm when one is given, and the structural primitives
`gaussx.solve` and `gaussx.logdet` decide it otherwise.
"""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractLogdetStrategy, AbstractSolveStrategy
from gaussx._strategies._lineax import as_solve_strategy


def dispatch_solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
    solver: AbstractSolveStrategy | lx.AbstractLinearSolver | None = None,
) -> Float[Array, " n"]:
    """Solve A x = b with ``solver``, or with structural dispatch.

        x = A⁻¹ b

    Args:
        operator: Linear operator A, N × N.
        vector: Right-hand side b, of length N.
        solver: Solve strategy, or a lineax solver (wrapped in
            `gaussx.LineaxSolver`). ``None`` uses `gaussx.solve`.

    Returns:
        Solution x, of length N.

    Raises:
        TypeError: If ``solver`` is neither a strategy nor a lineax solver.

    Examples:
        Structural dispatch, a strategy and a lineax solver agree:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> from gaussx._strategies._dispatch import dispatch_solve
        >>> A = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> b = jnp.ones(2)
        >>> [round(float(v), 4) for v in dispatch_solve(A, b)]
        [0.5, 0.25]
        >>> x = dispatch_solve(A, b, gaussx.DenseSolver())
        >>> bool(jnp.allclose(x, jnp.array([0.5, 0.25])))
        True
        >>> x = dispatch_solve(A, b, lx.Diagonal())
        >>> bool(jnp.allclose(x, jnp.array([0.5, 0.25])))
        True
    """
    solver = as_solve_strategy(solver)
    if solver is not None:
        return solver.solve(operator, vector)

    return solve(operator, vector)


def dispatch_logdet(
    operator: lx.AbstractLinearOperator,
    solver: AbstractLogdetStrategy | None = None,
    *,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """Compute log |det A| with ``solver``, or with structural dispatch.

        log |det A| = ∑_d log |γ_d|

    with γ_d, d = 1, …, N, the eigenvalues of A.

    Args:
        operator: Linear operator A, N × N.
        solver: Logdet strategy. ``None`` uses `gaussx.logdet`.
        key: Optional PRNG key, forwarded to a stochastic estimator.

    Returns:
        Scalar log |det A|, exact or estimated.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> from gaussx._strategies._dispatch import dispatch_logdet
        >>> A = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> round(float(dispatch_logdet(A)), 4)
        2.0794
        >>> round(float(dispatch_logdet(A, gaussx.DenseSolver())), 4)
        2.0794
    """
    if solver is not None:
        return solver.logdet(operator, key=key)

    return logdet(operator)
