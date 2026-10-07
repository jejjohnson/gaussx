"""A strategy wrapper that carries the PRNG key of its stochastic logdet."""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import (
    AbstractLogdetStrategy,
    AbstractSolverStrategy,
    AbstractSolveStrategy,
)


class KeyedSolver(AbstractSolverStrategy):
    """Bind a PRNG key to a strategy's stochastic log-determinant (gh-384).

    A stochastic logdet strategy (`gaussx.SLQLogdet`, `gaussx.CGSolver`,
    `gaussx.BBMMSolver`, ...) called without a ``key`` draws its probes from
    ``PRNGKey(seed)``, so every call sees *the same* probes: common random
    numbers. That suits stochastic-gradient training (the objective is a
    smooth, fixed function of the parameters), but averaging such estimates
    reduces no variance, and MCMC then targets a fixed pseudo-likelihood.

    Distribution methods such as ``MultivariateNormal.log_prob`` cannot take
    a key under the numpyro contract. Wrap the strategy instead and pass a
    fresh key per step. The key is a PyTree leaf, so it can change under
    ``jax.jit`` without recompiling:

        solver = gaussx.KeyedSolver(gaussx.CGSolver(), key)
        mvn = gaussx.MultivariateNormal(loc, cov, solver=solver)

    The functional API (`gaussx.gaussian_log_prob`,
    `gaussx.gaussian_entropy`, `gaussx.kl_standard_normal`) takes
    ``key=`` directly.

    Attributes:
        strategy: The wrapped strategy. ``solve`` needs it to be a solve
            strategy as well.
        key: The PRNG key passed to ``strategy.logdet``. A key passed to
            `logdet` explicitly takes precedence.
    """

    strategy: AbstractLogdetStrategy
    key: jax.Array

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve ``A x = b`` with the wrapped strategy.

        Args:
            operator: Linear operator ``A``.
            vector: Right-hand side ``b``, shape ``(n,)``.

        Returns:
            Solution ``x``, shape ``(n,)``.

        Raises:
            TypeError: If the wrapped strategy has no ``solve``.
        """
        if not isinstance(self.strategy, AbstractSolveStrategy):
            raise TypeError(
                f"{type(self.strategy).__name__} is a logdet-only strategy; "
                "wrap a solver strategy (or a ComposedSolver) to solve."
            )
        return self.strategy.solve(operator, vector)

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """``log|det(A)|`` from the wrapped strategy, with the bound key.

        Args:
            operator: Linear operator ``A``.
            key: Overrides the bound key when given.

        Returns:
            Scalar log-determinant.
        """
        return self.strategy.logdet(operator, key=self.key if key is None else key)
