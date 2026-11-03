"""A strategy wrapper that carries the PRNG key of its stochastic logdet.

A stochastic log |det A| draws probe vectors z^(i), i = 1, …, N_probe; the
key bound here decides which.
"""

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

        log |det A| ≈ (1 / N_probe) ∑_i z^(i)ᵀ log(A) z^(i),   z^(i) from ``key``

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

    Examples:
        Two keys give two different SLQ estimates; a bound key repeats:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = 3.0 * jnp.eye(6) + jnp.eye(6, k=1) + jnp.eye(6, k=-1)
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> slq = gaussx.SLQLogdet(num_probes=4, lanczos_order=2)
        >>> first = gaussx.KeyedSolver(slq, jr.key(0)).logdet(A)
        >>> second = gaussx.KeyedSolver(slq, jr.key(1)).logdet(A)
        >>> bool(first == second)
        False
        >>> bool(first == gaussx.KeyedSolver(slq, jr.key(0)).logdet(A))
        True

        A logdet-only strategy cannot solve:

        >>> keyed = gaussx.KeyedSolver(slq, jr.key(0))
        >>> keyed.solve(A, jnp.ones(6))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: SLQLogdet is a logdet-only strategy; ...
    """

    strategy: AbstractLogdetStrategy
    key: jax.Array

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b with the wrapped strategy.

            x = A⁻¹ b

        Args:
            operator: Linear operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.

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
        """Compute log |det A| with the wrapped strategy and the bound key.

            log |det A| ≈ (1 / N_probe) ∑_i z^(i)ᵀ log(A) z^(i)

        Args:
            operator: Linear operator A, N × N.
            key: Overrides the bound key when given.

        Returns:
            Scalar log |det A|, estimated with the bound (or given) key.
        """
        probe_key = self.key if key is None else key
        return self.strategy.logdet(operator, key=probe_key)
