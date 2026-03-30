"""Multivariate normal distribution parameterized by precision operator."""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.typing
import lineax as lx
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions.util import lazy_property, validate_sample

from gaussx._primitives._cholesky import cholesky as _cholesky
from gaussx._primitives._diag import diag as _diag
from gaussx._primitives._inv import inv as _inv
from gaussx._primitives._solve import solve as _solve
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractSolverStrategy


class MultivariateNormalPrecision(dist.Distribution):
    """Multivariate normal parameterized by a precision (inverse covariance) operator.

    This is the natural parameterization for many inference algorithms
    (e.g. message passing, variational inference in natural coordinates).
    The precision operator ``Lambda`` satisfies ``Lambda = Sigma^{-1}``.

    Args:
        loc: Mean vector of shape ``(N,)``.
        prec_operator: Precision matrix as a lineax linear operator of
            shape ``(N, N)``.
        solver: Solver strategy for ``solve`` and ``logdet``. Defaults
            to ``AutoSolver()``.
        validate_args: Whether to validate input arguments.

    Example::

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> Lambda = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> d = gaussx.MultivariateNormalPrecision(jnp.zeros(3), Lambda)
        >>> d.log_prob(jnp.ones(3))
    """

    arg_constraints: ClassVar[dict] = {"loc": dist.constraints.real_vector}
    support = dist.constraints.real_vector
    reparametrized_params: ClassVar[list] = ["loc"]
    pytree_data_fields = ("loc", "prec_operator", "solver")

    def __init__(
        self,
        loc: Float[Array, " N"],
        prec_operator: lx.AbstractLinearOperator,
        solver: AbstractSolverStrategy | None = None,
        *,
        validate_args: bool | None = None,
    ) -> None:
        if solver is None:
            solver = AutoSolver()
        self.loc = loc
        self.prec_operator = prec_operator
        self.solver = solver
        event_shape = loc.shape[-1:]
        batch_shape = loc.shape[:-1]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _log_prob_single(self, residual: Float[Array, " N"]) -> Float[Array, ""]:
        quad = residual @ self.prec_operator.mv(residual)
        ld = self.solver.logdet(self.prec_operator)
        n = self.loc.shape[-1]
        return -0.5 * (n * jnp.log(2.0 * jnp.pi) - ld + quad)

    @validate_sample
    def log_prob(self, value: Float[Array, ...]) -> Float[Array, ...]:
        residual = value - self.loc
        if residual.ndim > 1:
            return jax.vmap(self._log_prob_single)(residual)
        return self._log_prob_single(residual)

    def sample(
        self,
        key: jax.dtypes.prng_key | None,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.typing.ArrayLike:
        assert key is not None
        L = _cholesky(self.prec_operator)
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = jax.random.normal(key, shape=shape)  # type: ignore[arg-type]

        def _solve_one(z):
            return _solve(L.T, z)

        if sample_shape:
            solve_batched = _solve_one
            for _ in sample_shape:
                solve_batched = jax.vmap(solve_batched)
            return self.loc + solve_batched(eps)
        return self.loc + _solve_one(eps)

    @lazy_property
    def mean(self) -> Float[Array, " N"]:
        return self.loc

    @lazy_property
    def variance(self) -> Float[Array, " N"]:
        return _diag(_inv(self.prec_operator))

    def entropy(self) -> Float[Array, ""]:
        n = self.loc.shape[-1]
        ld = self.solver.logdet(self.prec_operator)
        return 0.5 * (n * (1.0 + jnp.log(2.0 * jnp.pi)) - ld)
