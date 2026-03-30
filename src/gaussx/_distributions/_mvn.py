"""Multivariate normal distribution parameterized by a lineax operator."""

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
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractSolverStrategy


class MultivariateNormal(dist.Distribution):
    """Multivariate normal parameterized by a lineax linear operator.

    Unlike ``numpyro.distributions.MultivariateNormal`` which requires
    dense arrays, this distribution accepts any
    ``lineax.AbstractLinearOperator`` as its covariance. This enables
    efficient log-prob, sampling, and entropy for structured covariances
    (Kronecker, block-diagonal, low-rank, diagonal, etc.) via gaussx
    structural dispatch.

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    Args:
        loc: Mean vector of shape ``(N,)``.
        cov_operator: Covariance as a lineax linear operator of shape
            ``(N, N)``.
        solver: Solver strategy for ``solve`` and ``logdet``. Defaults
            to ``AutoSolver()``.
        validate_args: Whether to validate input arguments.

    Example::

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._distributions import MultivariateNormal
        >>> Sigma = lx.MatrixLinearOperator(
        ...     jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> d = MultivariateNormal(jnp.zeros(3), Sigma)
        >>> d.log_prob(jnp.ones(3))
    """

    arg_constraints: ClassVar[dict] = {"loc": dist.constraints.real_vector}
    support = dist.constraints.real_vector
    reparametrized_params: ClassVar[list] = ["loc"]
    pytree_data_fields = ("loc", "cov_operator", "solver")

    def __init__(
        self,
        loc: Float[Array, " N"],
        cov_operator: lx.AbstractLinearOperator,
        solver: AbstractSolverStrategy | None = None,
        *,
        validate_args: bool | None = None,
    ) -> None:
        if solver is None:
            solver = AutoSolver()
        self.loc = loc
        self.cov_operator = cov_operator
        self.solver = solver
        event_shape = loc.shape[-1:]
        batch_shape = loc.shape[:-1]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _log_prob_single(self, residual: Float[Array, " N"]) -> Float[Array, ""]:
        alpha = self.solver.solve(self.cov_operator, residual)
        quad = jnp.sum(residual * alpha, axis=-1)
        ld = self.solver.logdet(self.cov_operator)
        n = self.loc.shape[-1]
        return -0.5 * (n * jnp.log(2.0 * jnp.pi) + ld + quad)

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
        if key is None:
            raise ValueError(
                "PRNG key must be provided to sample from MultivariateNormal."
            )
        L = _cholesky(self.cov_operator)
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = jax.random.normal(key, shape=shape)  # type: ignore[arg-type]
        if sample_shape:
            # vmap over leading sample dimensions
            mv_batched = L.mv
            for _ in sample_shape:
                mv_batched = jax.vmap(mv_batched)
            return self.loc + mv_batched(eps)
        return self.loc + L.mv(eps)

    @lazy_property
    def mean(self) -> Float[Array, " N"]:
        return self.loc

    @lazy_property
    def variance(self) -> Float[Array, " N"]:
        return _diag(self.cov_operator)

    def entropy(self) -> Float[Array, ""]:
        n = self.loc.shape[-1]
        ld = self.solver.logdet(self.cov_operator)
        return 0.5 * (n * (1.0 + jnp.log(2.0 * jnp.pi)) + ld)
