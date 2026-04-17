"""Multivariate normal distribution parameterized by a lineax operator."""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.typing
import lineax as lx
import numpyro.distributions as dist
from einops import rearrange
from jaxtyping import Array, Float
from numpyro.distributions.util import lazy_property, validate_sample

from gaussx._distributions._gaussian import (
    _gaussian_log_prob_residual,
    gaussian_entropy,
)
from gaussx._primitives._cholesky import cholesky as _cholesky
from gaussx._primitives._diag import diag as _diag
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractSolverStrategy


_BATCH_AXIS_NAMES = tuple("abcdefghijklmnopqrstuvwxyz")


def _reshape_batch(
    values: Float[Array, " flat"],
    batch_shape: tuple[int, ...],
) -> Float[Array, "*batch"]:
    if not batch_shape:
        return values[0]
    batch_axes = _BATCH_AXIS_NAMES[: len(batch_shape)]
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    return rearrange(values, f"({batch_pattern}) -> {batch_pattern}", **axis_lengths)


def _reshape_samples(
    values: Float[Array, "flat N"],
    batch_shape: tuple[int, ...],
) -> Float[Array, "*batch N"]:
    if not batch_shape:
        return values[0]
    batch_axes = _BATCH_AXIS_NAMES[: len(batch_shape)]
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    return rearrange(
        values,
        f"({batch_pattern}) N -> {batch_pattern} N",
        **axis_lengths,
    )


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
        loc: Float[Array, "*batch N"],
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
        return _gaussian_log_prob_residual(
            residual, self.cov_operator, solver=self.solver
        )

    @validate_sample
    def log_prob(self, value: Float[Array, "*batch N"]) -> Float[Array, "*batch"]:
        residual = value - self.loc
        leading_shape = residual.shape[:-1]
        residual_flat = rearrange(residual, "... D -> (...) D")
        log_prob_flat = jax.vmap(self._log_prob_single)(residual_flat)
        return _reshape_batch(log_prob_flat, leading_shape)

    def sample(
        self,
        key: jax.dtypes.prng_key | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch N"]:
        if key is None:
            raise ValueError(
                "PRNG key must be provided to sample from MultivariateNormal."
            )
        L = _cholesky(self.cov_operator)
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = jax.random.normal(key, shape=shape)  # type: ignore[arg-type]
        eps_flat = rearrange(eps, "... D -> (...) D")
        samples_flat = jax.vmap(L.mv)(eps_flat)
        return self.loc + _reshape_samples(samples_flat, shape[:-1])

    @lazy_property
    def mean(self) -> Float[Array, "*batch N"]:
        return self.loc

    @lazy_property
    def variance(self) -> Float[Array, "*batch N"]:
        return jnp.broadcast_to(
            _diag(self.cov_operator), self.batch_shape + self.event_shape
        )

    def entropy(self) -> Float[Array, ""]:
        return gaussian_entropy(self.cov_operator, solver=self.solver)
