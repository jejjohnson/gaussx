"""Multivariate normal distribution parameterized by precision operator."""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from einops import rearrange
from jaxtyping import Array, Float
from numpyro.distributions.util import lazy_property, validate_sample

from gaussx._primitives._cholesky import cholesky as _cholesky
from gaussx._primitives._diag import diag as _diag
from gaussx._primitives._inv import inv as _inv
from gaussx._primitives._solve import solve as _solve
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractSolverStrategy


def _axis_names(count: int) -> tuple[str, ...]:
    names = []
    for index in range(count):
        value = index
        chars = []
        while True:
            value, remainder = divmod(value, 26)
            chars.append(chr(ord("a") + remainder))
            if value == 0:
                break
            value -= 1
        names.append("".join(reversed(chars)))
    return tuple(names)


def _reshape_batch(
    values: Float[Array, " flat"],
    batch_shape: tuple[int, ...],
) -> Float[Array, "*batch"]:
    if not batch_shape:
        return values[0]
    batch_axes = _axis_names(len(batch_shape))
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    return rearrange(values, f"({batch_pattern}) -> {batch_pattern}", **axis_lengths)


def _reshape_samples(
    values: Float[Array, "flat N"],
    batch_shape: tuple[int, ...],
) -> Float[Array, "*batch N"]:
    if not batch_shape:
        return values[0]
    batch_axes = _axis_names(len(batch_shape))
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    return rearrange(
        values,
        f"({batch_pattern}) N -> {batch_pattern} N",
        **axis_lengths,
    )


class MultivariateNormalPrecision(dist.Distribution):
    """Multivariate normal parameterized by a precision (inverse covariance) operator.

    This is the natural parameterization for many inference algorithms
    (e.g. message passing, variational inference in natural coordinates).
    The precision operator ``Lambda`` satisfies ``Lambda = Sigma^{-1}``.

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

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
        >>> from gaussx._distributions import MultivariateNormalPrecision
        >>> Lambda = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> d = MultivariateNormalPrecision(jnp.zeros(3), Lambda)
        >>> d.log_prob(jnp.ones(3))
    """

    arg_constraints: ClassVar[dict] = {"loc": dist.constraints.real_vector}
    support = dist.constraints.real_vector
    reparametrized_params: ClassVar[list] = ["loc"]
    pytree_data_fields = ("loc", "prec_operator", "solver")

    def __init__(
        self,
        loc: Float[Array, "*batch N"],
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
        quad = jnp.sum(residual * self.prec_operator.mv(residual), axis=-1)
        ld = self.solver.logdet(self.prec_operator)
        n = self.loc.shape[-1]
        return -0.5 * (n * jnp.log(2.0 * jnp.pi) - ld + quad)

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
                "PRNG key must be provided to sample from MultivariateNormalPrecision."
            )
        L = _cholesky(self.prec_operator)
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = jax.random.normal(key, shape=shape)  # type: ignore[arg-type]

        def _solve_one(z):
            return _solve(L.T, z)

        eps_flat = rearrange(eps, "... D -> (...) D")
        samples_flat = jax.vmap(_solve_one)(eps_flat)
        return self.loc + _reshape_samples(samples_flat, shape[:-1])

    @lazy_property
    def mean(self) -> Float[Array, "*batch N"]:
        return self.loc

    @lazy_property
    def variance(self) -> Float[Array, "*batch N"]:
        return jnp.broadcast_to(
            _diag(_inv(self.prec_operator)), self.batch_shape + self.event_shape
        )

    def entropy(self) -> Float[Array, ""]:
        n = self.loc.shape[-1]
        ld = self.solver.logdet(self.prec_operator)
        return 0.5 * (n * (1.0 + jnp.log(2.0 * jnp.pi)) - ld)
