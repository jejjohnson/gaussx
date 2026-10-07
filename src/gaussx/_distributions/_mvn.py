"""Multivariate normal distribution parameterized by a lineax operator."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from numpyro.distributions.util import lazy_property

from gaussx._distributions._gaussian import (
    _gaussian_log_prob_residual,
    gaussian_entropy,
)
from gaussx._distributions._mvn_base import AbstractMultivariateNormal
from gaussx._distributions._sample import sample_mvn
from gaussx._distributions._utils import _unflatten_sample_axis
from gaussx._primitives._diag import diag as _diag
from gaussx._primitives._inv import inv as _inv
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractSolverStrategy


class MultivariateNormal(AbstractMultivariateNormal):
    """Multivariate normal parameterized by a lineax linear operator.

    Covariance-parameterised member of `AbstractMultivariateNormal`, which
    supplies the shared accessors (``covariance_operator``,
    ``precision_operator``, ``covariance_matrix``, ``scale_tril``, `kl`, ...).
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

    Examples:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._distributions import MultivariateNormal
        >>> Sigma = lx.MatrixLinearOperator(
        ...     jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> d = MultivariateNormal(jnp.zeros(3), Sigma)
        >>> round(float(d.log_prob(jnp.ones(3))), 4)
        -4.2568
    """

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

    @property
    def covariance_operator(self) -> lx.AbstractLinearOperator:
        """The covariance operator (native; the same object as ``cov_operator``)."""
        return self.cov_operator

    @property
    def precision_operator(self) -> lx.AbstractLinearOperator:
        """The precision: the lazy `gaussx.inv` of ``cov_operator``."""
        return _inv(self.cov_operator)

    def _log_prob_single(self, residual: Float[Array, " N"]) -> Float[Array, ""]:
        return _gaussian_log_prob_residual(
            residual, self.cov_operator, solver=self.solver
        )

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch N"]:
        if key is None:
            raise ValueError(
                "PRNG key must be provided to sample from MultivariateNormal."
            )
        # Delegate to sample_mvn, which dispatches on the covariance's
        # structure (Kronecker, low-rank, scalar multiples, ...) and takes a
        # symmetric square root at the dense fallback, so a semi-definite
        # covariance gives exact finite draws instead of a NaN Cholesky.
        num_samples = math.prod(sample_shape)
        loc = jnp.broadcast_to(self.loc, self.batch_shape + self.event_shape)
        if num_samples == 0:
            # sample_mvn needs at least one draw. Trace one abstractly, so an
            # empty sample still gets its shape checks and its dtype.
            one = jax.eval_shape(
                lambda: sample_mvn(loc, self.cov_operator, key=key, num_samples=1)
            )
            return jnp.zeros(sample_shape + one.shape[1:], dtype=one.dtype)
        draws = sample_mvn(loc, self.cov_operator, key=key, num_samples=num_samples)
        return _unflatten_sample_axis(draws, sample_shape)

    @lazy_property
    def variance(self) -> Float[Array, "*batch N"]:
        return jnp.broadcast_to(
            _diag(self.cov_operator), self.batch_shape + self.event_shape
        )

    def entropy(self) -> Float[Array, "*batch"]:
        # The entropy does not depend on loc, but numpyro's contract is one
        # entry per batch element (Independent.entropy sums over them).
        entropy = gaussian_entropy(self.cov_operator, solver=self.solver)
        return jnp.broadcast_to(entropy, self.batch_shape)
