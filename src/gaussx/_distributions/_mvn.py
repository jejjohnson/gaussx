"""Multivariate normal x ~ 𝒩(μ, Σ) with the covariance Σ as a lineax operator.

The residual is r = x − μ, D_x the event size and z ~ 𝒩(0, I) the standard
normal noise of a draw.
"""

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
    """Multivariate normal 𝒩(μ, Σ) whose covariance Σ is a lineax operator.

        x ~ 𝒩(μ, Σ)
        log 𝒩(x; μ, Σ) = −½ (D_x log 2π + log det Σ + (x − μ)ᵀ Σ⁻¹ (x − μ))

    Covariance-parameterised member of `AbstractMultivariateNormal`, which
    supplies the shared accessors (``covariance_operator``,
    ``precision_operator``, ``covariance_matrix``, ``scale_tril``, `kl`, …).
    Unlike ``numpyro.distributions.MultivariateNormal``, which needs dense
    arrays, it accepts any ``lineax.AbstractLinearOperator`` as Σ, so the
    log-density, sampling and entropy use gaussx structural dispatch for
    structured covariances (Kronecker, block-diagonal, low-rank, diagonal,
    …).

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    Args:
        loc: Mean μ; leading axes are the batch shape.
        cov_operator: Covariance Σ, shared by every batch element.
        solver: Solver strategy for the solves and log-determinants of Σ.
            Defaults to ``AutoSolver()``.
        validate_args: Whether to validate input arguments.

    Examples:
        A standard normal in three dimensions:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._distributions import MultivariateNormal
        >>> covariance = lx.MatrixLinearOperator(
        ...     jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> d = MultivariateNormal(jnp.zeros(3), covariance)
        >>> round(float(d.log_prob(jnp.ones(3))), 4)
        -4.2568

        Draws carry the sample shape in front of the event shape:

        >>> d.sample(jax.random.key(0), (4,)).shape
        (4, 3)

        A structured covariance gives the same entropy as its dense matrix:

        >>> diagonal = MultivariateNormal(
        ...     jnp.zeros(3), lx.DiagonalLinearOperator(jnp.ones(3))
        ... )
        >>> bool(jnp.allclose(diagonal.entropy(), d.entropy()))
        True
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
        """The covariance Σ (native; the same object as ``cov_operator``)."""
        return self.cov_operator

    @property
    def precision_operator(self) -> lx.AbstractLinearOperator:
        """The precision Λ = Σ⁻¹: the lazy `gaussx.inv` of ``cov_operator``."""
        return _inv(self.cov_operator)

    def _log_prob_single(self, residual: Float[Array, " N"]) -> Float[Array, ""]:
        """Log-density of one event from its residual r = x − μ.

            log 𝒩(x; μ, Σ) = −½ (D_x log 2π + log det Σ + rᵀ Σ⁻¹ r)

        Args:
            residual: Residual r = x − μ of a single event.

        Returns:
            The scalar log 𝒩(x; μ, Σ).
        """
        return _gaussian_log_prob_residual(
            residual, self.cov_operator, solver=self.solver
        )

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch N"]:
        """Draw x = μ + Σ^½ z with z ~ 𝒩(0, I), through `gaussx.sample_mvn`.

            x = μ + Σ^½ z,   z ~ 𝒩(0, I)

        `gaussx.sample_mvn` dispatches on the structure of Σ (Kronecker,
        low-rank, scalar multiples, …) and takes a symmetric square root at
        the dense fallback, so a semi-definite Σ gives exact finite draws
        instead of a NaN Cholesky.

        Args:
            key: PRNG key.
            sample_shape: Leading shape of the draws.

        Returns:
            Draws of shape ``sample_shape + batch_shape + event_shape``.

        Raises:
            ValueError: If ``key`` is ``None``.
        """
        if key is None:
            raise ValueError(
                "PRNG key must be provided to sample from MultivariateNormal."
            )
        num_samples = math.prod(sample_shape)
        # μ, one per batch element
        mean = jnp.broadcast_to(self.loc, self.batch_shape + self.event_shape)
        if num_samples == 0:
            # sample_mvn needs at least one draw. Trace one abstractly, so an
            # empty sample still gets its shape checks and its dtype.
            one = jax.eval_shape(
                lambda: sample_mvn(mean, self.cov_operator, key=key, num_samples=1)
            )
            return jnp.zeros(sample_shape + one.shape[1:], dtype=one.dtype)
        # x = μ + Σ^½ z, stacked along one leading axis
        draws = sample_mvn(mean, self.cov_operator, key=key, num_samples=num_samples)
        return _unflatten_sample_axis(draws, sample_shape)

    @lazy_property
    def variance(self) -> Float[Array, "*batch N"]:
        """Marginal variances diag(Σ), broadcast over the batch shape."""
        # diag(Σ)
        marginal_variance = _diag(self.cov_operator)
        return jnp.broadcast_to(marginal_variance, self.batch_shape + self.event_shape)

    def entropy(self) -> Float[Array, "*batch"]:
        """Entropy ℍ[𝒩(μ, Σ)], one entry per batch element.

            ℍ[𝒩(μ, Σ)] = ½ (D_x (1 + log 2π) + log det Σ)

        Returns:
            The entropy, broadcast over the batch shape.
        """
        # ℍ[𝒩(μ, Σ)] = ½ (D_x (1 + log 2π) + log det Σ)
        entropy = gaussian_entropy(self.cov_operator, solver=self.solver)
        # The entropy does not depend on μ, but numpyro's contract is one
        # entry per batch element (Independent.entropy sums over them).
        return jnp.broadcast_to(entropy, self.batch_shape)
