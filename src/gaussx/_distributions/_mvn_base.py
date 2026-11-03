"""Shared base class of the operator-parameterised multivariate normals.

A multivariate normal x ~ 𝒩(μ, Σ) on ℝ^{D_x} stores one native operator:
the covariance Σ or the precision Λ = Σ⁻¹. The residual is r = x − μ, and L
is the lower Cholesky factor of Σ = L Lᵀ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions.util import lazy_property, validate_sample

from gaussx._distributions._utils import _reshape_batch
from gaussx._einx import rearrange
from gaussx._strategies._base import AbstractSolverStrategy


class AbstractMultivariateNormal(dist.Distribution):
    """Common surface of `MultivariateNormal` and `MultivariateNormalPrecision`.

        x ~ 𝒩(μ, Σ),   Λ = Σ⁻¹

    A subclass stores one native operator: the covariance Σ
    (`MultivariateNormal`) or the precision Λ (`MultivariateNormalPrecision`).
    The other one is the lazy `gaussx.inv` of it. Code written against this
    base works for either parameterisation:

    - ``covariance_operator`` / ``precision_operator``: lineax operators,
      one native and one a lazy inverse.
    - ``covariance_matrix`` / ``precision_matrix`` / ``scale_tril``: dense
      arrays with numpyro's names and batch broadcasting. They materialise
      D_x × D_x matrices, so they are for small problems and interop.
    - ``mean``, ``log_prob``, `kl`.

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    Examples:
        Both parameterisations expose the same accessors:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._distributions import (
        ...     AbstractMultivariateNormal,
        ...     MultivariateNormal,
        ...     MultivariateNormalPrecision,
        ... )
        >>> covariance = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> precision = lx.MatrixLinearOperator(
        ...     0.5 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> p = MultivariateNormal(jnp.zeros(2), covariance)
        >>> q = MultivariateNormalPrecision(jnp.zeros(2), precision)
        >>> isinstance(q, AbstractMultivariateNormal)
        True
        >>> bool(jnp.allclose(p.precision_matrix, q.precision_matrix))
        True

        Equal Gaussians have zero KL, whatever the parameterisation:

        >>> bool(jnp.allclose(p.kl(q), 0.0, atol=1e-6))
        True

        Batched means broadcast the dense accessors:

        >>> batched = MultivariateNormal(jnp.zeros((3, 2)), covariance)
        >>> batched.covariance_matrix.shape, batched.log_prob(jnp.ones(2)).shape
        ((3, 2, 2), (3,))
    """

    arg_constraints = {"loc": dist.constraints.real_vector}  # noqa: RUF012
    support = dist.constraints.real_vector
    reparametrized_params = ["loc"]  # noqa: RUF012

    loc: Float[Array, "*batch N"]
    solver: AbstractSolverStrategy

    @property
    def covariance_operator(self) -> lx.AbstractLinearOperator:
        """The covariance Σ as a lineax operator."""
        raise NotImplementedError

    @property
    def precision_operator(self) -> lx.AbstractLinearOperator:
        """The precision Λ = Σ⁻¹ as a lineax operator."""
        raise NotImplementedError

    def _log_prob_single(self, residual: Float[Array, " N"]) -> Float[Array, ""]:
        """Log-density of one event from its residual r = x − μ.

        Args:
            residual: Residual r = x − μ of a single event.

        Returns:
            The scalar log 𝒩(x; μ, Σ).
        """
        raise NotImplementedError

    @validate_sample
    def log_prob(self, value: Float[Array, "*batch N"]) -> Float[Array, "*batch"]:
        """Log-density log 𝒩(x; μ, Σ), broadcast over the batch shape.

            log 𝒩(x; μ, Σ) = −½ (D_x log 2π + log det Σ + rᵀ Σ⁻¹ r),
            r = x − μ

        Algorithm:
            1. Form the residuals r = x − μ, broadcasting x against μ.
            2. Flatten the leading axes and evaluate one event at a time.
            3. Restore the leading shape.

        Args:
            value: Points x, with any leading shape that broadcasts against
                the batch shape.

        Returns:
            One log-density per broadcast batch element.
        """
        # r = x − μ
        residual = value - self.loc
        leading_shape = residual.shape[:-1]
        residual_flat = rearrange(residual, "... D -> (...) D")
        log_prob_flat = jax.vmap(self._log_prob_single)(residual_flat)
        return _reshape_batch(log_prob_flat, leading_shape)

    @lazy_property
    def mean(self) -> Float[Array, "*batch N"]:
        """The mean μ."""
        return self.loc

    def _broadcast_matrix(self, matrix: Float[Array, "N N"]) -> Array:
        """Broadcast a shared D_x × D_x matrix over the batch shape.

        Args:
            matrix: The shared matrix.

        Returns:
            The matrix repeated for every batch element.
        """
        return jnp.broadcast_to(matrix, self.batch_shape + self.event_shape * 2)

    @lazy_property
    def covariance_matrix(self) -> Float[Array, "*batch N N"]:
        """Dense Σ, broadcast over the batch shape."""
        return self._broadcast_matrix(self.covariance_operator.as_matrix())

    @lazy_property
    def precision_matrix(self) -> Float[Array, "*batch N N"]:
        """Dense Λ = Σ⁻¹, broadcast over the batch shape."""
        return self._broadcast_matrix(self.precision_operator.as_matrix())

    @lazy_property
    def scale_tril(self) -> Float[Array, "*batch N N"]:
        """Dense lower Cholesky factor L of Σ = L Lᵀ (numpyro's name)."""
        # L Lᵀ = Σ
        return jnp.linalg.cholesky(self.covariance_matrix)

    def kl(self, other: AbstractMultivariateNormal) -> Float[Array, "*batch"]:
        """KL(first ‖ second): KL[self ‖ other] in closed form.

            KL[p ‖ q] = ½ (tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − D_x
                           + log det Σ_q − log det Σ_p)

        with p = self and q = other. Any mix of covariance and precision
        parameterisations works. A precision-parameterised ``other`` needs no
        solve for tr(Σ_q⁻¹ Σ_p) or the quadratic term, because the inverse of
        its lazy covariance is its precision. Batch shapes broadcast as in
        numpyro's ``kl_divergence``. Unlike
        ``numpyro.distributions.kl_divergence``, it uses the closed form
        whatever the solver strategies are.

        Args:
            other: The second Gaussian q, with the same event shape.

        Returns:
            The KL divergence, with the broadcast batch shape.
        """
        # lazy import, cycle: _distributions._numpyro_kl -> _distributions._mvn_base
        from gaussx._distributions._numpyro_kl import closed_form_kl

        return closed_form_kl(self, other)
