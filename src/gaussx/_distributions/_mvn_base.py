"""Shared base class for the operator-parameterised multivariate normals."""

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
    r"""Common surface of `MultivariateNormal` and `MultivariateNormalPrecision`.

    A subclass stores one *native* operator: the covariance $\Sigma$
    (`MultivariateNormal`) or the precision $\Lambda = \Sigma^{-1}$
    (`MultivariateNormalPrecision`). The other one is the lazy
    `gaussx.inv` of it. Code written against this base works for either
    parameterisation:

    - ``covariance_operator`` / ``precision_operator``: lineax operators,
      one native and one a lazy inverse.
    - ``covariance_matrix`` / ``precision_matrix`` / ``scale_tril``: dense
      arrays with numpyro's names and batch broadcasting. They materialise
      $N \times N$ matrices, so they are for small problems and interop.
    - ``mean``, ``log_prob``, `kl`.

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).
    """

    arg_constraints = {"loc": dist.constraints.real_vector}  # noqa: RUF012
    support = dist.constraints.real_vector
    reparametrized_params = ["loc"]  # noqa: RUF012

    loc: Float[Array, "*batch N"]
    solver: AbstractSolverStrategy

    @property
    def covariance_operator(self) -> lx.AbstractLinearOperator:
        r"""The covariance $\Sigma$ as a lineax operator."""
        raise NotImplementedError

    @property
    def precision_operator(self) -> lx.AbstractLinearOperator:
        r"""The precision $\Lambda = \Sigma^{-1}$ as a lineax operator."""
        raise NotImplementedError

    def _log_prob_single(self, residual: Float[Array, " N"]) -> Float[Array, ""]:
        raise NotImplementedError

    @validate_sample
    def log_prob(self, value: Float[Array, "*batch N"]) -> Float[Array, "*batch"]:
        residual = value - self.loc
        leading_shape = residual.shape[:-1]
        residual_flat = rearrange(residual, "... D -> (...) D")
        log_prob_flat = jax.vmap(self._log_prob_single)(residual_flat)
        return _reshape_batch(log_prob_flat, leading_shape)

    @lazy_property
    def mean(self) -> Float[Array, "*batch N"]:
        return self.loc

    def _broadcast_matrix(self, matrix: Float[Array, "N N"]) -> Array:
        return jnp.broadcast_to(matrix, self.batch_shape + self.event_shape * 2)

    @lazy_property
    def covariance_matrix(self) -> Float[Array, "*batch N N"]:
        r"""Dense $\Sigma$, broadcast over the batch shape."""
        return self._broadcast_matrix(self.covariance_operator.as_matrix())

    @lazy_property
    def precision_matrix(self) -> Float[Array, "*batch N N"]:
        r"""Dense $\Lambda = \Sigma^{-1}$, broadcast over the batch shape."""
        return self._broadcast_matrix(self.precision_operator.as_matrix())

    @lazy_property
    def scale_tril(self) -> Float[Array, "*batch N N"]:
        r"""Dense lower Cholesky factor $L$ of $\Sigma = L L^\top$ (numpyro's name)."""
        return jnp.linalg.cholesky(self.covariance_matrix)

    def kl(self, other: AbstractMultivariateNormal) -> Float[Array, "*batch"]:
        r"""$\mathrm{KL}(\text{self} \,\|\, \text{other})$ in closed form.

        Any mix of covariance and precision parameterisations works. A
        precision-parameterised ``other`` needs no solve for
        $\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)$ or the quadratic term,
        because the inverse of its lazy covariance is its precision. Batch
        shapes broadcast as in numpyro's ``kl_divergence``. Unlike
        ``numpyro.distributions.kl_divergence``, it uses the closed form
        whatever the solver strategies are.

        Args:
            other: The second Gaussian, with the same event shape.

        Returns:
            The KL divergence, with the broadcast batch shape.
        """
        from gaussx._distributions._numpyro_kl import closed_form_kl

        return closed_form_kl(self, other)
