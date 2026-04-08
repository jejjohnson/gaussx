"""Gauss-Hermite quadrature integrator for uncertainty propagation."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._uncertain._assembly import assemble_propagation_result
from gaussx._uncertain._integrator import AbstractIntegrator
from gaussx._uncertain._types import GaussianState, PropagationResult


class GaussHermiteIntegrator(AbstractIntegrator):
    r"""Gauss-Hermite quadrature integrator.

    Approximates Gaussian expectations using tensor-product Gauss-Hermite
    quadrature::

        E[g(f)] \approx \sum_i w_i \cdot g(\mu + L z_i)

    where ``(z_i, w_i)`` are GH points/weights in standard normal space
    and ``L`` is the square root of the covariance.

    Exact for polynomials up to degree ``2 * order - 1``.
    Complexity: ``O(order^dim)``, practical for ``dim <= ~5``.

    Args:
        order: Number of quadrature points per dimension. Default ``20``.
    """

    order: int = eqx.field(static=True, default=20)

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate Gaussian via Gauss-Hermite quadrature."""
        from gaussx._primitives._sqrt import sqrt
        from gaussx._sugar._quadrature import gauss_hermite_points

        mu = state.mean
        N = mu.shape[0]

        # GH points in standard normal space
        z, w = gauss_hermite_points(self.order, N)

        # Transform to input space: xᵢ = μ + S zᵢ
        S = sqrt(state.cov).as_matrix()
        chi = mu[None, :] + z @ S.T  # (P, N)

        # Normalize weights to sum to 1
        w = w / jnp.sum(w)

        # Propagate all quadrature points
        Y = jax.vmap(fn)(chi)  # (P, M)

        return assemble_propagation_result(chi, Y, mu, w)
