"""Gauss-Hermite quadrature integrator for uncertainty propagation."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

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
        from gaussx._sugar._quadrature import gauss_hermite_points

        mu = state.mean
        N = mu.shape[0]

        # GH points in standard normal space
        z, w = gauss_hermite_points(self.order, N)

        # Transform to input space: x_i = mu + S @ z_i
        # Use structured sqrt (eigh-based) which handles PSD covariances
        from gaussx._primitives._sqrt import sqrt

        S = sqrt(state.cov).as_matrix()
        chi = mu[None, :] + z @ S.T  # (P, N)

        # Normalize weights (GH weights sum to (2*pi)^{D/2} for
        # probabilists' Hermite; normalize to sum to 1)
        w = w / jnp.sum(w)

        # Propagate all quadrature points
        Y = jax.vmap(fn)(chi)  # (P, M)

        # Output moments
        mu_y = jnp.sum(w[:, None] * Y, axis=0)  # (M,)

        dy = Y - mu_y[None, :]  # (P, M)
        dx = chi - mu[None, :]  # (P, N)

        # Output covariance
        Sigma_y = jnp.sum(w[:, None, None] * (dy[:, :, None] * dy[:, None, :]), axis=0)
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)

        # Cross-covariance
        cross_cov = jnp.sum(
            w[:, None, None] * (dx[:, :, None] * dy[:, None, :]), axis=0
        )

        cov_y = lx.MatrixLinearOperator(Sigma_y, lx.positive_semidefinite_tag)
        out_state = GaussianState(mean=mu_y, cov=cov_y)

        return PropagationResult(state=out_state, cross_cov=cross_cov)
