"""Third-order spherical-radial cubature integrator (CKF)."""

from __future__ import annotations

from collections.abc import Callable

import jax
from jaxtyping import Array, Float

from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._quadrature import cubature_points
from gaussx._quadrature._types import GaussianState, PropagationResult


class CubatureIntegrator(AbstractIntegrator):
    r"""Third-order spherical-radial cubature.

    The ``2N``-point, equally weighted rule of Arasaratnam & Haykin
    (2009), exact for polynomials up to total degree 3. Driving
    `gaussx.nonlinear_kalman_filter` with it gives the textbook cubature
    Kalman filter (CKF).

    It is the degree-3 sibling of `gaussx.FifthOrderCubatureIntegrator`,
    and closely related to `gaussx.UnscentedIntegrator`: the unscented
    transform with ``alpha=1``, ``beta=0``, ``kappa=0`` places its ``2N``
    non-centre points identically, differing only by the centre point and
    its weight. Unlike the scaled unscented transform, every weight here is
    positive (``1 / 2N``), so an assembled covariance cannot be dragged
    indefinite by a negative weight.
    """

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate Gaussian via third-order cubature."""
        chi, weights = cubature_points(state.mean, state.cov)
        Y = jax.vmap(fn)(chi)
        return assemble_propagation_result(chi, Y, state.mean, weights)

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """Return the third-order cubature points and weights.

        Args:
            state: Input Gaussian distribution.

        Returns:
            Tuple ``(points, w_m, w_c)`` with ``P = 2N``. The rule uses a
            single weight set, so ``w_m`` and ``w_c`` are the same array.
        """
        chi, weights = cubature_points(state.mean, state.cov)
        return chi, weights, weights
