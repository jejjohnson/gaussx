"""Fifth-order spherical-radial cubature integrator."""

from __future__ import annotations

from collections.abc import Callable

import jax
from jaxtyping import Array, Float

from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._quadrature import fifth_order_cubature_points
from gaussx._quadrature._types import GaussianState, PropagationResult


class FifthOrderCubatureIntegrator(AbstractIntegrator):
    r"""Fifth-order fully symmetric cubature (McNamee & Stenger).

    Exact for polynomials up to total degree 5 in ``N`` variables, using
    ``2 N^2 + 1`` points. This is the practical middle ground between the
    degree-3 sigma-point rules and a tensor-product Gauss-Hermite rule:

    ==================================  ==============  ================
    Rule                                Points          Exact to degree
    ==================================  ==============  ================
    `UnscentedIntegrator`               ``2N + 1``      3
    `FifthOrderCubatureIntegrator`      ``2 N^2 + 1``   5
    `GaussHermiteIntegrator` (order H)  ``H^N``         ``2H − 1``
    ==================================  ==============  ================

    It is the default integrator in BayesNewton, and the lowest-degree rule
    that gets the curvature right for likelihoods whose log-density behaves
    like a quartic near the mode (Poisson, Bernoulli with heavy-tailed
    inputs).

    Note:
        The rule carries negative weights on the axis shell for ``N > 4``
        — see `gaussx.fifth_order_cubature_points`. Moments through degree
        5 remain exact, but an output covariance assembled from very few
        function evaluations is not guaranteed PSD in high dimension.
    """

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate Gaussian via fifth-order cubature."""
        chi, weights = fifth_order_cubature_points(state.mean, state.cov)
        Y = jax.vmap(fn)(chi)
        return assemble_propagation_result(chi, Y, state.mean, weights)

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """Return the fifth-order cubature points and weights.

        Args:
            state: Input Gaussian distribution.

        Returns:
            Tuple ``(points, w_m, w_c)``. The rule uses a single weight
            set, so ``w_m`` and ``w_c`` are the same array.
        """
        chi, weights = fifth_order_cubature_points(state.mean, state.cov)
        return chi, weights, weights
