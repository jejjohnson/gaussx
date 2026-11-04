"""Fifth-order fully symmetric cubature integrator."""

from __future__ import annotations

from collections.abc import Callable

import jax
from jaxtyping import Array, Float

from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._quadrature import fifth_order_cubature_points
from gaussx._quadrature._types import GaussianState, PropagationResult


class FifthOrderCubatureIntegrator(AbstractIntegrator):
    """Fifth-order fully symmetric cubature on 2 D_x² + 1 points.

        𝔼[g(x)] ≈ ∑ᵢ wᵢ g(χᵢ),    x ~ 𝒩(μ_x, Σ_x)

    Exact for polynomials up to total degree 5 in D_x variables. This is the
    practical middle ground between the degree-3 sigma-point rules and a
    tensor-product Gauss-Hermite rule:

        Rule                                 Points       Exact to degree
        `UnscentedIntegrator`                2 D_x + 1    3
        `FifthOrderCubatureIntegrator`       2 D_x² + 1   5
        `GaussHermiteIntegrator` (order H)   H^D_x        2 H − 1

    It is the default integrator in BayesNewton, and the lowest-degree rule
    that gets the curvature right for likelihoods whose log-density behaves
    like a quartic near the mode (Poisson, Bernoulli with heavy-tailed
    inputs).

    Note:
        The rule carries negative weights on the axis shell for D_x > 4
        — see `gaussx.fifth_order_cubature_points`. Moments through degree
        5 remain exact, but an output covariance assembled from very few
        function evaluations is not guaranteed PSD in high dimension.

    Examples:
        The fourth moment of a standard normal, which degree-3 rules miss:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> state = gaussx.GaussianState(jnp.zeros(1), covariance)
        >>> rule = FifthOrderCubatureIntegrator()
        >>> result = rule.integrate(lambda x: x**4, state)
        >>> bool(jnp.allclose(result.state.mean, jnp.array([3.0])))
        True

    References:
        McNamee & Stenger (1967).
    """

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn`` via fifth-order cubature.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            `PropagationResult` holding 𝒩(μ_y, Σ_y) and Σ_xy.
        """
        points, weights = fifth_order_cubature_points(state.mean, state.cov)
        # yᵢ = g(χᵢ)
        outputs = jax.vmap(fn)(points)
        return assemble_propagation_result(
            points,
            outputs,
            state.mean,
            weights,
            psd=self.guarantees_psd(state.mean.shape[0]),
        )

    def guarantees_psd(self, dim: int) -> bool:
        """Whether the covariance is PSD: axis weights are negative for D_x > 4.

        Args:
            dim: Input dimension D_x.

        Returns:
            ``True`` if every weight is non-negative at ``dim``.

        Examples:
            >>> FifthOrderCubatureIntegrator().guarantees_psd(4)
            True
            >>> FifthOrderCubatureIntegrator().guarantees_psd(5)
            False
        """
        return dim <= 4

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """Return the fifth-order cubature points and weights.

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            Tuple ``(points, w_m, w_c)`` with P = 2 D_x² + 1. The rule uses a
            single weight set, so ``w_m`` and ``w_c`` are the same array.

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> rule = FifthOrderCubatureIntegrator()
            >>> points, _, _ = rule.points_and_weights(state)
            >>> points.shape
            (9, 2)
        """
        points, weights = fifth_order_cubature_points(state.mean, state.cov)
        return points, weights, weights
