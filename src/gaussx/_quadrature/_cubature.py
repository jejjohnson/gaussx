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
    """Third-order spherical-radial cubature on 2 D_x equally weighted points.

        𝔼[g(x)] ≈ 1 / (2 D_x) ∑ᵢ g(χᵢ),    x ~ 𝒩(μ_x, Σ_x)

    Exact for polynomials up to total degree 3. Driving
    `gaussx.nonlinear_kalman_filter` with it gives the textbook cubature
    Kalman filter (CKF).

    It is the degree-3 sibling of `gaussx.FifthOrderCubatureIntegrator`,
    and closely related to `gaussx.UnscentedIntegrator`: the unscented
    transform with ``alpha=1``, ``beta=0``, ``kappa=0`` places its 2 D_x
    non-centre points identically, differing only by the centre point and
    its weight. Unlike the scaled unscented transform, every weight here is
    positive, 1 / (2 D_x), so an assembled covariance cannot be dragged
    indefinite by a negative weight.

    Examples:
        The rule is exact for an affine map:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0]))
        >>> state = gaussx.GaussianState(jnp.ones(2), covariance)
        >>> result = CubatureIntegrator().integrate(lambda x: 3.0 * x, state)
        >>> bool(jnp.allclose(result.state.mean, jnp.array([3.0, 3.0])))
        True

    References:
        Arasaratnam & Haykin (2009).
    """

    def guarantees_psd(self, dim: int) -> bool:
        """Whether the covariance is PSD: all weights are 1 / (2 D_x).

        Args:
            dim: Input dimension D_x, unused since every weight is positive.

        Returns:
            Always ``True``.

        Examples:
            >>> CubatureIntegrator().guarantees_psd(3)
            True
        """
        del dim
        return True

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn`` via third-order cubature.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            `PropagationResult` holding 𝒩(μ_y, Σ_y) and Σ_xy.
        """
        points, weights = cubature_points(state.mean, state.cov)
        # yᵢ = g(χᵢ)
        outputs = jax.vmap(fn)(points)
        return assemble_propagation_result(
            points,
            outputs,
            state.mean,
            weights,
            psd=self.guarantees_psd(state.mean.shape[0]),
        )

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """Return the third-order cubature points and weights.

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            Tuple ``(points, w_m, w_c)`` with P = 2 D_x. The rule uses a
            single weight set, so ``w_m`` and ``w_c`` are the same array.

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> points, _, _ = CubatureIntegrator().points_and_weights(state)
            >>> points.shape
            (4, 2)
        """
        points, weights = cubature_points(state.mean, state.cov)
        return points, weights, weights
