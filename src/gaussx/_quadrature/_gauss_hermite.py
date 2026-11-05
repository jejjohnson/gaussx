"""Gauss-Hermite quadrature integrator for uncertainty propagation."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._primitives._sqrt import sqrt
from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._quadrature import gauss_hermite_points
from gaussx._quadrature._types import GaussianState, PropagationResult


class GaussHermiteIntegrator(AbstractIntegrator):
    """Gauss-Hermite integrator: a tensor-product rule on order^D_x points.

        𝔼[g(x)] ≈ ∑ᵢ wᵢ g(μ_x + S zᵢ),    S Sᵀ = Σ_x

    where (zᵢ, wᵢ) are the Gauss-Hermite points and weights in standard
    normal space. Exact for polynomials up to degree 2 order − 1.

    Cost: a tensor-product rule with P = order^D_x points, each one an
    evaluation of the integrand. At the default ``order=20`` that is 20
    points at D_x = 1, 400 at D_x = 2, 8 000 at D_x = 3 and 160 000 at
    D_x = 4. Use it for D_x ≤ 2 (or lower ``order``); for larger D_x prefer
    `CubatureIntegrator` (2 D_x points) or `FifthOrderCubatureIntegrator`
    (2 D_x² + 1 points).

    Attributes:
        order: Number of quadrature points per dimension. Default ``20``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> state = gaussx.GaussianState(jnp.zeros(1), covariance)
        >>> rule = GaussHermiteIntegrator(order=5)
        >>> result = rule.integrate(lambda x: x**2, state)
        >>> bool(jnp.allclose(result.state.mean, jnp.array([1.0])))
        True
    """

    order: int = eqx.field(static=True, default=20)

    def guarantees_psd(self, dim: int) -> bool:
        """Whether the covariance is PSD: Gauss-Hermite weights are positive.

        Args:
            dim: Input dimension D_x, unused since every weight is positive.

        Returns:
            Always ``True``.

        Examples:
            >>> GaussHermiteIntegrator().guarantees_psd(2)
            True
        """
        del dim
        return True

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn`` via Gauss-Hermite quadrature.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            `PropagationResult` holding 𝒩(μ_y, Σ_y) and Σ_xy.
        """
        points, weights, _ = self.points_and_weights(state)
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
        """Return the tensor-product Gauss-Hermite points and weights.

            χᵢ = μ_x + S zᵢ,    S Sᵀ = Σ_x

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            Tuple ``(points, w_m, w_c)`` with P = order^D_x. The rule uses a
            single weight set, so ``w_m`` and ``w_c`` are the same array,
            normalised to sum to one.

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> rule = GaussHermiteIntegrator(order=3)
            >>> points, weights, _ = rule.points_and_weights(state)
            >>> points.shape
            (9, 2)
            >>> bool(jnp.allclose(jnp.sum(weights), 1.0))
            True
        """
        input_mean = state.mean
        state_size = input_mean.shape[0]

        # (zᵢ, wᵢ) in standard normal space, in the input dtype so a float32
        # belief is not promoted under x64
        standard_points, weights = gauss_hermite_points(
            self.order, state_size, dtype=input_mean.dtype
        )

        # S Sᵀ = Σ_x
        covariance_root = sqrt(state.cov).as_matrix()
        # μ_x as one row; D_x ≥ 1 and P ≥ 1 here, since the rule rejects the rest
        mean_row = rearrange(input_mean, "state -> 1 state")
        # Sᵀ
        covariance_root_transpose = rearrange(covariance_root, "state col -> col state")
        # S zᵢ, with @ rather than einsum: the contraction changes the reverse-mode
        # summation through S by one ULP
        shifts = standard_points @ covariance_root_transpose
        # χᵢ = μ_x + S zᵢ
        points = mean_row + shifts

        # wᵢ / ∑ⱼ wⱼ
        weights = weights / jnp.sum(weights)

        return points, weights, weights
