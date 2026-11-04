"""Unscented integrator for uncertainty propagation (UKF-style)."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float

from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._quadrature import sigma_points
from gaussx._quadrature._types import GaussianState, PropagationResult


class UnscentedIntegrator(AbstractIntegrator):
    """Unscented transform: 2 D_x + 1 deterministic sigma points.

        yᵢ = g(χᵢ)
        μ_y = ∑ᵢ w_m,i yᵢ
        Σ_y = ∑ᵢ w_c,i (yᵢ − μ_y)(yᵢ − μ_y)ᵀ
        Σ_xy = ∑ᵢ w_c,i (χᵢ − μ_x)(yᵢ − μ_y)ᵀ

    with the points and weights of `gaussx.sigma_points`, whose spread is set
    by λ = α² (D_x + κ) − D_x.

    Warning:
        The default ``alpha=1e-3`` (the classic Wan-van der Merwe scaled
        transform) puts a centre mean weight of about ``-1e6`` on the
        points and recovers the moments by cancellation, losing about six
        digits: in float32 a smooth nonlinearity is wrong in the second
        digit (gh-310). Prefer ``UnscentedIntegrator(alpha=1.0)``, the
        symmetric 2 D_x + 1 rule with a zero centre mean weight (exact for
        affine maps and positive covariance weights), which is what
        `moment_transform` and the nonlinear Kalman filters use by
        default. A `UserWarning` is emitted when the centre
        mean weight reaches magnitude ``1e3`` with a float32 state.

    Attributes:
        alpha: Spread parameter α. Default ``1e-3``; ``1.0`` recommended.
        beta: Prior knowledge parameter β (2.0 optimal for a Gaussian).
        kappa: Secondary scaling κ. Default ``0.0``.

    Examples:
        The rule is exact for an affine map:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0]))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> rule = UnscentedIntegrator(alpha=1.0)
        >>> result = rule.integrate(lambda x: 2.0 * x, state)
        >>> expected = jnp.diag(jnp.array([4.0, 16.0]))
        >>> bool(jnp.allclose(result.state.cov.as_matrix(), expected))
        True
    """

    alpha: float = eqx.field(static=True, default=1e-3)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn`` via the unscented transform.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            `PropagationResult` holding 𝒩(μ_y, Σ_y) and Σ_xy.
        """
        points, mean_weights, covariance_weights = self.points_and_weights(state)
        # yᵢ = g(χᵢ)
        outputs = jax.vmap(fn)(points)
        return assemble_propagation_result(
            points,
            outputs,
            state.mean,
            mean_weights,
            covariance_weights,
            psd=self.guarantees_psd(state.mean.shape[0]),
        )

    def guarantees_psd(self, dim: int) -> bool:
        """Whether every scaled-unscented covariance weight is non-negative.

            w_c,0 = λ / (D_x + λ) + 1 − α² + β,    w_c,i = 1 / (2 (D_x + λ))

        with λ = α² (D_x + κ) − D_x, so small α (the ``1e-3`` default) makes
        the centre weight large and negative; α = 1 does not.

        Args:
            dim: Input dimension D_x the rule is applied at.

        Returns:
            ``True`` if every covariance weight is non-negative at ``dim``.

        Examples:
            >>> UnscentedIntegrator(alpha=1.0).guarantees_psd(3)
            True
            >>> UnscentedIntegrator().guarantees_psd(3)
            False
        """
        # λ = α² (D_x + κ) − D_x
        scaling = self.alpha**2 * (dim + self.kappa) - dim
        # D_x + λ
        spread = dim + scaling
        if spread <= 0:
            return False
        # w_c,0 = λ / (D_x + λ) + 1 − α² + β
        return scaling / spread + 1.0 - self.alpha**2 + self.beta >= 0

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """Return the scaled unscented sigma points and weights.

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            Tuple ``(points, w_m, w_c)`` with P = 2 D_x + 1. Mean and
            covariance weights differ in the centre point.

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> points, _, _ = UnscentedIntegrator(alpha=1.0).points_and_weights(state)
            >>> points.shape
            (5, 2)
        """
        return sigma_points(
            state.mean,
            state.cov,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
