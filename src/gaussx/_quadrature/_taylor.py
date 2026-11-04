"""Taylor-expansion integrator for uncertainty propagation (EKF-style)."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._symmetrize import symmetrize
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState, PropagationResult


class TaylorIntegrator(AbstractIntegrator):
    """First- or second-order Taylor expansion of g about the mean.

    First order (EKF):

        μ_y = g(μ_x)
        Σ_y = J_g Σ_x J_gᵀ
        Σ_xy = Σ_x J_gᵀ

    Second order, with H_g,m = ∂²g_m / ∂x² the Hessian of output m:

        μ_y,m = g_m(μ_x) + ½ tr(H_g,m Σ_x)
        Σ_y,mn = [J_g Σ_x J_gᵀ]_mn + ½ tr(H_g,m Σ_x H_g,n Σ_x)

    Attributes:
        order: Taylor expansion order (1 or 2). Default 1.
        correct_variance: If True and order=2, apply 2nd-order covariance
            correction using 4th Gaussian moments. Default True to preserve
            the historical ``order=2`` behaviour. Set to False for the
            mean-only correction used in the standard EKF literature.
            Ignored when order=1.

    Examples:
        The mean correction recovers 𝔼[x²] = μ² + σ²:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.array([2.0]))
        >>> state = gaussx.GaussianState(jnp.array([1.0]), covariance)
        >>> result = TaylorIntegrator(order=2).integrate(lambda x: x**2, state)
        >>> bool(jnp.allclose(result.state.mean, jnp.array([3.0])))
        True
    """

    order: int = eqx.field(static=True, default=1)
    correct_variance: bool = eqx.field(static=True, default=True)

    def guarantees_psd(self, dim: int) -> bool:
        """Whether the covariance is PSD: a linearisation is a consistent joint.

        J_g Σ_x J_gᵀ plus a PSD second-order correction.

        Args:
            dim: Input dimension D_x, unused.

        Returns:
            Always ``True``.

        Examples:
            >>> TaylorIntegrator().guarantees_psd(2)
            True
        """
        del dim
        return True

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn`` via a Taylor expansion.

        Algorithm:
            1. Evaluate g and J_g at μ_x.
            2. Linearise: Σ_y = J_g Σ_x J_gᵀ.
            3. At second order, add the Hessian mean correction and, if
               ``correct_variance``, the Hessian covariance correction.
            4. Symmetrise Σ_y and take Σ_xy = Σ_x J_gᵀ.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            `PropagationResult` holding 𝒩(μ_y, Σ_y) and Σ_xy.

        Raises:
            ValueError: If ``order`` is not 1 or 2.

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> result = TaylorIntegrator().integrate(lambda x: 2.0 * x, state)
            >>> bool(jnp.allclose(result.cross_cov, 2.0 * jnp.eye(2)))
            True
        """
        if self.order not in (1, 2):
            msg = f"TaylorIntegrator.order must be 1 or 2, got {self.order}"
            raise ValueError(msg)
        input_mean = state.mean
        input_covariance = state.cov.as_matrix()

        # g(μ_x)
        output_at_mean = fn(input_mean)
        # J_g = ∂g/∂x at μ_x
        jacobian = jax.jacobian(fn)(input_mean)

        # jnp, not einx, for every product below: einx cannot resolve an empty
        # input or output (D_x = 0 or D_y = 0), which propagates here today.
        if self.order == 1:
            # μ_y = g(μ_x)
            output_mean = output_at_mean
            # Σ_y = J_g Σ_x J_gᵀ
            output_covariance = jacobian @ input_covariance @ jacobian.T
        else:
            # H_g,m = ∂²g_m / ∂x²
            hessian = jax.hessian(fn)(input_mean)

            def _mean_correction(
                hessian_m: Float[Array, "N N"],
            ) -> Float[Array, ""]:
                # tr(H_g,m Σ_x)
                return jnp.trace(hessian_m @ input_covariance)

            corrections = jax.vmap(_mean_correction)(hessian)
            # μ_y,m = g_m(μ_x) + ½ tr(H_g,m Σ_x)
            output_mean = output_at_mean + 0.5 * corrections
            # J_g Σ_x J_gᵀ
            output_covariance = jacobian @ input_covariance @ jacobian.T
            if self.correct_variance:

                def _covariance_correction(
                    hessian_m: Float[Array, "N N"],
                    hessian_n: Float[Array, "N N"],
                ) -> Float[Array, ""]:
                    # ½ tr(H_g,m Σ_x H_g,n Σ_x)
                    product = (
                        hessian_m @ input_covariance @ hessian_n @ input_covariance
                    )
                    return 0.5 * jnp.trace(product)

                second_order_covariance = jax.vmap(
                    lambda hessian_m: jax.vmap(
                        lambda hessian_n: _covariance_correction(hessian_m, hessian_n)
                    )(hessian)
                )(hessian)
                # Σ_y,mn = [J_g Σ_x J_gᵀ]_mn + ½ tr(H_g,m Σ_x H_g,n Σ_x)
                output_covariance = output_covariance + second_order_covariance

        # Symmetrised against round-off
        output_covariance = symmetrize(output_covariance)

        # Σ_xy = Σ_x J_gᵀ
        cross_covariance = input_covariance @ jacobian.T

        output_covariance_op = lx.MatrixLinearOperator(
            output_covariance, lx.positive_semidefinite_tag
        )
        output_state = GaussianState(mean=output_mean, cov=output_covariance_op)

        return PropagationResult(state=output_state, cross_cov=cross_covariance)
