"""Assumed density filter integrator for uncertainty propagation."""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import reduce
from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._eig import eigvals as _gaussx_eigvals
from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._monte_carlo import MonteCarloIntegrator
from gaussx._quadrature._types import GaussianState, PropagationResult


class AssumedDensityFilter(AbstractIntegrator):
    """KL-optimal Gaussian projection of g(x) via moment matching.

        q⋆ = argmin_q KL(p(y) ‖ q(y)),    q Gaussian
        Σ_y,reg = Σ_y + β I,    β = β₀ tr(Σ_y) / D_y

    Projects the (possibly non-Gaussian) output distribution onto the
    Gaussian family by matching first and second moments, with adaptive
    regularization β and optional diagnostics for detecting non-Gaussianity.

    Algorithm:
        1. Draw the `MonteCarloIntegrator` samples and evaluate g on them.
        2. Assemble μ_y, Σ_y and Σ_xy from the Bessel-corrected weights.
        3. Regularise and symmetrise Σ_y.
        4. Optionally compute skewness, kurtosis and the spectrum of Σ_y.

    Attributes:
        n_samples: Number of Monte Carlo samples |ℰ|. Default ``5000``.
        regularization: Base regularization β₀. Default ``1e-6``.
        adaptive_regularization: Scale regularization by output
            variance. Default ``True``.
        key: PRNG key. If ``None``, uses ``jax.random.key(0)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> rule = AssumedDensityFilter(n_samples=16, key=jr.key(0))
        >>> rule.integrate(lambda x: 2.0 * x, state).state.mean.shape
        (2,)
    """

    n_samples: int = eqx.field(static=True, default=5000)
    regularization: float = eqx.field(static=True, default=1e-6)
    adaptive_regularization: bool = eqx.field(static=True, default=True)
    key: jax.Array | None = None

    def guarantees_psd(self, dim: int) -> bool:
        """Whether the covariance is PSD: a regularised sample covariance is.

        Args:
            dim: Input dimension D_x, unused since no weight is negative.

        Returns:
            Always ``True``.

        Examples:
            >>> AssumedDensityFilter().guarantees_psd(2)
            True
        """
        del dim
        return True

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """The `MonteCarloIntegrator` samples and weights for this rule.

        Lets `gaussx.moment_match` / `gaussx.statistical_linear_regression`
        use the same samples ``integrate`` does.

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            Tuple ``(points, w_m, w_c)`` as
            `MonteCarloIntegrator.points_and_weights`.

        Raises:
            ValueError: If ``n_samples < 2``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> rule = AssumedDensityFilter(n_samples=8, key=jr.key(0))
            >>> rule.points_and_weights(state)[0].shape
            (8, 2)
        """
        return MonteCarloIntegrator(
            n_samples=self.n_samples, key=self.key
        ).points_and_weights(state)

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn`` via assumed density filtering.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            `PropagationResult` holding 𝒩(μ_y, Σ_y,reg) and Σ_xy.

        Raises:
            ValueError: If ``n_samples < 2``.
        """
        result, _ = self._integrate_impl(fn, state)
        return result

    def integrate_with_diagnostics(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> tuple[PropagationResult, dict]:
        """Propagate a Gaussian and return non-Gaussianity diagnostics.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            Tuple ``(result, diagnostics)`` where diagnostics contains
            ``skewness``, ``kurtosis``, ``min_eigval``, and
            ``condition_number``.

        Raises:
            ValueError: If ``n_samples < 2``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> rule = AssumedDensityFilter(n_samples=16, key=jr.key(0))
            >>> _, diagnostics = rule.integrate_with_diagnostics(lambda x: x, state)
            >>> diagnostics["skewness"].shape
            (2,)
        """
        return self._integrate_impl(fn, state, compute_diagnostics=True)

    def _integrate_impl(
        self,
        fn: Callable,
        state: GaussianState,
        compute_diagnostics: bool = False,
    ) -> tuple[PropagationResult, dict]:
        """Regularised moments of g(x), with optional diagnostics.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).
            compute_diagnostics: Also compute the non-Gaussianity diagnostics.

        Returns:
            Tuple ``(result, diagnostics)``; ``diagnostics`` is empty unless
            requested.
        """
        # Same samples and Bessel-corrected moments as MonteCarloIntegrator
        # (shared points_and_weights + assemble_propagation_result).
        points, mean_weights, covariance_weights = self.points_and_weights(state)
        # y⁽ᵉ⁾ = g(x⁽ᵉ⁾)
        outputs = jax.vmap(fn)(points)
        moments = assemble_propagation_result(
            points, outputs, state.mean, mean_weights, covariance_weights
        )
        output_mean = moments.state.mean
        output_covariance = moments.state.cov.as_matrix()
        output_size = output_mean.shape[0]

        if self.adaptive_regularization:
            # β = β₀ tr(Σ_y) / D_y
            jitter = self.regularization * jnp.trace(output_covariance) / output_size
        else:
            # β = β₀
            jitter = self.regularization
        # Σ_y,reg = Σ_y + β I
        output_covariance = symmetrize(
            output_covariance
            + jitter * jnp.eye(output_size, dtype=output_covariance.dtype)
        )

        output_covariance_op = lx.MatrixLinearOperator(
            output_covariance, lx.positive_semidefinite_tag
        )
        output_state = GaussianState(mean=output_mean, cov=output_covariance_op)
        result = PropagationResult(state=output_state, cross_cov=moments.cross_cov)

        diagnostics: dict = {}
        if compute_diagnostics:
            # --- non-Gaussianity diagnostics ---
            # Through the gaussx primitive, which dispatches on structure.
            eigenvalues = _gaussx_eigvals(output_covariance_op)
            min_eigenvalue = jnp.min(eigenvalues)
            max_eigenvalue = jnp.max(eigenvalues)
            # cond(Σ_y) = γ_max / max(γ_min, 10⁻³⁰)
            condition_number = max_eigenvalue / jnp.maximum(min_eigenvalue, 1e-30)

            # y⁽ᵉ⁾ − μ_y
            anomalies = einx.subtract(
                "member out, out -> member out", outputs, output_mean
            )
            # (y⁽ᵉ⁾ − μ_y) / σ_y, per output
            standardised = einx.divide(
                "member out, out -> member out",
                anomalies,
                jnp.sqrt(jnp.diag(output_covariance)),
            )
            # third standardised moment
            skewness = reduce(standardised**3, "member out -> out", "mean")
            # fourth standardised moment
            kurtosis = reduce(standardised**4, "member out -> out", "mean")

            diagnostics = {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "min_eigval": min_eigenvalue,
                "condition_number": condition_number,
            }

        return result, diagnostics
