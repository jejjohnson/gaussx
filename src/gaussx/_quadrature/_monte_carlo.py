"""Monte Carlo integrator for uncertainty propagation."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._cholesky import cholesky
from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState, PropagationResult


class MonteCarloIntegrator(AbstractIntegrator):
    """Monte Carlo moment matching: sample, propagate, take empirical moments.

        x⁽ᵉ⁾ ~ 𝒩(μ_x, Σ_x),    y⁽ᵉ⁾ = g(x⁽ᵉ⁾),    e ∈ ℰ
        μ_y = 1 / |ℰ| ∑ₑ y⁽ᵉ⁾
        Σ_y = 1 / (|ℰ| − 1) ∑ₑ (y⁽ᵉ⁾ − μ_y)(y⁽ᵉ⁾ − μ_y)ᵀ + β I
        Σ_xy = 1 / (|ℰ| − 1) ∑ₑ (x⁽ᵉ⁾ − μ_x)(y⁽ᵉ⁾ − μ_y)ᵀ

    Algorithm:
        1. Draw the members from 𝒩(μ_x, Σ_x) through the Cholesky factor.
        2. Evaluate g on each member.
        3. Assemble the empirical moments and add β I to Σ_y.

    Attributes:
        n_samples: Number of members |ℰ|. Default ``1000``.
        regularization: Diagonal jitter β for numerical stability.
        key: PRNG key. If ``None``, uses ``jax.random.key(0)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> rule = MonteCarloIntegrator(n_samples=16, key=jr.key(0))
        >>> rule.integrate(lambda x: 2.0 * x, state).cross_cov.shape
        (2, 2)
    """

    n_samples: int = eqx.field(static=True, default=1000)
    regularization: float = eqx.field(static=True, default=1e-6)
    key: jax.Array | None = None

    def guarantees_psd(self, dim: int) -> bool:
        """Whether the covariance is PSD: a sample covariance always is.

        Args:
            dim: Input dimension D_x, unused since no weight is negative.

        Returns:
            Always ``True``.

        Examples:
            >>> MonteCarloIntegrator().guarantees_psd(2)
            True
        """
        del dim
        return True

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn`` via Monte Carlo sampling.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            `PropagationResult` holding 𝒩(μ_y, Σ_y) and Σ_xy.

        Raises:
            ValueError: If ``n_samples < 2``.
        """
        points, mean_weights, covariance_weights = self.points_and_weights(state)
        input_mean = state.mean

        # y⁽ᵉ⁾ = g(x⁽ᵉ⁾)
        outputs = jax.vmap(fn)(points)

        result = assemble_propagation_result(
            points, outputs, input_mean, mean_weights, covariance_weights
        )

        output_covariance = result.state.cov.as_matrix()
        output_size = outputs.shape[1]
        # Σ_y + β I
        output_covariance = output_covariance + self.regularization * jnp.eye(
            output_size, dtype=output_covariance.dtype
        )
        output_covariance_op = lx.MatrixLinearOperator(
            output_covariance, lx.positive_semidefinite_tag
        )
        output_state = GaussianState(mean=result.state.mean, cov=output_covariance_op)

        return PropagationResult(state=output_state, cross_cov=result.cross_cov)

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """Return the Monte Carlo samples and their weights.

            x⁽ᵉ⁾ = μ_x + L z⁽ᵉ⁾,    L Lᵀ = Σ_x,    z⁽ᵉ⁾ ~ 𝒩(0, I)

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            Tuple ``(points, w_m, w_c)`` with P = |ℰ| = ``n_samples``. Mean
            weights are 1 / |ℰ|; covariance weights carry the Bessel
            correction 1 / (|ℰ| − 1).

        Raises:
            ValueError: If ``n_samples < 2``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> rule = MonteCarloIntegrator(n_samples=8, key=jr.key(0))
            >>> points, _, _ = rule.points_and_weights(state)
            >>> points.shape
            (8, 2)
        """
        if self.n_samples < 2:
            msg = (
                f"MonteCarloIntegrator requires n_samples >= 2 for "
                f"Bessel-corrected covariance, got {self.n_samples}."
            )
            raise ValueError(msg)

        input_mean = state.mean
        state_size = input_mean.shape[0]

        key = self.key if self.key is not None else jr.key(0)

        # L Lᵀ = Σ_x
        cholesky_factor = cholesky(state.cov).as_matrix()
        # z⁽ᵉ⁾ ~ 𝒩(0, I), in the input dtype: the default would land in float64
        # under x64 and promote the whole transform
        standard_draws = jr.normal(
            key, (self.n_samples, state_size), dtype=input_mean.dtype
        )
        # x⁽ᵉ⁾ = μ_x + L z⁽ᵉ⁾
        # jnp, not einx: einx cannot resolve an empty state (D_x = 0), which
        # returns an empty sample matrix here today.
        points = input_mean[None, :] + standard_draws @ cholesky_factor.T

        ensemble_size = self.n_samples
        # w_m = 1 / |ℰ|
        mean_weights = jnp.full(
            ensemble_size, 1.0 / ensemble_size, dtype=input_mean.dtype
        )
        # w_c = 1 / (|ℰ| − 1)
        covariance_weights = jnp.full(
            ensemble_size, 1.0 / (ensemble_size - 1), dtype=input_mean.dtype
        )

        return points, mean_weights, covariance_weights
