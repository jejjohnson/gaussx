"""Assumed Density Filter integrator for uncertainty propagation."""

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
    r"""KL-optimal Gaussian projection via moment matching.

    Projects the (possibly non-Gaussian) output distribution onto the
    Gaussian family by matching first and second moments. Equivalent to
    ``argmin_q KL(p(y) || q(y))`` within the Gaussian family.

    Adds adaptive regularization and optional diagnostics for detecting
    non-Gaussianity:

        eps = eps_base * trace(Sigma_y) / n_dim

    Attributes:
        n_samples: Number of Monte Carlo samples. Default ``5000``.
        regularization: Base regularization. Default ``1e-6``.
        adaptive_regularization: Scale regularization by output
            variance. Default ``True``.
        key: PRNG key. If ``None``, uses ``jax.random.key(0)``.
    """

    n_samples: int = eqx.field(static=True, default=5000)
    regularization: float = eqx.field(static=True, default=1e-6)
    adaptive_regularization: bool = eqx.field(static=True, default=True)
    key: jax.Array | None = None

    def guarantees_psd(self, dim: int) -> bool:
        """A regularised sample covariance is PSD."""
        del dim
        return True

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """The `MonteCarloIntegrator` samples and weights for this rule.

        Lets `gaussx.moment_match` / `gaussx.statistical_linear_regression`
        use the same samples ``integrate`` does.

        Raises:
            ValueError: If ``n_samples < 2``.
        """
        return MonteCarloIntegrator(
            n_samples=self.n_samples, key=self.key
        ).points_and_weights(state)

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate Gaussian via assumed density filtering."""
        result, _ = self._integrate_impl(fn, state)
        return result

    def integrate_with_diagnostics(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> tuple[PropagationResult, dict]:
        """Propagate Gaussian and return non-Gaussianity diagnostics.

        Args:
            fn: Nonlinear function mapping ``(N,) -> (M,)``.
            state: Input Gaussian distribution.

        Returns:
            Tuple ``(result, diagnostics)`` where diagnostics contains
            ``skewness``, ``kurtosis``, ``min_eigval``, and
            ``condition_number``.
        """
        return self._integrate_impl(fn, state, compute_diagnostics=True)

    def _integrate_impl(
        self,
        fn: Callable,
        state: GaussianState,
        compute_diagnostics: bool = False,
    ) -> tuple[PropagationResult, dict]:
        """Core implementation with optional diagnostics."""
        # Same samples and Bessel-corrected moments as MonteCarloIntegrator
        # (shared points_and_weights + assemble_propagation_result).
        chi, w_m, w_c = self.points_and_weights(state)
        y_samples = jax.vmap(fn)(chi)
        moments = assemble_propagation_result(chi, y_samples, state.mean, w_m, w_c)
        mu_y = moments.state.mean
        Sigma_y = moments.state.cov.as_matrix()
        M = mu_y.shape[0]

        # Adaptive regularization
        if self.adaptive_regularization:
            eps_reg = self.regularization * jnp.trace(Sigma_y) / M
        else:
            eps_reg = self.regularization
        Sigma_y = symmetrize(Sigma_y + eps_reg * jnp.eye(M, dtype=Sigma_y.dtype))

        cov_y = lx.MatrixLinearOperator(Sigma_y, lx.positive_semidefinite_tag)
        out_state = GaussianState(mean=mu_y, cov=cov_y)
        result = PropagationResult(state=out_state, cross_cov=moments.cross_cov)

        diagnostics: dict = {}
        if compute_diagnostics:
            # Compute non-Gaussianity diagnostics via the gaussx
            # eigvals primitive (dispatches on operator structure).
            eigvals = _gaussx_eigvals(cov_y)
            min_eigval = jnp.min(eigvals)
            max_eigval = jnp.max(eigvals)
            cond = max_eigval / jnp.maximum(min_eigval, 1e-30)

            # Per-dimension skewness and kurtosis
            dy = einx.subtract("s m, m -> s m", y_samples, mu_y)
            std_dy = einx.divide("s m, m -> s m", dy, jnp.sqrt(jnp.diag(Sigma_y)))
            skewness = reduce(std_dy**3, "s m -> m", "mean")
            kurtosis = reduce(std_dy**4, "s m -> m", "mean")

            diagnostics = {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "min_eigval": min_eigval,
                "condition_number": cond,
            }

        return result, diagnostics
