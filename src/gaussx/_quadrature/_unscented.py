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
    r"""Unscented transform: deterministic sigma points.

    Generates ``2N+1`` sigma points around the mean, propagates them
    through the nonlinear function, and reconstructs output moments::

        chi_i = mu + sqrt((N + lambda) * Sigma) @ xi_i
        y_i = f(chi_i)
        mu_y = sum(w_m * y_i)
        Sigma_y = sum(w_c * (y_i - mu_y)(y_i - mu_y)^T)
        cross_cov = sum(w_c * (chi_i - mu)(y_i - mu_y)^T)

    where ``lambda = alpha^2 * (N + kappa) - N``.

    Args:
        alpha: Spread parameter. Default ``1e-3``.
        beta: Prior knowledge parameter (2.0 optimal for Gaussian).
        kappa: Secondary scaling. Default ``0.0``.
    """

    alpha: float = eqx.field(static=True, default=1e-3)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate Gaussian via unscented transform."""
        chi, w_m, w_c = sigma_points(
            state.mean,
            state.cov,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
        Y = jax.vmap(fn)(chi)
        return assemble_propagation_result(chi, Y, state.mean, w_m, w_c)
