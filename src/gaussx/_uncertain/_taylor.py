"""Taylor-expansion integrator for uncertainty propagation (EKF-style)."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._uncertain._integrator import AbstractIntegrator
from gaussx._uncertain._types import GaussianState, PropagationResult


class TaylorIntegrator(AbstractIntegrator):
    r"""1st or 2nd order Taylor expansion for uncertainty propagation.

    **1st order (EKF)**::

        mu_y = f(mu_x)
        Sigma_y = J @ Sigma_x @ J^T
        cross_cov = Sigma_x @ J^T

    **2nd order**::

        mu_y_i += 0.5 * tr(H_i @ Sigma_x)
        Sigma_y += correction from Hessians

    Args:
        order: Taylor expansion order (1 or 2). Default 1.
    """

    order: int = eqx.field(static=True, default=1)

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate Gaussian via Taylor expansion."""
        if self.order not in (1, 2):
            msg = f"TaylorIntegrator.order must be 1 or 2, got {self.order}"
            raise ValueError(msg)
        mu = state.mean
        Sigma = state.cov.as_matrix()

        # Evaluate function and Jacobian at the mean
        f_mu = fn(mu)
        J = jax.jacobian(fn)(mu)  # (M, N)

        if self.order == 1:
            mu_y = f_mu
        else:
            # 2nd order correction: mu_y_i += 0.5 * tr(H_i @ Sigma_x)
            H_fn = jax.hessian(fn)
            H = H_fn(mu)  # (M, N, N)
            corrections = jax.vmap(lambda H_i: jnp.trace(H_i @ Sigma))(H)
            mu_y = f_mu + 0.5 * corrections

        # Output covariance: J @ Sigma @ J^T
        Sigma_y = J @ Sigma @ J.T

        # Symmetrize for numerical stability
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)

        # Cross-covariance: Sigma_x @ J^T
        cross_cov = Sigma @ J.T

        cov_y = lx.MatrixLinearOperator(Sigma_y, lx.positive_semidefinite_tag)
        out_state = GaussianState(mean=mu_y, cov=cov_y)

        return PropagationResult(state=out_state, cross_cov=cross_cov)
