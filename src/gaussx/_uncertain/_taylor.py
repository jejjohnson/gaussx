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
        correct_variance: If True and order=2, apply 2nd-order covariance
            correction using 4th Gaussian moments. Default True to preserve
            the historical ``order=2`` behaviour. Set to False for the
            mean-only correction used in the standard EKF literature.
            Ignored when order=1.
    """

    order: int = eqx.field(static=True, default=1)
    correct_variance: bool = eqx.field(static=True, default=True)

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
            Sigma_y = J @ Sigma @ J.T
        else:
            # 2nd order correction: mu_y_i += 0.5 * tr(H_i @ Sigma_x)
            H_fn = jax.hessian(fn)
            H = H_fn(mu)  # (M, N, N)
            corrections = jax.vmap(lambda H_i: jnp.trace(H_i @ Sigma))(H)
            mu_y = f_mu + 0.5 * corrections
            Sigma_y = J @ Sigma @ J.T
            if self.correct_variance:
                second_order_cov = jax.vmap(
                    lambda H_i: jax.vmap(
                        lambda H_j: 0.5 * jnp.trace(H_i @ Sigma @ H_j @ Sigma)
                    )(H)
                )(H)
                Sigma_y = Sigma_y + second_order_cov

        # Symmetrize for numerical stability
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)

        # Cross-covariance: Sigma_x @ J^T
        cross_cov = Sigma @ J.T

        cov_y = lx.MatrixLinearOperator(Sigma_y, lx.positive_semidefinite_tag)
        out_state = GaussianState(mean=mu_y, cov=cov_y)

        return PropagationResult(state=out_state, cross_cov=cross_cov)
