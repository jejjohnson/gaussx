"""Unscented integrator for uncertainty propagation (UKF-style)."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._uncertain._integrator import AbstractIntegrator
from gaussx._uncertain._types import GaussianState, PropagationResult


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
        mu = state.mean
        Sigma = state.cov.as_matrix()
        N = mu.shape[0]

        lam = self.alpha**2 * (N + self.kappa) - N
        c = N + lam

        # Sigma points
        S = jnp.linalg.cholesky(c * Sigma)  # (N, N)

        # chi_0 = mu, chi_{1..N} = mu + S_col, chi_{N+1..2N} = mu - S_col
        chi = jnp.concatenate(
            [mu[None, :], mu[None, :] + S.T, mu[None, :] - S.T], axis=0
        )  # (2N+1, N)

        # Weights
        w_m_0 = lam / c
        w_c_0 = lam / c + (1.0 - self.alpha**2 + self.beta)
        w_rest = 1.0 / (2.0 * c)

        w_m = jnp.concatenate([jnp.array([w_m_0]), jnp.full(2 * N, w_rest)])
        w_c = jnp.concatenate([jnp.array([w_c_0]), jnp.full(2 * N, w_rest)])

        # Propagate sigma points
        Y = jax.vmap(fn)(chi)  # (2N+1, M)

        # Output moments
        mu_y = jnp.sum(w_m[:, None] * Y, axis=0)

        dy = Y - mu_y[None, :]  # (2N+1, M)
        dx = chi - mu[None, :]  # (2N+1, N)

        # Output covariance
        Sigma_y = jnp.sum(
            w_c[:, None, None] * (dy[:, :, None] * dy[:, None, :]), axis=0
        )
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)

        # Cross-covariance
        cross_cov = jnp.sum(
            w_c[:, None, None] * (dx[:, :, None] * dy[:, None, :]), axis=0
        )

        cov_y = lx.MatrixLinearOperator(Sigma_y, lx.positive_semidefinite_tag)
        out_state = GaussianState(mean=mu_y, cov=cov_y)

        return PropagationResult(state=out_state, cross_cov=cross_cov)
