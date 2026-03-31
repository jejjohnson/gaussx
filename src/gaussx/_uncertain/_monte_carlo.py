"""Monte Carlo integrator for uncertainty propagation."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._uncertain._integrator import AbstractIntegrator
from gaussx._uncertain._types import GaussianState, PropagationResult


class MonteCarloIntegrator(AbstractIntegrator):
    r"""Monte Carlo moment matching: sample, propagate, compute moments.

    Propagates uncertainty by drawing samples from the input Gaussian,
    evaluating the function at each sample, and computing empirical
    output moments::

        x_i ~ N(mu, Sigma)       (n_samples points)
        y_i = f(x_i)
        mu_y = mean(y_i)
        Sigma_y = cov(y_i) + regularization * I
        cross_cov = cov(x_i, y_i)

    Args:
        n_samples: Number of Monte Carlo samples. Default ``1000``.
        regularization: Diagonal jitter for numerical stability.
        key: PRNG key. If ``None``, uses ``jax.random.key(0)``.
    """

    n_samples: int = eqx.field(static=True, default=1000)
    regularization: float = eqx.field(static=True, default=1e-6)
    key: jax.Array | None = None

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate Gaussian via Monte Carlo sampling."""
        mu = state.mean
        Sigma = state.cov.as_matrix()
        N = mu.shape[0]

        key = self.key if self.key is not None else jr.key(0)

        # Sample from input Gaussian
        L = jnp.linalg.cholesky(Sigma)
        eps = jr.normal(key, (self.n_samples, N))
        x_samples = mu[None, :] + eps @ L.T  # (S, N)

        # Propagate samples
        y_samples = jax.vmap(fn)(x_samples)  # (S, M)

        # Empirical output moments
        mu_y = jnp.mean(y_samples, axis=0)
        dy = y_samples - mu_y[None, :]
        Sigma_y = (dy.T @ dy) / (self.n_samples - 1)

        # Regularize
        M = mu_y.shape[0]
        Sigma_y = Sigma_y + self.regularization * jnp.eye(M)
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)

        # Cross-covariance
        dx = x_samples - mu[None, :]
        cross_cov = (dx.T @ dy) / (self.n_samples - 1)

        cov_y = lx.MatrixLinearOperator(Sigma_y, lx.positive_semidefinite_tag)
        out_state = GaussianState(mean=mu_y, cov=cov_y)

        return PropagationResult(state=out_state, cross_cov=cross_cov)
