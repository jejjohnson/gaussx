"""Monte Carlo integrator for uncertainty propagation."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState, PropagationResult


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
        if self.n_samples < 2:
            msg = (
                f"MonteCarloIntegrator requires n_samples >= 2 for "
                f"Bessel-corrected covariance, got {self.n_samples}."
            )
            raise ValueError(msg)
        mu = state.mean
        Sigma = state.cov.as_matrix()
        N = mu.shape[0]

        key = self.key if self.key is not None else jr.key(0)

        # Sample from input Gaussian: xᵢ = μ + L εᵢ
        L = jnp.linalg.cholesky(Sigma)
        eps = jr.normal(key, (self.n_samples, N))
        chi = mu[None, :] + eps @ L.T  # (S, N)

        # Propagate samples
        Y = jax.vmap(fn)(chi)  # (S, M)

        # Uniform weights = 1/S for empirical moments
        # Use 1/(S−1) Bessel correction via covariance weights
        S = self.n_samples
        w_m = jnp.full(S, 1.0 / S)
        w_c = jnp.full(S, 1.0 / (S - 1))

        result = assemble_propagation_result(chi, Y, mu, w_m, w_c)

        # Add regularization jitter to output covariance
        Sigma_y = result.state.cov.as_matrix()
        M = Y.shape[1]
        import lineax as lx

        Sigma_y = Sigma_y + self.regularization * jnp.eye(M)
        cov_y = lx.MatrixLinearOperator(Sigma_y, lx.positive_semidefinite_tag)
        out_state = GaussianState(mean=result.state.mean, cov=cov_y)

        return PropagationResult(state=out_state, cross_cov=result.cross_cov)
