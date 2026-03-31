"""Assumed Density Filter integrator for uncertainty propagation."""

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


class AssumedDensityFilter(AbstractIntegrator):
    r"""KL-optimal Gaussian projection via moment matching.

    Projects the (possibly non-Gaussian) output distribution onto the
    Gaussian family by matching first and second moments. Equivalent to
    ``argmin_q KL(p(y) || q(y))`` within the Gaussian family.

    Adds adaptive regularization and optional diagnostics for detecting
    non-Gaussianity::

        eps = eps_base * trace(Sigma_y) / n_dim

    Args:
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
        mu = state.mean
        Sigma = state.cov.as_matrix()
        N = mu.shape[0]

        key = self.key if self.key is not None else jr.key(0)

        # Sample from input Gaussian
        L = jnp.linalg.cholesky(Sigma)
        eps = jr.normal(key, (self.n_samples, N))
        x_samples = mu[None, :] + eps @ L.T

        # Propagate samples
        y_samples = jax.vmap(fn)(x_samples)

        # Moment matching (KL-optimal Gaussian projection)
        mu_y = jnp.mean(y_samples, axis=0)
        dy = y_samples - mu_y[None, :]
        Sigma_y = (dy.T @ dy) / (self.n_samples - 1)
        M = mu_y.shape[0]

        # Adaptive regularization
        if self.adaptive_regularization:
            eps_reg = self.regularization * jnp.trace(Sigma_y) / M
        else:
            eps_reg = self.regularization
        Sigma_y = Sigma_y + eps_reg * jnp.eye(M)
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)

        # Cross-covariance
        dx = x_samples - mu[None, :]
        cross_cov = (dx.T @ dy) / (self.n_samples - 1)

        cov_y = lx.MatrixLinearOperator(Sigma_y, lx.positive_semidefinite_tag)
        out_state = GaussianState(mean=mu_y, cov=cov_y)
        result = PropagationResult(state=out_state, cross_cov=cross_cov)

        diagnostics: dict = {}
        if compute_diagnostics:
            # Compute non-Gaussianity diagnostics
            eigvals = jnp.linalg.eigvalsh(Sigma_y)
            min_eigval = jnp.min(eigvals)
            max_eigval = jnp.max(eigvals)
            cond = max_eigval / jnp.maximum(min_eigval, 1e-30)

            # Per-dimension skewness and kurtosis
            std_dy = dy / jnp.sqrt(jnp.diag(Sigma_y))[None, :]
            skewness = jnp.mean(std_dy**3, axis=0)
            kurtosis = jnp.mean(std_dy**4, axis=0)

            diagnostics = {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "min_eigval": min_eigval,
                "condition_number": cond,
            }

        return result, diagnostics
