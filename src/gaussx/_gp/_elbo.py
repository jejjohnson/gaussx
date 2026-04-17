"""Variational ELBO sugar: Gaussian and Monte Carlo ELBO objectives."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp


def variational_elbo_gaussian(
    y: jnp.ndarray,
    f_loc: jnp.ndarray,
    f_var: jnp.ndarray,
    noise_var: float,
    kl: jnp.ndarray,
) -> jnp.ndarray:
    """Titsias collapsed ELBO for Gaussian likelihoods.

    Computes::

        ELBO = E_q[log p(y|f)] - KL(q||p)

    where the expected log-likelihood under a Gaussian variational
    distribution with diagonal variance has the closed form::

        E_q[log N(y|f, sigma^2 I)]
            = -0.5 * N * log(2 pi sigma^2)
              -0.5 / sigma^2 * (||y - f_loc||^2 + sum(f_var))

    Args:
        y: Observations, shape ``(N,)``.
        f_loc: Variational mean, shape ``(N,)``.
        f_var: Variational marginal variances, shape ``(N,)``.
        noise_var: Observation noise variance (scalar).
        kl: KL divergence term ``KL(q || p)`` (scalar).

    Returns:
        Scalar ELBO value.
    """
    N = y.shape[-1]
    log_2pi = jnp.log(2.0 * jnp.pi)
    residual = y - f_loc
    ell = -0.5 * N * jnp.log(noise_var) - 0.5 * N * log_2pi
    ell = ell - 0.5 / noise_var * (jnp.sum(residual**2) + jnp.sum(f_var))
    return ell - kl


def variational_elbo_mc(
    log_likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray],
    f_samples: jnp.ndarray,
    kl: jnp.ndarray,
) -> jnp.ndarray:
    """Monte Carlo ELBO for non-conjugate likelihoods.

    Computes::

        ELBO = (1/S) sum_s log p(y|f_s) - KL(q||p)

    where ``f_s ~ q(f)`` are samples from the variational distribution.
    Supports any likelihood (Poisson, Bernoulli, Pareto, etc.).

    Args:
        log_likelihood_fn: Function mapping latent samples to scalar
            log-likelihood. Signature ``(N,) -> scalar``.
        f_samples: Samples from the variational posterior, shape
            ``(S, N)`` where S is the number of samples.
        kl: KL divergence term ``KL(q || p)`` (scalar).

    Returns:
        Scalar ELBO value.
    """
    import jax

    ell = jnp.mean(jax.vmap(log_likelihood_fn)(f_samples))
    return ell - kl
