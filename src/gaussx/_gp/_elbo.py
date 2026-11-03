"""Variational ELBOs: the Gaussian closed form and the Monte Carlo estimate.

Both return the evidence lower bound itself (the quantity to maximise, so
ℒ_ELBO = −ELBO is the loss), with the KL term KL[q(f_ℐ) ‖ p(f_ℐ)] of the
variational distribution over the inducing values supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._distributions._gaussian import _LOG_2PI


def variational_elbo_gaussian(
    y: Float[Array, " N"],
    f_loc: Float[Array, " N"],
    f_var: Float[Array, " N"],
    noise_var: float,
    kl: Float[Array, ""],
) -> Float[Array, ""]:
    """Uncollapsed (SVGP) ELBO for a Gaussian likelihood.

        ELBO = ∑_n 𝔼_q(f_n)[log 𝒩(y_n; f_n, σ_ε²)] − KL[q(f_ℐ) ‖ p(f_ℐ)]
             = −(N/2) log(2π σ_ε²)
               − (1 / (2σ_ε²)) ∑_n ((y_n − μ_n)² + σ²_n) − KL[q(f_ℐ) ‖ p(f_ℐ)]

    This is the stochastic-variational bound of Hensman et al. (2013): the
    variational distribution q(f_ℐ) stays explicit, so the caller supplies
    the per-input marginals q(f_n) = 𝒩(μ_n, σ²_n) and the KL term.

    Algorithm:
        1. Form the residuals y_n − μ_n.
        2. Add the normaliser −(N/2) log(2π σ_ε²) to the expected squared
           error, ∑_n ((y_n − μ_n)² + σ²_n), scaled by −1 / (2σ_ε²).
        3. Subtract the KL term.

    Args:
        y: Observations y_n.
        f_loc: Means μ_n of the marginals q(f_n).
        f_var: Variances σ²_n of the marginals q(f_n).
        noise_var: Observation noise variance σ_ε².
        kl: KL[q(f_ℐ) ‖ p(f_ℐ)], e.g. from `gaussx.gauss_kl`.

    Returns:
        The scalar ELBO.

    Examples:
        Zero-variance marginals at the data give the Gaussian normaliser:

        >>> import math
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> y = jnp.array([0.5, -1.0, 2.0])
        >>> elbo = gaussx.variational_elbo_gaussian(
        ...     y, y, jnp.zeros(3), 1.0, jnp.array(0.0)
        ... )
        >>> bool(jnp.allclose(elbo, -1.5 * math.log(2 * math.pi)))
        True

        Marginal variances and the KL term lower the bound:

        >>> lower = gaussx.variational_elbo_gaussian(
        ...     y, y, jnp.ones(3), 1.0, jnp.array(0.5)
        ... )
        >>> round(float(elbo - lower), 4)
        2.0

    See Also:
        `collapsed_elbo`: The Titsias (2009) collapsed bound, with the
        optimal q(f_ℐ) integrated out analytically.

    References:
        Hensman, J., Fusi, N. and Lawrence, N. D. (2013). Gaussian
        processes for big data. *UAI*.

        Titsias, M. (2009). Variational learning of inducing variables in
        sparse Gaussian processes. *AISTATS*.
    """
    num_samples = y.shape[-1]
    # y_n − μ_n
    residual = y - f_loc
    # log σ_ε²
    log_noise_variance = jnp.log(noise_var)
    # −(N/2) log σ_ε² − (N/2) log 2π
    log_normaliser = (
        -0.5 * num_samples * log_noise_variance - 0.5 * num_samples * _LOG_2PI
    )
    # 1 / (2σ_ε²)
    half_noise_precision = 0.5 / noise_var
    # (y_n − μ_n)²
    squared_residual = residual**2
    # ∑_n (y_n − μ_n)²
    residual_norm = jnp.sum(squared_residual)
    # ∑_n σ²_n
    total_variance = jnp.sum(f_var)
    # ∑_n ((y_n − μ_n)² + σ²_n)
    expected_squared_error = residual_norm + total_variance
    # ∑_n 𝔼_q(f_n)[log 𝒩(y_n; f_n, σ_ε²)]
    expected_log_likelihood = (
        log_normaliser - half_noise_precision * expected_squared_error
    )
    # ELBO = ∑_n 𝔼_q(f_n)[log p(y_n ∣ f_n)] − KL[q(f_ℐ) ‖ p(f_ℐ)]
    return expected_log_likelihood - kl


def variational_elbo_mc(
    log_likelihood_fn: Callable[[Float[Array, " N"]], Float[Array, ""]],
    f_samples: Float[Array, "S N"],
    kl: Float[Array, ""],
) -> Float[Array, ""]:
    """Monte Carlo ELBO for non-conjugate likelihoods.

        ELBO ≈ (1/S) ∑_e log p(y ∣ f^(e)) − KL[q(f_ℐ) ‖ p(f_ℐ)],
        f^(e) ~ q(f), e = 1, …, S

    Any likelihood works (Poisson, Bernoulli, Pareto, …): it enters only
    through ``log_likelihood_fn``, evaluated once per draw.

    Algorithm:
        1. Evaluate log p(y ∣ f^(e)) for every draw, under `jax.vmap`.
        2. Average over the S draws and subtract the KL term.

    Args:
        log_likelihood_fn: Map from one draw f^(e) of the latent values to
            the scalar log-likelihood log p(y ∣ f^(e)).
        f_samples: The S draws f^(e) ~ q(f), one per row.
        kl: KL[q(f_ℐ) ‖ p(f_ℐ)], e.g. from `gaussx.gauss_kl`.

    Returns:
        The scalar Monte Carlo ELBO.

    Examples:
        A Gaussian log-likelihood averaged over two draws:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> y = jnp.array([0.0, 1.0])
        >>> def log_likelihood(f):
        ...     return -0.5 * jnp.sum((y - f) ** 2)
        >>> draws = jnp.array([[0.0, 1.0], [1.0, 1.0]])
        >>> elbo = gaussx.variational_elbo_mc(log_likelihood, draws, jnp.array(0.1))
        >>> round(float(elbo), 4)
        -0.35

        It works under `jax.jit` with the likelihood static:

        >>> import jax
        >>> jitted = jax.jit(gaussx.variational_elbo_mc, static_argnums=0)
        >>> bool(jnp.allclose(jitted(log_likelihood, draws, jnp.array(0.1)), elbo))
        True
    """
    # log p(y ∣ f^(e)), one per draw
    log_likelihoods = jax.vmap(log_likelihood_fn)(f_samples)
    # (1/S) ∑_e log p(y ∣ f^(e))
    expected_log_likelihood = jnp.mean(log_likelihoods)
    # ELBO ≈ (1/S) ∑_e log p(y ∣ f^(e)) − KL[q(f_ℐ) ‖ p(f_ℐ)]
    return expected_log_likelihood - kl
