"""KL divergence between Gaussian distributions."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsla
from jaxtyping import Array, Float


def gauss_kl(
    q_mu: Float[Array, "M R"],
    q_sqrt: Float[Array, "R M M"] | Float[Array, "M R"],
    K: Float[Array, "M M"] | None = None,
) -> Float[Array, ""]:
    r"""KL divergence ``KL[q(u) || p(u)]`` between Gaussian distributions.

    Computes the KL divergence where:

    - ``q = N(q_mu, q_sqrt @ q_sqrt^T)``
    - ``p = N(0, K)`` or ``p = N(0, I)`` if ``K is None`` (white prior)

    Handles both full and diagonal ``q_sqrt``:

    - **Full** ``q_sqrt``: shape ``(R, M, M)`` — lower-triangular Cholesky
      factors of the variational covariance per output dimension.
    - **Diagonal** ``q_sqrt``: shape ``(M, R)`` — diagonal standard
      deviations.

    Args:
        q_mu: Variational mean, shape ``(M, R)``.
        q_sqrt: Variational Cholesky factor or diagonal std devs.
        K: Prior covariance matrix, shape ``(M, M)``.
            If ``None``, uses white prior (identity).

    Returns:
        Scalar KL divergence summed over all ``R`` output dimensions.
    """
    M = q_mu.shape[0]
    R = q_mu.shape[1]

    is_diagonal = q_sqrt.ndim == 2

    # Prior Cholesky factor
    L_K = jnp.linalg.cholesky(K) if K is not None else None

    kl = jnp.array(0.0)

    if is_diagonal:
        # q_sqrt shape: (M, R) — diagonal standard deviations
        q_var = q_sqrt**2  # (M, R)

        if L_K is not None:
            # Solve L_K^{-1} q_mu for the Mahalanobis term
            # alpha = L_K^{-1} q_mu => K^{-1} q_mu = L_K^{-T} alpha
            alpha = jsla.solve_triangular(L_K, q_mu, lower=True)  # (M, R)
            mahal = jnp.sum(alpha**2)

            # tr(K^{-1} S) = tr(L_K^{-T} L_K^{-1} diag(q_var))
            #              = sum_r sum_i q_var[i,r] * ||L_K^{-1}[:,i]||^2
            L_K_inv = jsla.solve_triangular(L_K, jnp.eye(M), lower=True)  # (M, M)
            # ||L_K^{-1}[:,i]||^2 = sum over rows
            L_K_inv_col_sq = jnp.sum(L_K_inv**2, axis=0)  # (M,)
            trace_term = jnp.sum(q_var * L_K_inv_col_sq[:, None])

            # log|K| - log|S|
            logdet_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)))
            logdet_S = jnp.sum(jnp.log(q_var))
            logdet_diff = R * logdet_K - logdet_S
        else:
            # White prior: K = I
            mahal = jnp.sum(q_mu**2)
            trace_term = jnp.sum(q_var)
            logdet_diff = -jnp.sum(jnp.log(q_var))

        kl = 0.5 * (logdet_diff - M * R + trace_term + mahal)

    else:
        # q_sqrt shape: (R, M, M) — full lower-triangular Cholesky factors
        for r in range(R):
            q_mu_r = q_mu[:, r]  # (M,)
            L_q_r = q_sqrt[r]  # (M, M)

            if L_K is not None:
                # alpha = L_K^{-1} q_mu_r
                alpha = jsla.solve_triangular(L_K, q_mu_r, lower=True)
                mahal_r = jnp.sum(alpha**2)

                # L_K^{-1} L_q
                L_K_inv_L_q = jsla.solve_triangular(L_K, L_q_r, lower=True)
                trace_r = jnp.sum(L_K_inv_L_q**2)

                logdet_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)))
                logdet_q = 2.0 * jnp.sum(jnp.log(jnp.diag(L_q_r)))
                logdet_diff_r = logdet_K - logdet_q
            else:
                mahal_r = jnp.sum(q_mu_r**2)
                trace_r = jnp.sum(L_q_r**2)
                logdet_q = 2.0 * jnp.sum(jnp.log(jnp.diag(L_q_r)))
                logdet_diff_r = -logdet_q

            kl = kl + 0.5 * (logdet_diff_r - M + trace_r + mahal_r)

    return kl
