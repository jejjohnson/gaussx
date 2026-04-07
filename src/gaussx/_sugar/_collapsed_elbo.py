"""Collapsed ELBO (Titsias variational bound) for sparse GP regression."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def collapsed_elbo(
    y: jnp.ndarray,
    K_diag: jnp.ndarray,
    K_xz: jnp.ndarray,
    K_zz: jnp.ndarray,
    noise_var: float,
) -> jnp.ndarray:
    """Collapsed ELBO (Titsias bound) for sparse GP regression.

    Computes the variational lower bound on the log marginal likelihood
    using the matrix determinant lemma for O(NM^2 + M^3) cost::

        ELBO = log N(y | 0, Q_ff + sigma^2 I)
               - 0.5 / sigma^2 * tr(K_ff - Q_ff)

    where ``Q_ff = K_xz @ K_zz^{-1} @ K_xz^T`` is the Nystrom
    approximation.

    Args:
        y: Observations, shape ``(N,)``.
        K_diag: Diagonal of the full kernel matrix ``K_ff``, shape ``(N,)``.
        K_xz: Cross-covariance between data and inducing points,
            shape ``(N, M)``.
        K_zz: Inducing point kernel matrix, shape ``(M, M)``.
        noise_var: Observation noise variance (scalar).

    Returns:
        Scalar ELBO value.
    """
    N = y.shape[0]
    M = K_zz.shape[0]
    log_2pi = jnp.log(2.0 * jnp.pi)

    # L_zz @ L_zz^T = K_zz
    L_zz = jnp.linalg.cholesky(K_zz)

    # V = L_zz^{-1} @ K_xz^T, shape (M, N)
    V = lax.linalg.triangular_solve(L_zz, K_xz.T, left_side=True, lower=True)

    # B = I_M + (1/sigma^2) V @ V^T, shape (M, M)
    B = jnp.eye(M) + (1.0 / noise_var) * (V @ V.T)
    L_B = jnp.linalg.cholesky(B)

    # Log-determinant: log|Q_ff + sigma^2 I| = N*log(sigma^2) + 2*sum(log(diag(L_B)))
    log_det = N * jnp.log(noise_var) + 2.0 * jnp.sum(jnp.log(jnp.diag(L_B)))

    # Quadratic form: y^T (Q_ff + sigma^2 I)^{-1} y via Woodbury
    # = (1/sigma^2)(y^T y - (1/sigma^2) ||L_B^{-1} V y||^2)
    Vy = V @ y  # (M,)
    LBinv_Vy = lax.linalg.triangular_solve(L_B, Vy, left_side=True, lower=True)
    quad = (1.0 / noise_var) * (
        jnp.sum(y**2) - (1.0 / noise_var) * jnp.sum(LBinv_Vy**2)
    )

    # Trace penalty: -0.5/sigma^2 * (tr(K_ff) - tr(Q_ff))
    # tr(Q_ff) = ||V||_F^2
    trace_penalty = -0.5 / noise_var * (jnp.sum(K_diag) - jnp.sum(V**2))

    return -0.5 * (log_det + quad + N * log_2pi) + trace_penalty
