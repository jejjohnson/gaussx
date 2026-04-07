"""Gaussian conditional via Schur complement (base_conditional)."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsla
from jaxtyping import Array, Float


def base_conditional(
    K_mm: Float[Array, "M M"],
    K_mn: Float[Array, "M N"],
    K_nn: Float[Array, "N N"] | Float[Array, " N"],
    f: Float[Array, "M R"],
    *,
    q_sqrt: Float[Array, "R M M"] | Float[Array, "M R"] | None = None,
    white: bool = False,
) -> tuple[Float[Array, "N R"], Float[Array, ...]]:
    r"""Gaussian conditional distribution via Schur complement.

    Computes the conditional distribution ``q(f_* | u)`` given:

    - Prior covariance ``K_mm`` at inducing locations
    - Cross-covariance ``K_mn`` between inducing and test locations
    - Prior (co)variance ``K_nn`` at test locations (full or diagonal)
    - Inducing function values ``f`` (or whitened values if ``white=True``)
    - Optional variational posterior ``q(u) = N(f, q_sqrt q_sqrt^T)``

    The conditional mean is::

        mu = K_nm K_mm^{-1} f   (or  K_nm L_mm^{-T} f  if white)

    The conditional covariance is::

        Sigma = K_nn - K_nm K_mm^{-1} K_mn + K_nm K_mm^{-1} S K_mm^{-1} K_mn

    where ``S = q_sqrt @ q_sqrt^T`` is the variational covariance.

    Args:
        K_mm: Prior covariance at inducing points, shape ``(M, M)``.
        K_mn: Cross-covariance, shape ``(M, N)``.
        K_nn: Test-point covariance.  Full ``(N, N)`` or diagonal ``(N,)``.
        f: Inducing function values, shape ``(M, R)``.
        q_sqrt: Optional variational Cholesky factor.
            Full: ``(R, M, M)``, diagonal: ``(M, R)``, or ``None``.
        white: If ``True``, ``f`` and ``q_sqrt`` are in whitened space
            (prior is ``N(0, I)``).

    Returns:
        ``(mean, var)`` where ``mean`` has shape ``(N, R)`` and ``var``
        has shape ``(N, N, R)`` (full K_nn) or ``(N, R)`` (diagonal K_nn).
    """
    N = K_mn.shape[1]

    # Cholesky of prior
    L_mm = jnp.linalg.cholesky(K_mm)  # (M, M)

    # A = L_mm^{-1} K_mn  ->  (M, N)
    A = jsla.solve_triangular(L_mm, K_mn, lower=True)

    # --- Conditional mean ---
    if white:
        # f is already in whitened space: mean = A^T f
        mean = A.T @ f  # (N, R)
    else:
        # mean = K_mn^T K_mm^{-1} f = A^T (L_mm^{-1} f)
        alpha = jsla.solve_triangular(L_mm, f, lower=True)  # (M, R)
        mean = A.T @ alpha  # (N, R)

    # --- Conditional variance ---
    is_diag_knn = K_nn.ndim == 1

    # Prior variance reduction: K_nn - K_nm K_mm^{-1} K_mn
    # = K_nn - A^T A
    if is_diag_knn:
        # K_nn is diagonal (N,)
        # diag(A^T A) = sum(A^2, axis=0)
        prior_reduction = jnp.sum(A**2, axis=0)  # (N,)
        var_base = K_nn - prior_reduction  # (N,)
    else:
        # K_nn is full (N, N)
        var_base = K_nn - A.T @ A  # (N, N)

    if q_sqrt is not None:
        is_diag_q = q_sqrt.ndim == 2
        R = f.shape[1]

        if is_diag_q:
            # q_sqrt: (M, R) — diagonal standard deviations
            # S = diag(q_sqrt^2)
            # Variance adjustment: A^T diag(q_sqrt_r^2) A per output r
            if is_diag_knn:
                # var shape: (N, R)
                var_list = []
                for r in range(R):
                    if white:
                        A_scaled = q_sqrt[:, r : r + 1] * A
                    else:
                        L_q = jnp.diag(q_sqrt[:, r])
                        A_scaled = jsla.solve_triangular(L_mm, L_q, lower=True).T @ A
                    var_adj = jnp.sum(A_scaled**2, axis=0)
                    var_list.append(var_base + var_adj)
                var = jnp.stack(var_list, axis=-1)  # (N, R)
            else:
                var_list = []
                for r in range(R):
                    if white:
                        A_scaled = q_sqrt[:, r : r + 1] * A
                    else:
                        L_q = jnp.diag(q_sqrt[:, r])
                        A_scaled = jsla.solve_triangular(L_mm, L_q, lower=True).T @ A
                    var_adj = A_scaled.T @ A_scaled
                    var_list.append(var_base + var_adj)
                var = jnp.stack(var_list, axis=-1)  # (N, N, R)
        else:
            # q_sqrt: (R, M, M) — full Cholesky factors
            if is_diag_knn:
                var_list = []
                for r in range(R):
                    L_q = q_sqrt[r]  # (M, M)
                    if white:
                        A_scaled = L_q.T @ A
                    else:
                        L_q = jsla.solve_triangular(L_mm, L_q, lower=True)
                        A_scaled = L_q.T @ A
                    var_adj = jnp.sum(A_scaled**2, axis=0)
                    var_list.append(var_base + var_adj)
                var = jnp.stack(var_list, axis=-1)  # (N, R)
            else:
                var_list = []
                for r in range(R):
                    L_q = q_sqrt[r]
                    if white:
                        A_scaled = L_q.T @ A
                    else:
                        L_q = jsla.solve_triangular(L_mm, L_q, lower=True)
                        A_scaled = L_q.T @ A
                    var_adj = A_scaled.T @ A_scaled
                    var_list.append(var_base + var_adj)
                var = jnp.stack(var_list, axis=-1)  # (N, N, R)
    else:
        # No variational posterior — just prior conditional
        if is_diag_knn:
            # Broadcast to (N, R) shape
            R = f.shape[1]
            var = jnp.broadcast_to(var_base[:, None], (N, R))
        else:
            R = f.shape[1]
            var = jnp.broadcast_to(var_base[..., None], (N, N, R))

    return mean, var
