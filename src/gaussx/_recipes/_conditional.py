"""Gaussian conditional via Schur complement (base_conditional)."""

from __future__ import annotations

import jax
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
    R = f.shape[1]

    # Cholesky of prior
    L_mm = jnp.linalg.cholesky(K_mm)  # (M, M)

    # A = L_mm^{-1} K_mn  ->  (M, N)
    A = jsla.solve_triangular(L_mm, K_mn, lower=True)

    # --- Conditional mean ---
    if white:
        mean = A.T @ f  # (N, R)
    else:
        alpha = jsla.solve_triangular(L_mm, f, lower=True)  # (M, R)
        mean = A.T @ alpha  # (N, R)

    # --- Conditional variance ---
    is_diag_knn = K_nn.ndim == 1

    # Prior variance reduction: K_nn - A^T A
    if is_diag_knn:
        prior_reduction = jnp.sum(A**2, axis=0)  # (N,)
        var_base = K_nn - prior_reduction  # (N,)
    else:
        var_base = K_nn - A.T @ A  # (N, N)

    if q_sqrt is not None:
        is_diag_q = q_sqrt.ndim == 2

        if is_diag_q:
            # q_sqrt: (M, R) — diagonal standard deviations
            # For non-white: precompute B = L_mm^{-1}, so
            #   var_adj_r = diag(A^T diag(B q_sqrt_r)^T diag(B q_sqrt_r) A)
            #             = diag((q_sqrt_r * B)^T A)^2 summed over M
            # For white: A_scaled = q_sqrt_r * A elementwise

            if white:

                def _var_adj_diag_white(q_r: Float[Array, " M"]) -> Float[Array, " N"]:
                    A_scaled = q_r[:, None] * A  # (M, N)
                    return jnp.sum(A_scaled**2, axis=0)

                var_adj = jax.vmap(_var_adj_diag_white, in_axes=1, out_axes=1)(
                    q_sqrt
                )  # (N, R)
            else:
                # B = L_mm^{-T} (solve once)
                # Then for each r: scaled = q_sqrt_r * B^T @ A
                # But simpler: A_scaled_r = (B * q_sqrt_r[None, :]) @ ... no.
                # Actually: var_adj = diag(A^T (L_mm^{-1} diag(s_r))^T
                #                         (L_mm^{-1} diag(s_r)) A)
                # Let C = L_mm^{-1}  (M, M), then
                #   A_scaled = (C * s_r[None,:])^T @ ... nah.
                # Simplest efficient: C = L_mm^{-1} (precompute once)
                # Then for each r: D_r = C * s_r  (broadcast M,M * M -> M,M)
                #   var_adj_r = sum((D_r @ ... wait, we need D_r.T @ A
                # Let me just do: for each r, solve L_mm @ x = diag(s_r),
                # but diag(s_r) is diagonal so L_mm^{-1} diag(s_r) = C * s_r
                # where C[:,j] * s_r[j].  Then (C * s_r).T @ A = ...
                # Let's just precompute C once:
                C = jsla.solve_triangular(L_mm, jnp.eye(L_mm.shape[0]), lower=True)

                def _var_adj_diag_nonwhite(
                    q_r: Float[Array, " M"],
                ) -> Float[Array, " N"]:
                    # C_scaled[i,j] = C[i,j] * q_r[j]
                    C_scaled = C * q_r[None, :]  # (M, M)
                    A_scaled = C_scaled.T @ A  # (M, N)
                    return jnp.sum(A_scaled**2, axis=0)

                var_adj = jax.vmap(_var_adj_diag_nonwhite, in_axes=1, out_axes=1)(
                    q_sqrt
                )  # (N, R)

            if is_diag_knn:
                var = var_base[:, None] + var_adj  # (N, R)
            else:
                # Full K_nn: need full covariance adjustment per r
                # Recompute with full matrices
                if white:

                    def _var_full_diag_white(
                        q_r: Float[Array, " M"],
                    ) -> Float[Array, "N N"]:
                        A_scaled = q_r[:, None] * A
                        return var_base + A_scaled.T @ A_scaled

                    var = jax.vmap(_var_full_diag_white, in_axes=1, out_axes=-1)(
                        q_sqrt
                    )  # (N, N, R)
                else:

                    def _var_full_diag_nonwhite(
                        q_r: Float[Array, " M"],
                    ) -> Float[Array, "N N"]:
                        C_scaled = C * q_r[None, :]
                        A_scaled = C_scaled.T @ A
                        return var_base + A_scaled.T @ A_scaled

                    var = jax.vmap(_var_full_diag_nonwhite, in_axes=1, out_axes=-1)(
                        q_sqrt
                    )  # (N, N, R)
        else:
            # q_sqrt: (R, M, M) — full Cholesky factors
            if white:

                def _var_adj_full_white(
                    L_q: Float[Array, "M M"],
                ) -> Float[Array, " N"]:
                    A_scaled = L_q.T @ A  # (M, N)
                    return jnp.sum(A_scaled**2, axis=0)

                def _var_full_full_white(
                    L_q: Float[Array, "M M"],
                ) -> Float[Array, "N N"]:
                    A_scaled = L_q.T @ A
                    return var_base + A_scaled.T @ A_scaled

            else:

                def _var_adj_full_nonwhite(
                    L_q: Float[Array, "M M"],
                ) -> Float[Array, " N"]:
                    L_q_proj = jsla.solve_triangular(L_mm, L_q, lower=True)
                    A_scaled = L_q_proj.T @ A
                    return jnp.sum(A_scaled**2, axis=0)

                def _var_full_full_nonwhite(
                    L_q: Float[Array, "M M"],
                ) -> Float[Array, "N N"]:
                    L_q_proj = jsla.solve_triangular(L_mm, L_q, lower=True)
                    A_scaled = L_q_proj.T @ A
                    return var_base + A_scaled.T @ A_scaled

            if is_diag_knn:
                if white:
                    var_adj = jax.vmap(_var_adj_full_white)(q_sqrt)  # (R, N)
                else:
                    var_adj = jax.vmap(_var_adj_full_nonwhite)(q_sqrt)  # (R, N)
                var = var_base[None, :] + var_adj  # (R, N)
                var = var.T  # (N, R)
            else:
                if white:
                    var = jax.vmap(_var_full_full_white)(q_sqrt)  # (R, N, N)
                else:
                    var = jax.vmap(_var_full_full_nonwhite)(q_sqrt)  # (R, N, N)
                # Transpose from (R, N, N) to (N, N, R)
                var = jnp.moveaxis(var, 0, -1)
    else:
        # No variational posterior — just prior conditional
        if is_diag_knn:
            var = jnp.broadcast_to(var_base[:, None], (N, R))
        else:
            var = jnp.broadcast_to(var_base[..., None], (N, N, R))

    return mean, var
