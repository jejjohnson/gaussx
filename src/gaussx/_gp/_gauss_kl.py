"""KL divergence between Gaussian distributions."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import rearrange, reduce
from gaussx._gp._base_conditional import _as_psd_operator
from gaussx._linalg._linalg import solve_columns
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._logdet import cholesky_logdet
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolverStrategy


def gauss_kl(
    q_mu: Float[Array, "M R"],
    q_sqrt: Float[Array, "R M M"] | Float[Array, "M R"],
    K: Float[Array, "M M"] | lx.AbstractLinearOperator | None = None,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    r"""KL(first ‖ second): ``KL[q(u) || p(u)]`` with ``p(u) = N(0, K)``.

    Cholesky-parameterised variant of
    `gaussian_kl` designed for
    GP/SVGP models.  The Cholesky representation avoids explicit covariance
    matrix construction and supports both full and diagonal ``q_sqrt``.
    For lineax-operator covariances, use
    `gaussian_kl` instead.

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
        K: Prior covariance, ``(M, M)`` array or PSD operator (a
            `Kronecker` / `BlockDiag` operator is factorised per factor).
            If ``None``, uses white prior (identity).
        solver: Deprecated and ignored (the Cholesky factorisation takes
            no solver); passing one warns. Removed in gaussx
            0.7.0.

    Returns:
        Scalar KL divergence summed over all ``R`` output dimensions.

    See Also:
        `gaussian_kl`: General KL
        between two multivariate normals with lineax covariance operators.
    """
    if solver is not None:
        warn_deprecated(
            "gauss_kl(solver=...) is ignored and deprecated; it will be "
            "removed in gaussx 0.7.0."
        )
    M = q_mu.shape[0]
    R = q_mu.shape[1]

    is_diagonal = q_sqrt.ndim == 2

    # Prior Cholesky factor (structured where gaussx.cholesky preserves it)
    L_K = cholesky(_as_psd_operator(K)) if K is not None else None

    if is_diagonal:
        # q_sqrt shape: (M, R) — diagonal standard deviations
        q_var = q_sqrt**2  # (M, R)

        if L_K is not None:
            # alpha = L_K^{-1} q_mu  ->  Mahalanobis term
            alpha = solve_columns(L_K, q_mu)  # (M, R)
            mahal = jnp.sum(alpha**2)

            # tr(K^{-1} diag(q_var_r)) = sum_i q_var[i,r] * (K^{-1})_{ii}
            # diag(K^{-1}) = diag(L_K^{-T} L_K^{-1}) via cho_solve on
            # each standard basis vector — but that's O(M^2) total.
            # More efficient: solve L_K x_i = e_i for each i, then
            # diag(K^{-1})_i = ||x_i||^2.  Use a single batched solve:
            Kinv_diag = reduce(
                solve_columns(L_K, jnp.eye(M, dtype=q_mu.dtype)) ** 2,
                "i j -> j",
                "sum",
            )  # (M,)
            trace_term = jnp.sum(einx.multiply("m r, m -> m r", q_var, Kinv_diag))

            # log|K| − log|S|
            logdet_K = _factor_logdet(L_K)
            logdet_S = jnp.sum(jnp.log(q_var))
            logdet_diff = R * logdet_K - logdet_S
        else:
            # White prior: K = I
            mahal = jnp.sum(q_mu**2)
            trace_term = jnp.sum(q_var)
            logdet_diff = -jnp.sum(jnp.log(q_var))

        return 0.5 * (logdet_diff - M * R + trace_term + mahal)

    # q_sqrt shape: (R, M, M) — full lower-triangular Cholesky factors
    # Hoist prior-only quantities outside the per-output computation.
    logdet_K = _factor_logdet(L_K) if L_K is not None else 0.0

    def _kl_single(
        q_mu_r: Float[Array, " M"], L_q_r: Float[Array, "M M"]
    ) -> Float[Array, ""]:
        logdet_q = cholesky_logdet(L_q_r)

        if L_K is not None:
            alpha = solve(L_K, q_mu_r)
            mahal_r = jnp.sum(alpha**2)
            L_K_inv_L_q = solve_columns(L_K, L_q_r)
            trace_r = jnp.sum(L_K_inv_L_q**2)
            logdet_diff_r = logdet_K - logdet_q
        else:
            mahal_r = jnp.sum(q_mu_r**2)
            trace_r = jnp.sum(L_q_r**2)
            logdet_diff_r = -logdet_q

        return 0.5 * (logdet_diff_r - M + trace_r + mahal_r)

    # vmap over R output dimensions: q_mu.T -> (R, M), q_sqrt -> (R, M, M)
    kl_per_output = jax.vmap(_kl_single)(rearrange(q_mu, "m r -> r m"), q_sqrt)
    return jnp.sum(kl_per_output)


def _factor_logdet(L: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """``log|L Lᵀ| = 2 Σ log diag(L)`` for a (structured) Cholesky factor."""
    return 2.0 * jnp.sum(jnp.log(diag(L)))
