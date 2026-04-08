"""Schur complement and conditional variance operations."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._operators._low_rank_update import LowRankUpdate


def schur_complement(
    K_XX: lx.AbstractLinearOperator,
    K_XZ: jnp.ndarray,
    K_ZZ: lx.AbstractLinearOperator,
) -> LowRankUpdate:
    """Schur complement: ``K_XX - K_XZ @ K_ZZ^{-1} @ K_ZX``.

    Central to GP conditional distributions. Returns a
    ``LowRankUpdate`` operator so that downstream operations
    (solve, logdet) can exploit the low-rank structure.

    Args:
        K_XX: Prior covariance, shape ``(N, N)``.
        K_XZ: Cross-covariance, shape ``(N, M)``.
        K_ZZ: Inducing covariance, shape ``(M, M)``.

    Returns:
        A ``LowRankUpdate`` representing K_XX - K_XZ K_ZZ^{-1} K_ZX.
    """
    # Solve K_ZZ @ W_j = K_XZ[j, :]^T for each row j of K_XZ
    # W = K_ZZ^{-1} K_XZ^T has shape (M, N)
    # Then Schur = K_XX - K_XZ @ K_ZZ^{-1} @ K_XZ^T
    _N, M = K_XZ.shape

    # Solve K_ZZ w_j = k_xz_j for each row of K_XZ
    # W^T = vmap(solve(K_ZZ, ·))(K_XZ)  =>  (N, M)
    from gaussx._sugar._linalg import solve_rows

    W_T = solve_rows(K_ZZ, K_XZ)  # (N, M)
    W = W_T.T  # (M, N)

    # Represent as K_XX + K_XZ @ (-I_M) @ W
    # = K_XX - K_XZ @ K_ZZ^{-1} @ K_XZ^T
    # LowRankUpdate: base + U @ diag(d) @ V^T
    # U = K_XZ (N, M), d = -ones(M), V = W^T (N, M)
    d = -jnp.ones(M)
    return LowRankUpdate(K_XX, K_XZ, d, W.T)


def conditional_variance(
    K_XX_diag: jnp.ndarray,
    A_X: jnp.ndarray,
    S_u: lx.AbstractLinearOperator,
) -> jnp.ndarray:
    """Predictive variance for GP conditionals.

    Computes::

        diag(K_XX) - diag(A @ K_ZZ @ A^T) + diag(A @ S_u @ A^T)

    which simplifies to::

        K_XX_diag + sum_j (A_X S_u A_X^T)_ii

    In practice, ``A_X`` already absorbs the ``K_ZZ^{-1}`` projection,
    so this computes::

        K_XX_diag + diag(A_X @ S_u @ A_X^T)

    The diagonal of ``A @ S_u @ A^T`` is computed row-by-row to avoid
    materializing the full ``(N, N)`` product.

    Args:
        K_XX_diag: Prior diagonal variances, shape ``(N,)``.
        A_X: Projection matrix, shape ``(N, M)``.
        S_u: Variational covariance, shape ``(M, M)``.

    Returns:
        Predictive variances, shape ``(N,)``.
    """
    # diag(A @ S @ A^T) = sum_j (A * (S @ A^T)^T)  per row
    S_mat = S_u.as_matrix()
    AS = A_X @ S_mat  # (N, M)
    diag_ASAt = jnp.sum(AS * A_X, axis=1)  # (N,)
    return K_XX_diag + diag_ASAt
