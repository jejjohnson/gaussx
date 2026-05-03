"""Schur complement and conditional variance operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._low_rank_update import LowRankUpdate


def schur_complement(
    K_XX: lx.AbstractLinearOperator,
    K_XZ: Float[Array, "N M"],
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
    from gaussx._linalg._linalg import solve_rows

    W_T = solve_rows(K_ZZ, K_XZ)  # (N, M)
    W = W_T.T  # (M, N)

    # Represent as K_XX + K_XZ @ (-I_M) @ W
    # = K_XX - K_XZ @ K_ZZ^{-1} @ K_XZ^T
    # LowRankUpdate: base + U @ diag(d) @ V^T
    # U = K_XZ (N, M), d = -ones(M), V = W^T (N, M)
    d = -jnp.ones(M)
    return LowRankUpdate(K_XX, K_XZ, d, W.T)


def conditional_variance(
    K_XX_diag: Float[Array, " N"],
    K_XZ: Float[Array, "N M"] | lx.AbstractLinearOperator | None = None,
    A_X: Float[Array, "N M"] | None = None,
    S_u: lx.AbstractLinearOperator | None = None,
) -> Float[Array, " N"]:
    """Predictive variance: Schur complement diagonal plus optional variational
    correction.

    Computes the diagonal of the conditional covariance::

        diag(K_XX - A_X K_XZ^T) + diag(A_X S_u A_X^T)

    where ``A_X = K_XZ K_ZZ^{-1}`` is the projection matrix.
    Negative base variances are clamped to zero.

    Without ``S_u`` this is the exact GP predictive variance (diagonal of the
    Schur complement).  With ``S_u`` it is the sparse GP variational predictive
    variance that adds a variational covariance correction.

    Args:
        K_XX_diag: Prior diagonal variances, shape ``(N,)``.
        K_XZ: Cross-covariance matrix, shape ``(N, M)``.
        A_X: Projection matrix ``K_XZ K_ZZ^{-1}``, shape ``(N, M)``.
        S_u: Optional variational covariance, shape ``(M, M)``.  When
            provided, adds ``diag(A_X S_u A_X^T)`` to the base variance.

    Returns:
        Predictive variances, shape ``(N,)``.

    Note:
        For one release the legacy three-positional-argument form
        ``conditional_variance(base_diag, A_X, S_u)`` (where
        ``base_diag`` was the *Schur* diagonal already, ``A_X`` was the
        projection, and ``S_u`` was the variational covariance) is
        still accepted: it is detected when the second positional
        argument is a :class:`lineax.AbstractLinearOperator` (the old
        ``S_u`` slot type). Such calls emit a
        :class:`DeprecationWarning` and compute
        ``base_diag + diag(A_X S_u A_X^T)`` without the
        Schur subtraction.
    """
    # Backwards-compat: the legacy 3-positional signature was
    # ``conditional_variance(base_diag, A_X, S_u)``. The pre-#152 docs
    # called it as ``conditional_variance(adjusted_diag, A, S_u)`` with
    # the second arg as the projection matrix and the third as the
    # variational covariance operator. We can detect that pattern by
    # the third positional being an AbstractLinearOperator (it
    # corresponds to the old ``S_u``) while ``S_u`` is left at the
    # default ``None``.
    if (
        S_u is None
        and isinstance(A_X, lx.AbstractLinearOperator)
        and isinstance(K_XZ, jax.Array)
    ):
        import warnings

        warnings.warn(
            "conditional_variance(base_diag, A_X, S_u) is deprecated; "
            "use conditional_variance(K_XX_diag, K_XZ, A_X, S_u=S_u). "
            "The legacy form treats the first argument as the "
            "precomputed Schur diagonal and skips the K_XZ-based "
            "subtraction.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy_A_X = K_XZ  # second positional was the projection matrix
        legacy_S_u: lx.AbstractLinearOperator = A_X  # third was S_u
        S_mat = legacy_S_u.as_matrix()
        AS = legacy_A_X @ S_mat
        diag_ASAt = jnp.sum(AS * legacy_A_X, axis=1)
        return jnp.clip(K_XX_diag, 0.0) + diag_ASAt

    if K_XZ is None or A_X is None:
        raise TypeError(
            "conditional_variance requires K_XZ and A_X (or the legacy "
            f"(base_diag, A_X, S_u) call). Got K_XZ={K_XZ!r}, "
            f"A_X={A_X!r}, S_u={S_u!r}."
        )
    if not isinstance(K_XZ, jax.Array):
        raise TypeError(
            "K_XZ must be a jax array; the legacy 3-positional form "
            "passes an AbstractLinearOperator as the third positional "
            "argument."
        )

    # Diagonal of Schur complement: diag(K_XX - K_XZ K_ZZ^{-1} K_ZX)
    base = jnp.clip(K_XX_diag - jnp.sum(A_X * K_XZ, axis=1), 0.0)
    if S_u is None:
        return base
    # Variational correction: diag(A_X S_u A_X^T)
    S_mat = S_u.as_matrix()
    AS = A_X @ S_mat  # (N, M)
    diag_ASAt = jnp.sum(AS * A_X, axis=1)  # (N,)
    return base + diag_ASAt
