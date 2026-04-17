"""Unwhitening: transform whitened variational parameters to original space."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._linalg._linalg import cov_transform


def unwhiten(
    m_tilde: jnp.ndarray,
    L: lx.AbstractLinearOperator,
) -> jnp.ndarray:
    """Unwhiten variational mean: ``m = L @ m_tilde``.

    Args:
        m_tilde: Whitened mean vector, shape ``(M,)``.
        L: Cholesky factor, shape ``(M, M)``.

    Returns:
        Unwhitened mean m, shape ``(M,)``.
    """
    return L.mv(m_tilde)


def unwhiten_covariance(
    L: lx.AbstractLinearOperator,
    S_tilde: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Unwhiten variational covariance: S = L S̃ Lᵀ.

    Delegates to :func:`~gaussx.cov_transform`.

    Args:
        L: Cholesky factor, shape ``(M, M)``.
        S_tilde: Whitened variational covariance, shape ``(M, M)``.

    Returns:
        Unwhitened covariance operator S.
    """
    return cov_transform(L.as_matrix(), S_tilde)


# Backward-compatible alias (the old name was misleading — this unwhitens).
whiten_covariance = unwhiten_covariance
