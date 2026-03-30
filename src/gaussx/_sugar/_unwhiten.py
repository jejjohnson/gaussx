"""Unwhitening: transform whitened variational parameters to original space."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx


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


def whiten_covariance(
    L: lx.AbstractLinearOperator,
    S_tilde: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Unwhiten variational covariance: ``S = L @ S_tilde @ L^T``.

    Computes the dense product ``L @ S_tilde @ L^T``. For large
    systems, consider using structured operators directly.

    Args:
        L: Cholesky factor, shape ``(M, M)``.
        S_tilde: Whitened variational covariance, shape ``(M, M)``.

    Returns:
        Unwhitened covariance operator S.
    """
    L_mat = L.as_matrix()
    S_mat = S_tilde.as_matrix()
    result = L_mat @ S_mat @ L_mat.T
    tags = frozenset()
    if lx.is_symmetric(S_tilde):
        tags = frozenset({lx.symmetric_tag})
    return lx.MatrixLinearOperator(result, tags)
