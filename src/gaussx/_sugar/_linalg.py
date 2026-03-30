"""Linear algebra sugar: covariance transform, trace of product."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx


def cov_transform(
    J: jnp.ndarray,
    cov_operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Covariance propagation through a linear map: ``J @ Sigma @ J^T``.

    Used in error propagation, Kalman filter updates, and
    first-order uncertainty propagation.

    Args:
        J: Jacobian or linear map, shape ``(M, N)``.
        cov_operator: Input covariance, shape ``(N, N)``.

    Returns:
        Transformed covariance operator, shape ``(M, M)``.
    """
    Sigma = cov_operator.as_matrix()
    result = J @ Sigma @ J.T
    tags = frozenset()
    if lx.is_symmetric(cov_operator):
        tags = frozenset({lx.symmetric_tag})
    return lx.MatrixLinearOperator(result, tags)


def trace_product(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
) -> jnp.ndarray:
    """Trace of a matrix product: ``tr(A @ B)`` without forming the product.

    Uses the identity ``tr(AB) = sum(A * B^T)`` element-wise.

    Args:
        A: Linear operator, shape ``(N, N)``.
        B: Linear operator, shape ``(N, N)``.

    Returns:
        Scalar ``tr(A @ B)``.
    """
    A_mat = A.as_matrix()
    B_mat = B.as_matrix()
    return jnp.sum(A_mat * B_mat.T)


def diag_conditional_variance(
    K_XX_diag: jnp.ndarray,
    K_XZ: jnp.ndarray,
    A_X: jnp.ndarray,
) -> jnp.ndarray:
    """Base conditional variance without variational covariance.

    Computes::

        var_i = K_XX_diag[i] - sum_m A_X[i,m] * K_XZ[i,m]

    This is the Schur complement diagonal: ``diag(K_XX - A K_ZX)``
    where ``A = K_XZ K_ZZ^{-1}``. Negative values are clamped to 0.

    Used in sparse GP prediction as the base variance before adding
    the variational covariance contribution.

    Args:
        K_XX_diag: Prior diagonal variances, shape ``(N,)``.
        K_XZ: Cross-covariance, shape ``(N, M)``.
        A_X: Projection matrix ``K_XZ K_ZZ^{-1}``, shape ``(N, M)``.

    Returns:
        Conditional variances, shape ``(N,)``.
    """
    correction = jnp.sum(A_X * K_XZ, axis=1)
    return jnp.clip(K_XX_diag - correction, 0.0)
