"""Linear algebra sugar: covariance transform, trace of product, solve helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from einops import reduce
from jaxtyping import Array, Float

from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve


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
    return reduce(A_mat * B_mat.T, "i j -> ", "sum")


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
    correction = reduce(A_X * K_XZ, "N M -> N", "sum")
    return jnp.clip(K_XX_diag - correction, 0.0)


def solve_columns(
    operator: lx.AbstractLinearOperator,
    matrix: Float[Array, "N K"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, "N K"]:
    """Solve A X = B column-by-column via vmap.

    Equivalent to ``A⁻¹ @ B`` but uses structured dispatch per column.
    Use this when ``A`` is a lineax operator with efficient per-vector
    solve (e.g., Cholesky, diagonal, block-diagonal).

    Args:
        operator: Linear operator A, shape ``(N, N)``.
        matrix: Right-hand side B, shape ``(N, K)``.
        solver: Optional solver strategy.

    Returns:
        Solution X = A⁻¹ B, shape ``(N, K)``.
    """
    if solver is not None:
        return jax.vmap(
            lambda col: dispatch_solve(operator, col, solver),
            in_axes=1,
            out_axes=1,
        )(matrix)
    return jax.vmap(
        lambda col: solve(operator, col),
        in_axes=1,
        out_axes=1,
    )(matrix)


def solve_rows(
    operator: lx.AbstractLinearOperator,
    matrix: Float[Array, "K N"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, "K N"]:
    """Solve A x = bᵢ for each row bᵢ of a matrix via vmap.

    Equivalent to ``B @ A⁻¹`` row-by-row. Used in Kalman gain, Schur
    complement, and GP prediction computations.

    Args:
        operator: Linear operator A, shape ``(N, N)``.
        matrix: Rows of right-hand sides, shape ``(K, N)``.
        solver: Optional solver strategy.

    Returns:
        Solutions, shape ``(K, N)``.
    """
    if solver is not None:
        return jax.vmap(
            lambda row: dispatch_solve(operator, row, solver),
        )(matrix)
    return jax.vmap(lambda row: solve(operator, row))(matrix)
