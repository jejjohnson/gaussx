"""Linear algebra sugar: covariance transform, trace of product, solve helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from einops import reduce
from jaxtyping import Array, Float

from gaussx._linalg._schur import conditional_variance as _conditional_variance
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve


def cov_transform(
    J: Float[Array, "M N"],
    cov_operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Covariance propagation through a linear map: ``J @ Sigma @ J^T``.

    Used in error propagation, Kalman filter updates, and
    first-order uncertainty propagation.

    Exploits structure where it can:

    - **Diagonal** ``cov_operator``: computes ``(J * d) @ J^T`` directly,
      skipping the ``(N, N)`` materialization of ``Sigma``.

    Otherwise materializes ``Sigma`` and forms the dense product. The
    returned operator is always tagged symmetric when the input is.

    Args:
        J: Jacobian or linear map, shape ``(M, N)``.
        cov_operator: Input covariance, shape ``(N, N)``.

    Returns:
        Transformed covariance operator, shape ``(M, M)``.
    """
    tags: frozenset[object] = frozenset()
    if lx.is_symmetric(cov_operator):
        tags = frozenset({lx.symmetric_tag})

    if isinstance(cov_operator, lx.DiagonalLinearOperator):
        d = lx.diagonal(cov_operator)
        result = (J * d[None, :]) @ J.T
        return lx.MatrixLinearOperator(result, tags)

    Sigma = cov_operator.as_matrix()
    result = J @ Sigma @ J.T
    return lx.MatrixLinearOperator(result, tags)


def trace_product(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
) -> Float[Array, ""]:
    """Trace of a matrix product: ``tr(A @ B)`` with structural dispatch.

    Uses operator structure where possible to avoid materialization:

    - **Both diagonal**: ``sum(diag(A) * diag(B))``.
    - **Diagonal × general** (or vice versa): contract the diagonal with
      the diagonal of the other operator (no full materialization).
    - **Matched** :class:`gaussx.BlockDiag` (same block sizes): sum of
      per-block ``trace_product``.
    - **Matched** :class:`gaussx.Kronecker` (same factor structure):
      ``prod_i tr(A_i @ B_i)``.
    - Otherwise falls back to ``sum(A * B^T)`` on the materialized
      matrices — the same O(N²) cost the previous implementation paid.

    Args:
        A: Linear operator, shape ``(N, N)``.
        B: Linear operator, shape ``(N, N)``.

    Returns:
        Scalar ``tr(A @ B)``.
    """
    from gaussx._operators._block_diag import BlockDiag
    from gaussx._operators._kronecker import Kronecker
    from gaussx._primitives._diag import diag

    # Both diagonal: O(N) inner product of diagonals.
    if isinstance(A, lx.DiagonalLinearOperator) and isinstance(
        B, lx.DiagonalLinearOperator
    ):
        return jnp.sum(lx.diagonal(A) * lx.diagonal(B))

    # Diagonal × anything: tr(D @ B) = sum(diag(D) * diag(B)).
    if isinstance(A, lx.DiagonalLinearOperator):
        return jnp.sum(lx.diagonal(A) * diag(B))
    if isinstance(B, lx.DiagonalLinearOperator):
        return jnp.sum(diag(A) * lx.diagonal(B))

    # Matched BlockDiag: tr(blockdiag(A_i) @ blockdiag(B_i)) = sum_i tr(A_i @ B_i).
    if (
        isinstance(A, BlockDiag)
        and isinstance(B, BlockDiag)
        and len(A.operators) == len(B.operators)
        and all(
            a.in_size() == b.in_size()
            for a, b in zip(A.operators, B.operators, strict=True)
        )
    ):
        parts = [
            trace_product(a, b) for a, b in zip(A.operators, B.operators, strict=True)
        ]
        return jnp.sum(jnp.stack(parts))

    # Matched Kronecker: tr((A1⊗A2⊗…) @ (B1⊗B2⊗…)) = prod_i tr(A_i @ B_i).
    if (
        isinstance(A, Kronecker)
        and isinstance(B, Kronecker)
        and len(A.operators) == len(B.operators)
        and all(
            a.in_size() == b.in_size()
            for a, b in zip(A.operators, B.operators, strict=True)
        )
    ):
        parts = [
            trace_product(a, b) for a, b in zip(A.operators, B.operators, strict=True)
        ]
        return jnp.prod(jnp.stack(parts))

    return _trace_product_dense(A, B)


def _trace_product_dense(
    A: lx.AbstractLinearOperator, B: lx.AbstractLinearOperator
) -> Float[Array, ""]:
    """Dense fallback: ``sum(A * B^T)`` element-wise."""
    A_mat = A.as_matrix()
    B_mat = B.as_matrix()
    return reduce(A_mat * B_mat.T, "i j -> ", "sum")


def diag_conditional_variance(
    K_XX_diag: Float[Array, " N"],
    K_XZ: Float[Array, "N M"],
    A_X: Float[Array, "N M"],
) -> Float[Array, " N"]:
    """Diagonal of Schur complement: ``diag(K_XX - A K_ZX)``.

    Thin wrapper around :func:`gaussx.conditional_variance` (defined in
    :func:`gaussx._linalg._schur.conditional_variance`) without a
    variational covariance.  Use :func:`gaussx.conditional_variance`
    directly when a variational correction ``S_u`` is also needed.

    Args:
        K_XX_diag: Prior diagonal variances, shape ``(N,)``.
        K_XZ: Cross-covariance, shape ``(N, M)``.
        A_X: Projection matrix ``K_XZ K_ZZ^{-1}``, shape ``(N, M)``.

    Returns:
        Conditional variances, shape ``(N,)``.
    """
    return _conditional_variance(K_XX_diag, K_XZ, A_X)


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


def solve_matrix(
    operator: lx.AbstractLinearOperator,
    matrix: Float[Array, "N K"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, "N K"]:
    """Solve ``A X = B`` with a single factorization on the matrix RHS.

    When ``solver`` is ``None`` and ``A`` is positive semidefinite, this
    factors ``A = L L^T`` once via :func:`gaussx.cholesky` and then uses
    a single ``cho_solve`` on the full matrix RHS — avoiding the
    per-column re-factorization incurred by :func:`solve_columns`.

    For non-PSD operators (or when a custom ``solver`` is supplied),
    falls back to :func:`solve_columns`.

    Args:
        operator: Linear operator A, shape ``(N, N)``.
        matrix: Right-hand side B, shape ``(N, K)``.
        solver: Optional solver strategy. When provided, dispatch is
            delegated column-by-column via :func:`solve_columns`.

    Returns:
        Solution X = A⁻¹ B, shape ``(N, K)``.
    """
    if solver is None and lx.is_positive_semidefinite(operator):
        L = cholesky(operator).as_matrix()
        return jax.scipy.linalg.cho_solve((L, True), matrix)
    return solve_columns(operator, matrix, solver=solver)


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
