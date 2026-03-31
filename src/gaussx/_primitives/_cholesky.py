"""Structured Cholesky factorization with dispatch on operator type."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag, LowerBlockTriDiag
from gaussx._operators._kronecker import Kronecker


def cholesky(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Compute Cholesky factor L such that A = L L^T.

    Returns a linear operator (not a raw array). For structured
    operators, the result preserves structure.

    Args:
        operator: A positive-definite linear operator.

    Returns:
        Lower-triangular operator L.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _cholesky_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _cholesky_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _cholesky_kronecker(operator)
    if isinstance(operator, BlockTriDiag):
        return _cholesky_block_tridiag(operator)
    return _cholesky_dense(operator)


def _cholesky_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> lx.DiagonalLinearOperator:
    diag = lx.diagonal(operator)
    return lx.DiagonalLinearOperator(jnp.sqrt(diag))


def _cholesky_block_diag(operator: BlockDiag) -> BlockDiag:
    return BlockDiag(*(cholesky(op) for op in operator.operators))


def _cholesky_kronecker(operator: Kronecker) -> Kronecker:
    return Kronecker(*(cholesky(op) for op in operator.operators))


def _cholesky_block_tridiag(operator: BlockTriDiag) -> LowerBlockTriDiag:
    """Block-banded Cholesky factorization in O(Nd³).

    For each block k:
        L_k = chol(D_k - B_k L_{k-1}^{-T} L_{k-1}^{-1} B_k^T)
            = chol(D_k - B_k B_k^T)  ... simplified via recurrence
        B_{k} = A_k @ L_{k-1}^{-T}

    where A_k are the sub-diagonal blocks of the original matrix.
    """
    N = operator._num_blocks

    def scan_fn(carry, k):
        L_prev = carry
        A_k = operator.sub_diagonal[k - 1]
        # B_k = A_k @ L_{k-1}^{-T}
        B_k = jax.scipy.linalg.solve_triangular(L_prev, A_k.T, lower=True).T
        # D_k - B_k @ B_k^T
        S_k = operator.diagonal[k] - B_k @ B_k.T
        L_k = jnp.linalg.cholesky(S_k)
        return L_k, (L_k, B_k)

    # First block
    L_0 = jnp.linalg.cholesky(operator.diagonal[0])
    _, (L_diag_rest, B_sub) = jax.lax.scan(scan_fn, L_0, jnp.arange(1, N))
    # Assemble
    L_diag = jnp.concatenate([L_0[None], L_diag_rest], axis=0)
    return LowerBlockTriDiag(L_diag, B_sub)


def _cholesky_dense(
    operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    mat = operator.as_matrix()
    L = jax.scipy.linalg.cholesky(mat, lower=True)
    return lx.MatrixLinearOperator(L, lx.lower_triangular_tag)
