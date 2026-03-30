"""Structured Cholesky factorization with dispatch on operator type."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
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


def _cholesky_dense(
    operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    mat = operator.as_matrix()
    L = jax.scipy.linalg.cholesky(mat, lower=True)
    return lx.MatrixLinearOperator(L, lx.lower_triangular_tag)
