"""Structured matrix square root with dispatch on operator type."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker


def sqrt(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Compute matrix square root S such that S @ S = A.

    Requires A to be positive semi-definite.

    Args:
        operator: A PSD linear operator.

    Returns:
        Operator S satisfying S @ S = A.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _sqrt_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _sqrt_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _sqrt_kronecker(operator)
    return _sqrt_dense(operator)


def _sqrt_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> lx.DiagonalLinearOperator:
    diag = lx.diagonal(operator)
    return lx.DiagonalLinearOperator(jnp.sqrt(diag))


def _sqrt_block_diag(operator: BlockDiag) -> BlockDiag:
    return BlockDiag(*(sqrt(op) for op in operator.operators))


def _sqrt_kronecker(operator: Kronecker) -> Kronecker:
    return Kronecker(*(sqrt(op) for op in operator.operators))


def _sqrt_dense(
    operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Eigendecomposition: S = Q diag(sqrt(lam)) Q^T."""
    mat = operator.as_matrix()
    eigenvalues, eigenvectors = jnp.linalg.eigh(mat)
    sqrt_eigs = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    S = eigenvectors @ jnp.diag(sqrt_eigs) @ eigenvectors.T
    return lx.MatrixLinearOperator(S, lx.symmetric_tag)
