"""Structured diagonal extraction with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker


def diag(operator: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Extract the diagonal of an operator as a 1D array.

    Args:
        operator: A linear operator.

    Returns:
        1D array of diagonal entries.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _diag_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _diag_kronecker(operator)
    return jnp.diag(operator.as_matrix())


def _diag_block_diag(operator: BlockDiag) -> jnp.ndarray:
    return jnp.concatenate([diag(op) for op in operator.operators])


def _diag_kronecker(operator: Kronecker) -> jnp.ndarray:
    """diag(A kron B) = kron(diag(A), diag(B))."""
    return ft.reduce(jnp.kron, (diag(op) for op in operator.operators))
