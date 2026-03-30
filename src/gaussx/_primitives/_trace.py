"""Structured trace with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker


def trace(operator: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Compute the trace of an operator.

    Args:
        operator: A square linear operator.

    Returns:
        Scalar trace value.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return jnp.sum(lx.diagonal(operator))
    if isinstance(operator, BlockDiag):
        return _trace_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _trace_kronecker(operator)
    return jnp.trace(operator.as_matrix())


def _trace_block_diag(operator: BlockDiag) -> jnp.ndarray:
    return ft.reduce(jnp.add, (trace(op) for op in operator.operators))


def _trace_kronecker(operator: Kronecker) -> jnp.ndarray:
    """trace(A kron B) = trace(A) * trace(B)."""
    return ft.reduce(jnp.multiply, (trace(op) for op in operator.operators))
