"""Structured trace with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker


def trace(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute the trace of an operator.

    When ``stochastic=True``, uses Hutchinson's estimator via
    matfree — only requires matvec access, no materialization.

    Args:
        operator: A square linear operator.
        stochastic: If ``True``, use stochastic trace estimation.
        num_probes: Number of probe vectors for stochastic mode.
        key: PRNG key for stochastic mode.

    Returns:
        Scalar trace value (exact or estimated).
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return jnp.sum(lx.diagonal(operator))
    if isinstance(operator, BlockDiag):
        return _trace_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _trace_kronecker(operator)
    if stochastic:
        return _trace_stochastic(operator, num_probes, key)
    return jnp.trace(operator.as_matrix())


def _trace_block_diag(operator: BlockDiag) -> jnp.ndarray:
    return ft.reduce(jnp.add, (trace(op) for op in operator.operators))


def _trace_kronecker(operator: Kronecker) -> jnp.ndarray:
    """trace(A kron B) = trace(A) * trace(B)."""
    return ft.reduce(jnp.multiply, (trace(op) for op in operator.operators))


def _trace_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jnp.ndarray | None,
) -> jnp.ndarray:
    """Hutchinson trace estimator via matfree."""
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    sample_shape = jnp.zeros(n)
    integrand = matfree.stochtrace.integrand_trace()
    sampler = matfree.stochtrace.sampler_rademacher(sample_shape, num=num_probes)
    estimator = matfree.stochtrace.estimator(integrand, sampler)
    return estimator(operator.mv, key)
