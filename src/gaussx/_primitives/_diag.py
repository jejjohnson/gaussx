"""Structured diagonal extraction with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._kronecker import Kronecker


def diag(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Extract the diagonal of an operator as a 1D array.

    When ``stochastic=True``, uses Hutchinson's diagonal estimator
    via matfree — only requires matvec access, no materialization.

    Args:
        operator: A linear operator.
        stochastic: If ``True``, use stochastic diagonal estimation.
        num_probes: Number of probe vectors for stochastic mode.
        key: PRNG key for stochastic mode.

    Returns:
        1D array of diagonal entries (exact or estimated).
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _diag_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _diag_kronecker(operator)
    if isinstance(operator, BlockTriDiag):
        return _diag_block_tridiag(operator)
    if stochastic:
        return _diag_stochastic(operator, num_probes, key)
    return jnp.diag(operator.as_matrix())


def _diag_block_diag(operator: BlockDiag) -> jnp.ndarray:
    return jnp.concatenate([diag(op) for op in operator.operators])


def _diag_kronecker(operator: Kronecker) -> jnp.ndarray:
    """diag(A kron B) = kron(diag(A), diag(B))."""
    return ft.reduce(jnp.kron, (diag(op) for op in operator.operators))


def _diag_block_tridiag(operator: BlockTriDiag) -> jnp.ndarray:
    """Extract block-diagonal entries of a block-tridiagonal operator."""
    from einops import rearrange

    # diagonal blocks contain the diagonal entries
    block_diags = jax.vmap(jnp.diag)(operator.diagonal)  # (N, d)
    return rearrange(block_diags, "N d -> (N d)")


def _diag_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jnp.ndarray | None,
) -> jnp.ndarray:
    """Hutchinson diagonal estimator via matfree."""
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    sample_shape = jnp.zeros(n)
    integrand = matfree.stochtrace.integrand_diagonal()
    sampler = matfree.stochtrace.sampler_rademacher(sample_shape, num=num_probes)
    estimator = matfree.stochtrace.estimator(integrand, sampler)
    return estimator(operator.mv, key)
