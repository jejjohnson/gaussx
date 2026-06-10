"""Structured diagonal extraction with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sum_kronecker import SumKronecker
from gaussx._primitives._samplers import SamplerName, resolve_sampler


def diag(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName = "signs",
) -> Float[Array, " n"]:
    """Extract the diagonal of an operator as a 1D array.

    When ``stochastic=True``, uses Hutchinson's diagonal estimator
    via matfree — only requires matvec access, no materialization.

    Args:
        operator: A linear operator.
        stochastic: If ``True``, use stochastic diagonal estimation.
        num_probes: Number of probe vectors for stochastic mode.
        key: PRNG key for stochastic mode.
        sampler: Probe distribution for stochastic mode (``"signs"``,
            ``"normal"``, ``"sphere"``).

    Returns:
        1D array of diagonal entries (exact or estimated).
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        return jnp.ones(operator.in_size(), dtype=operator.in_structure().dtype)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _diag_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _diag_kronecker(operator)
    if isinstance(operator, BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag):
        return _diag_block_tridiag(operator)
    if isinstance(operator, LowRankUpdate):
        return _diag_low_rank(operator)
    if isinstance(operator, KroneckerSum):
        return _diag_kronecker_sum(operator)
    if isinstance(operator, SumKronecker):
        return ft.reduce(jnp.add, (diag(kron) for kron in operator.operators))
    if isinstance(operator, lx.TaggedLinearOperator):
        return diag(
            operator.operator,
            stochastic=stochastic,
            num_probes=num_probes,
            key=key,
            sampler=sampler,
        )
    if isinstance(operator, lx.AddLinearOperator):
        return diag(operator.operator1) + diag(operator.operator2)
    if isinstance(operator, lx.MulLinearOperator):
        return operator.scalar * diag(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        return diag(operator.operator) / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        return -diag(operator.operator)
    if stochastic:
        return _diag_stochastic(operator, num_probes, key, sampler)
    return jnp.diag(operator.as_matrix())


def _diag_block_diag(operator: BlockDiag) -> Float[Array, " n"]:
    return jnp.concatenate([diag(op) for op in operator.operators])


def _diag_kronecker(operator: Kronecker) -> Float[Array, " n"]:
    """diag(A kron B) = kron(diag(A), diag(B))."""
    return ft.reduce(jnp.kron, (diag(op) for op in operator.operators))


def _diag_block_tridiag(
    operator: BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag,
) -> Float[Array, " n"]:
    """Extract block-diagonal entries of a block-(tri/bi)diagonal operator."""
    from gaussx._einx import rearrange

    # diagonal blocks contain the diagonal entries
    block_diags = jax.vmap(jnp.diag)(operator.diagonal)  # (N, d)
    return rearrange(block_diags, "N d -> (N d)")


def _diag_low_rank(operator: LowRankUpdate) -> Float[Array, " n"]:
    """diag(L + U diag(d) V^T) = diag(L) + sum_k U[:, k] d[k] V[:, k]."""
    from gaussx._einx import reduce

    update = reduce(operator.U * operator.d * operator.V, "n k -> n", "sum")
    return diag(operator.base) + update


def _diag_kronecker_sum(operator: KroneckerSum) -> Float[Array, " n"]:
    """diag(A (+) B) = kron(diag(A), 1_b) + kron(1_a, diag(B))."""
    diag_a = diag(operator.A)
    diag_b = diag(operator.B)
    from gaussx._einx import rearrange

    grid = diag_a[:, None] + diag_b[None, :]
    return rearrange(grid, "a b -> (a b)")


def _diag_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array | None,
    sampler: SamplerName,
) -> Float[Array, " n"]:
    """Hutchinson diagonal estimator via matfree."""
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    probe_fn = resolve_sampler(sampler, n, num_probes)
    integrand = matfree.stochtrace.monte_carlo_diagonal()
    estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    return estimate(operator.mv, key)
