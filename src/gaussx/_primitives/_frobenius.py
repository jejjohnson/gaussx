"""Structured Frobenius norm with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._primitives._samplers import SamplerName, resolve_sampler, split_keys


def frobenius_norm(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName = "signs",
) -> Float[Array, ""]:
    """Compute the Frobenius norm ``||A||_F`` with structural dispatch.

    Structured operators avoid materialization:

    - Diagonal: vector 2-norm of the diagonal.
    - BlockDiag: root of the sum of squared block norms.
    - Kronecker: ``||A (x) B||_F = ||A||_F * ||B||_F``.
    - Scaled/negated/tagged operators delegate to the wrapped operator.

    When ``stochastic=True``, estimates ``||A||_F^2 = tr(A^T A)`` via
    matfree's Hutchinson estimator — matvec access only.

    Args:
        operator: A linear operator.
        stochastic: If ``True``, use stochastic estimation.
        num_probes: Number of probe vectors for stochastic mode.
        key: PRNG key for stochastic mode.
        sampler: Probe distribution for stochastic mode (``"signs"``,
            ``"normal"``, ``"sphere"``).

    Returns:
        Scalar Frobenius norm (exact or estimated).
    """

    # Every recursive call forwards the estimator options, so a wrapped or
    # structured matrix-free operator is never materialised (gh-320).
    def rec(op: lx.AbstractLinearOperator, k: jax.Array | None = key) -> Array:
        return frobenius_norm(
            op, stochastic=stochastic, num_probes=num_probes, key=k, sampler=sampler
        )

    def rec_all(ops) -> list[Array]:
        ops = tuple(ops)
        keys = split_keys(key, len(ops))
        return [rec(op, k) for op, k in zip(ops, keys, strict=True)]

    if isinstance(operator, lx.IdentityLinearOperator):
        return jnp.sqrt(jnp.asarray(float(operator.in_size())))
    if isinstance(operator, lx.DiagonalLinearOperator):
        d = lx.diagonal(operator)
        return jnp.sqrt(jnp.sum(jnp.abs(d) ** 2))
    if isinstance(operator, BlockDiag):
        norms = jnp.stack(rec_all(operator.operators))
        return jnp.sqrt(jnp.sum(norms * norms))
    if isinstance(operator, Kronecker):
        return ft.reduce(jnp.multiply, rec_all(operator.operators))
    if isinstance(operator, lx.TaggedLinearOperator):
        return rec(operator.operator)
    if isinstance(operator, lx.MulLinearOperator):
        return jnp.abs(operator.scalar) * rec(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        return rec(operator.operator) / jnp.abs(operator.scalar)
    if isinstance(operator, lx.NegLinearOperator):
        return rec(operator.operator)
    if stochastic:
        return _frobenius_stochastic(operator, num_probes, key, sampler)
    # |a|², so a complex operator gets a real norm.
    mat = operator.as_matrix()
    return jnp.sqrt(jnp.sum(jnp.abs(mat) ** 2))


def _frobenius_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array | None,
    sampler: SamplerName,
) -> Float[Array, ""]:
    """Hutchinson estimator of ``||A||_F = sqrt(tr(A^T A))`` via matfree."""
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    probe_fn = resolve_sampler(
        sampler, n, num_probes, dtype=operator.in_structure().dtype
    )
    integrand = matfree.stochtrace.monte_carlo_frobeniusnorm_squared()
    estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    norm_sq = estimate(operator.mv, key)
    return jnp.sqrt(jnp.maximum(norm_sq, 0.0))
