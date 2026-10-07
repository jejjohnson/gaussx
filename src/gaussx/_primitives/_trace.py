"""Structured trace with dispatch on operator type."""

from __future__ import annotations

import functools as ft
from typing import Literal

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._diagonalised import DiagonalisedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sum_kronecker import SumOfKroneckers
from gaussx._operators._toeplitz import Toeplitz
from gaussx._primitives._samplers import SamplerName, resolve_sampler, split_keys


def trace(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName | None = None,
    algorithm: Literal["hutchinson", "xtrace"] = "hutchinson",
) -> Float[Array, ""]:
    """Compute the trace of an operator.

    When ``stochastic=True``, uses a matfree stochastic estimator —
    only requires matvec access, no materialization.

    Args:
        operator: A square linear operator.
        stochastic: If ``True``, use stochastic trace estimation.
        num_probes: Number of probe vectors for stochastic mode.
        key: PRNG key for stochastic mode.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``). Defaults to ``"signs"`` for Hutchinson and
            ``"sphere"`` for XTrace (which requires a rotationally
            invariant distribution).
        algorithm: ``"hutchinson"`` (Monte-Carlo) or ``"xtrace"``
            (leave-one-out, Epperly et al. 2024 — much lower variance
            for the same number of matvecs).

    Returns:
        Scalar trace value (exact or estimated).
    """

    # Every recursive call forwards the estimator options, so a wrapped or
    # structured matrix-free operator is never materialised (gh-320).
    def rec(op: lx.AbstractLinearOperator, k: jax.Array | None = key) -> Array:
        return trace(
            op,
            stochastic=stochastic,
            num_probes=num_probes,
            key=k,
            sampler=sampler,
            algorithm=algorithm,
        )

    def rec_all(ops) -> list[Array]:
        ops = tuple(ops)
        keys = split_keys(key, len(ops))
        return [rec(op, k) for op, k in zip(ops, keys, strict=True)]

    if isinstance(operator, lx.IdentityLinearOperator):
        return jnp.asarray(operator.in_size(), dtype=operator.in_structure().dtype)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return jnp.sum(lx.diagonal(operator))
    if isinstance(operator, DiagonalisedOperator):
        total = jnp.sum(operator.eigenvalues)
        return jnp.real(total) if operator.real_output else total
    if isinstance(operator, Toeplitz):
        # Constant diagonal c[0] (gh-373).
        return operator.in_size() * operator.column[0]
    if isinstance(operator, BlockDiag):
        return ft.reduce(jnp.add, rec_all(operator.operators))
    if isinstance(operator, Kronecker):
        # trace(A ⊗ B) = trace(A) · trace(B).
        return ft.reduce(jnp.multiply, rec_all(operator.operators))
    if isinstance(operator, BlockTriDiag):
        return _trace_block_tridiag(operator)
    if isinstance(operator, LowRankUpdate):
        return _trace_low_rank(operator, rec(operator.base))
    if isinstance(operator, KroneckerSum):
        # trace(A ⊕ B) = n_b · trace(A) + n_a · trace(B).
        trace_a, trace_b = rec_all((operator.A, operator.B))
        return operator.B.out_size() * trace_a + operator.A.out_size() * trace_b
    if isinstance(operator, SumOfKroneckers):
        return ft.reduce(jnp.add, rec_all(operator.operators))
    if isinstance(operator, lx.TaggedLinearOperator):
        return rec(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        first, second = rec_all((operator.operator1, operator.operator2))
        return first + second
    if isinstance(operator, lx.MulLinearOperator):
        return operator.scalar * rec(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        return rec(operator.operator) / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        return -rec(operator.operator)
    if stochastic:
        return _trace_stochastic(operator, num_probes, key, sampler, algorithm)
    return jnp.trace(operator.as_matrix())


def _trace_block_tridiag(operator: BlockTriDiag) -> Float[Array, ""]:
    """trace of block-tridiagonal = sum of traces of diagonal blocks."""
    return jnp.sum(jax.vmap(jnp.trace)(operator.diagonal))


def _trace_low_rank(
    operator: LowRankUpdate, base_trace: Float[Array, ""]
) -> Float[Array, ""]:
    """trace(L + U diag(d) V^T) = trace(L) + sum_k d[k] (V[:, k] . U[:, k])."""
    update = jnp.sum(operator.U * operator.d * operator.V)
    return base_trace + update


def _trace_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array | None,
    sampler: SamplerName | None,
    algorithm: Literal["hutchinson", "xtrace"],
) -> Float[Array, ""]:
    """Stochastic trace estimator via matfree (Hutchinson or XTrace)."""
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    if algorithm == "xtrace":
        # XTrace requires rotationally invariant probes.
        if sampler == "signs":
            raise ValueError(
                "XTrace requires a rotationally invariant sampler; "
                'use sampler="normal" or sampler="sphere".'
            )
        probe_fn = resolve_sampler(
            sampler or "sphere", n, num_probes, dtype=operator.in_structure().dtype
        )
        integrand = matfree.stochtrace.leave_one_out_xtrace()
        estimate = matfree.stochtrace.estimator_leave_one_out(integrand, probe_fn)
    elif algorithm == "hutchinson":
        probe_fn = resolve_sampler(
            sampler or "signs", n, num_probes, dtype=operator.in_structure().dtype
        )
        integrand = matfree.stochtrace.monte_carlo_trace()
        estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    else:
        raise ValueError(
            f'Unknown algorithm {algorithm!r}; expected "hutchinson" or "xtrace".'
        )
    return estimate(operator.mv, key)


def trace_and_diag(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName = "signs",
) -> tuple[Float[Array, ""], Float[Array, " n"]]:
    """Jointly estimate the trace and diagonal from one probe pass.

    Halves the matvec budget relative to calling
    ``trace(..., stochastic=True)`` and ``diag(..., stochastic=True)``
    separately — both statistics are accumulated from the same
    ``A @ probe`` products.

    Args:
        operator: A square linear operator.
        num_probes: Number of probe vectors.
        key: PRNG key. If ``None``, uses ``jax.random.PRNGKey(0)``.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).

    Returns:
        Tuple ``(trace_estimate, diagonal_estimate)``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    probe_fn = resolve_sampler(
        sampler, n, num_probes, dtype=operator.in_structure().dtype
    )
    integrand = matfree.stochtrace.monte_carlo_trace_and_diagonal()
    estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    result = estimate(operator.mv, key)
    return result["trace"], result["diagonal"]
