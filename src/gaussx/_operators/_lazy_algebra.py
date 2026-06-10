"""Lazy algebra factories: Sum, Scaled, Product.

These build directly on lineax's native composition operators
(``AddLinearOperator``, ``MulLinearOperator``, ``ComposedLinearOperator``)
instead of re-implementing them. lineax already propagates the structural
tag queries (``is_symmetric``, ``is_diagonal``, ``is_positive_semidefinite``,
…) through its compositions, and the gaussx primitives dispatch on the
native classes — so the factories only add input validation, variadic
sums, and optional explicit tags (via ``lineax.TaggedLinearOperator``).
"""

from __future__ import annotations

import functools as ft
import operator as _op

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset


def _maybe_tag(
    operator: lx.AbstractLinearOperator,
    tags: object | frozenset[object],
) -> lx.AbstractLinearOperator:
    tags = _to_frozenset(tags)
    if tags:
        return lx.TaggedLinearOperator(operator, tags)
    return operator


def SumOperator(
    *operators: lx.AbstractLinearOperator,
    tags: object | frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """Lazy sum ``(A + B + …) v = A v + B v + …``.

    Defers materialization so that structured sub-operators keep their
    efficient matvec. All operators must have the same input and output
    sizes. Returns a (possibly tagged) chain of lineax
    ``AddLinearOperator`` compositions.

    Args:
        *operators: Two or more ``lineax.AbstractLinearOperator`` instances
            with matching shapes.
        tags: Optional explicit lineax tags for the combined operator.

    Returns:
        The lazy sum as a lineax operator.
    """
    if len(operators) < 2:
        raise ValueError("SumOperator requires at least two operators.")
    in0 = operators[0].in_size()
    out0 = operators[0].out_size()
    for i, op in enumerate(operators[1:], 1):
        if op.in_size() != in0 or op.out_size() != out0:
            raise ValueError(
                f"Shape mismatch: operator 0 has shape ({out0}, {in0}) "
                f"but operator {i} has shape ({op.out_size()}, {op.in_size()})."
            )
    return _maybe_tag(ft.reduce(_op.add, operators), tags)


def ScaledOperator(
    operator: lx.AbstractLinearOperator,
    scalar: float | Float[Array, ""],
    *,
    tags: object | frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """Lazy scalar multiply ``(c A) v = c (A v)``.

    Returns a (possibly tagged) lineax ``MulLinearOperator``.

    Args:
        operator: A ``lineax.AbstractLinearOperator``.
        scalar: A scalar multiplier.
        tags: Optional explicit lineax tags for the scaled operator.

    Returns:
        The lazy scaled operator.
    """
    scalar_array = jnp.asarray(scalar)
    if scalar_array.ndim != 0:
        msg = (
            "ScaledOperator scalar must be a rank-0 scalar, got "
            f"shape {scalar_array.shape}."
        )
        raise ValueError(msg)
    return _maybe_tag(scalar_array * operator, tags)


def ProductOperator(
    left: lx.AbstractLinearOperator,
    right: lx.AbstractLinearOperator,
    *,
    tags: object | frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """Lazy matmul ``(A B) v = A (B v)``.

    The inner dimension must match: ``left.in_size() == right.out_size()``.
    Returns a (possibly tagged) lineax ``ComposedLinearOperator``.

    Args:
        left: The left operator A.
        right: The right operator B.
        tags: Optional explicit lineax tags for the composed operator.

    Returns:
        The lazy product as a lineax operator.
    """
    if left.in_size() != right.out_size():
        raise ValueError(
            f"Inner dimension mismatch: left.in_size()={left.in_size()} "
            f"!= right.out_size()={right.out_size()}."
        )
    return _maybe_tag(left @ right, tags)
