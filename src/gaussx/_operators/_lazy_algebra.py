"""Lazy sum, scalar multiple and product of lineax operators.

The factories build on lineax's own compositions (``AddLinearOperator``,
``MulLinearOperator``, ``ComposedLinearOperator``) instead of re-implementing
them. lineax already propagates the structural tag queries
(``is_symmetric``, ``is_diagonal``, ``is_positive_semidefinite``, …) through
its compositions, and the gaussx primitives dispatch on the native classes,
so the factories only add input validation, variadic sums and optional
explicit tags (through ``lineax.TaggedLinearOperator``). A_d, d = 1, …, D,
are summands of one shape, c is a scalar, and A B is a product whose inner
sizes match.
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
    """Wrap an operator in its explicit tags, or return it unchanged.

    Args:
        operator: The composed operator.
        tags: One tag or a collection of tags; empty leaves the operator
            unwrapped.

    Returns:
        A ``lineax.TaggedLinearOperator`` when ``tags`` is non-empty,
        otherwise ``operator`` itself.
    """
    tags = _to_frozenset(tags)
    if tags:
        return lx.TaggedLinearOperator(operator, tags)
    return operator


def sum_operator(
    *operators: lx.AbstractLinearOperator,
    tags: object | frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """Lazy sum A = A_1 + … + A_D, applied one summand at a time.

        (A_1 + … + A_D) x = A_1 x + … + A_D x

    Nothing is materialised, so structured summands keep their fast matvec.
    The result is a left-nested chain of lineax ``AddLinearOperator``s,
    tagged when ``tags`` is given.

    Args:
        *operators: The summands A_1, …, A_D, two or more lineax operators
            of the same shape.
        tags: Extra lineax tags for the sum.

    Returns:
        The lazy sum as a lineax operator.

    Raises:
        ValueError: If fewer than two summands are given, or their shapes
            differ.

    Examples:
        A diagonal plus a dense 2 × 2 operator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> B = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]))
        >>> S = gaussx.sum_operator(A, B)
        >>> bool(jnp.allclose(S.as_matrix(), A.as_matrix() + B.as_matrix()))
        True

        Explicit tags wrap the sum:

        >>> psd = lx.positive_semidefinite_tag
        >>> lx.is_positive_semidefinite(gaussx.sum_operator(A, B, tags=psd))
        True
    """
    if len(operators) < 2:
        raise ValueError("sum_operator requires at least two operators.")
    in0 = operators[0].in_size()
    out0 = operators[0].out_size()
    for i, op in enumerate(operators[1:], 1):
        if op.in_size() != in0 or op.out_size() != out0:
            raise ValueError(
                f"Shape mismatch: operator 0 has shape ({out0}, {in0}) "
                f"but operator {i} has shape ({op.out_size()}, {op.in_size()})."
            )
    # A_1 + … + A_D, left-nested
    summed = ft.reduce(_op.add, operators)
    return _maybe_tag(summed, tags)


def scaled_operator(
    operator: lx.AbstractLinearOperator,
    scalar: float | Float[Array, ""],
    *,
    tags: object | frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """Lazy scalar multiple c A, applied after the matvec.

        (c A) x = c (A x)

    The result is a lineax ``MulLinearOperator``, tagged when ``tags`` is
    given.

    Args:
        operator: The operator A.
        scalar: The rank-0 multiplier c.
        tags: Extra lineax tags for c A.

    Returns:
        The lazy scaled operator.

    Raises:
        ValueError: If ``scalar`` is not rank-0.

    Examples:
        Scale a 2 × 2 operator by 3:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]))
        >>> scaled = gaussx.scaled_operator(A, 3.0)
        >>> bool(jnp.allclose(scaled.as_matrix(), 3.0 * A.as_matrix()))
        True

        A vector multiplier is rejected:

        >>> gaussx.scaled_operator(A, jnp.ones(2))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: scaled_operator scalar must be a rank-0 scalar, got shape (2,).
    """
    scalar_array = jnp.asarray(scalar)
    if scalar_array.ndim != 0:
        msg = (
            "scaled_operator scalar must be a rank-0 scalar, got "
            f"shape {scalar_array.shape}."
        )
        raise ValueError(msg)
    # c A
    scaled = scalar_array * operator
    return _maybe_tag(scaled, tags)


def product_operator(
    left: lx.AbstractLinearOperator,
    right: lx.AbstractLinearOperator,
    *,
    tags: object | frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """Lazy product A B, applied right to left.

        (A B) x = A (B x)

    The inner sizes must match, ``left.in_size() == right.out_size()``. The
    result is a lineax ``ComposedLinearOperator``, tagged when ``tags`` is
    given.

    Args:
        left: The left factor A.
        right: The right factor B.
        tags: Extra lineax tags for A B.

    Returns:
        The lazy product as a lineax operator.

    Raises:
        ValueError: If the inner sizes differ.

    Examples:
        A 2 × 3 times a 3 × 2 operator gives a 2 × 2 one:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(jnp.ones((2, 3)))
        >>> B = lx.MatrixLinearOperator(jnp.ones((3, 2)))
        >>> P = gaussx.product_operator(A, B)
        >>> P.as_matrix().shape
        (2, 2)
        >>> bool(jnp.allclose(P.mv(jnp.ones(2)), jnp.full(2, 6.0)))
        True
    """
    if left.in_size() != right.out_size():
        raise ValueError(
            f"Inner dimension mismatch: left.in_size()={left.in_size()} "
            f"!= right.out_size()={right.out_size()}."
        )
    # A B, composed lazily
    composed = left @ right
    return _maybe_tag(composed, tags)


# Pre-gh-315 names. Plain private aliases (``gaussx.SumOperator`` & co. warn
# through the package ``__getattr__``); removed with the public aliases.
SumOperator = sum_operator
ScaledOperator = scaled_operator
ProductOperator = product_operator
