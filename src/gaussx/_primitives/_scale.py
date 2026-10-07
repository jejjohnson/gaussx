"""Scale a factor by a scalar without losing its structured type (gh-326)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import lineax as lx
from jax.core import Tracer
from jaxtyping import Array, ArrayLike

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import LowerBlockTriDiag
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._sparse import SparseOperator


def scale_factor(
    operator: lx.AbstractLinearOperator, scale: ArrayLike | Array
) -> lx.AbstractLinearOperator:
    """``scale · operator`` that keeps the operator's structured type.

    Used by `cholesky` / `sqrt` on ``c · A`` and ``A / c``: they factor ``A``
    and scale the factor by ``√c`` (or ``1/√c``). Downstream code dispatches
    on the factor's type (``solve(L, ·)``, ``logdet(L)``), so the scale is
    folded into one Kronecker factor, every block of a `BlockDiag`, the
    entries of a diagonal / dense / block-bidiagonal factor or the
    eigenvalues of a `DiagonalizedOperator`; anything else becomes a lineax
    ``MulLinearOperator``.
    """
    scale = jnp.asarray(scale)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.DiagonalLinearOperator(scale * lx.diagonal(operator))
    if isinstance(operator, lx.IdentityLinearOperator):
        dtype = operator.in_structure().dtype
        ones = jnp.ones(operator.in_size(), dtype=dtype)
        return lx.DiagonalLinearOperator(scale * ones)
    if isinstance(operator, lx.MatrixLinearOperator):
        return lx.MatrixLinearOperator(scale * operator.matrix, operator.tags)
    if isinstance(operator, lx.TaggedLinearOperator):
        return lx.TaggedLinearOperator(
            scale_factor(operator.operator, scale), operator.tags
        )
    if isinstance(operator, Kronecker):
        first, *rest = operator.operators
        return Kronecker(scale_factor(first, scale), *rest)
    if isinstance(operator, BlockDiag):
        return BlockDiag(*(scale_factor(op, scale) for op in operator.operators))
    if isinstance(operator, LowerBlockTriDiag):
        return LowerBlockTriDiag(
            scale * operator.diagonal,
            scale * operator.sub_diagonal,
            tags=operator.tags,
        )
    if isinstance(operator, DiagonalizedOperator):
        return operator.with_eigenvalues(scale * operator.eigenvalues)
    return scale * operator


def split_scalar(
    operator: lx.AbstractLinearOperator,
) -> tuple[ArrayLike | Array, lx.AbstractLinearOperator]:
    """Peel nested ``Mul`` / ``Div`` / ``Neg`` wrappers: ``A = c · base``."""
    scale: ArrayLike | Array = 1.0
    while True:
        if isinstance(operator, lx.MulLinearOperator):
            scale = scale * operator.scalar
        elif isinstance(operator, lx.DivLinearOperator):
            scale = scale / operator.scalar
        elif isinstance(operator, lx.NegLinearOperator):
            scale = -scale
        else:
            return scale, operator
        operator = operator.operator


def _concretely_negative(scale: ArrayLike | Array) -> bool:
    if isinstance(scale, Tracer):
        return False
    return bool(jnp.real(jnp.asarray(scale)) < 0)


def scaled_root(
    operator: lx.MulLinearOperator | lx.DivLinearOperator | lx.NegLinearOperator,
    root: Callable[[lx.AbstractLinearOperator], Any],
    dense: Callable[[lx.AbstractLinearOperator], lx.AbstractLinearOperator],
    name: str,
) -> Any:
    """``root(c · A) = √c · root(A)`` for `cholesky` / `sqrt` (gh-326).

    The nested scalar wrappers are peeled into one ``c`` and the root of the
    base ``A`` is scaled by ``√c`` with `scale_factor`, so its structured
    type survives. A concretely negative ``c`` (e.g. ``-A``) is not PSD and
    raises — unless ``A`` is tagged negative semi-definite, which keeps the
    previous dense route. A traced negative ``c`` gives NaN, as for any
    non-PSD input.
    """
    scale, base = split_scalar(operator)
    if _concretely_negative(scale):
        if lx.is_negative_semidefinite(base):
            return dense(operator)
        raise ValueError(
            f"{name} requires a positive semi-definite operator; got a "
            f"negative multiple of a {type(base).__name__}."
        )
    if _contains_sparse(base):
        # A sparse Cholesky is a permuted `SparseCholeskyFactor`, not an
        # operator, so it can neither be scaled nor sit in a BlockDiag.
        return dense(operator)
    return scale_factor(root(base), jnp.sqrt(scale))


def _contains_sparse(operator: lx.AbstractLinearOperator) -> bool:

    if isinstance(operator, SparseOperator):
        return True
    if isinstance(operator, lx.TaggedLinearOperator):
        return _contains_sparse(operator.operator)
    if isinstance(operator, BlockDiag | Kronecker):
        return any(_contains_sparse(op) for op in operator.operators)
    return False
