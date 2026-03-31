"""Structural tags for gaussx operators.

Extends lineax's tag system with tags for Kronecker, block diagonal,
and low-rank structure. These tags enable isinstance-based dispatch
in primitives like ``solve``, ``logdet``, and ``cholesky``.
"""

from __future__ import annotations

import functools as ft

import lineax as lx


# ---------------------------------------------------------------------------
# GaussX tags — same pattern as lineax._tags._HasRepr
# ---------------------------------------------------------------------------


class _Tag:
    """Singleton tag with a readable repr."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


kronecker_tag = _Tag("kronecker_tag")
"""Operator is a Kronecker product."""

block_diagonal_tag = _Tag("block_diagonal_tag")
"""Operator is block diagonal."""

low_rank_tag = _Tag("low_rank_tag")
"""Operator has low-rank structure (e.g. L + U D V^T)."""

kronecker_sum_tag = _Tag("kronecker_sum_tag")
"""Operator is a Kronecker sum A (+) B = A (x) I_b + I_a (x) B."""

block_tridiagonal_tag = _Tag("block_tridiagonal_tag")
"""Operator is block tridiagonal."""


# ---------------------------------------------------------------------------
# Re-export lineax tags for unified import
# ---------------------------------------------------------------------------

symmetric_tag = lx.symmetric_tag
diagonal_tag = lx.diagonal_tag
tridiagonal_tag = lx.tridiagonal_tag
unit_diagonal_tag = lx.unit_diagonal_tag
lower_triangular_tag = lx.lower_triangular_tag
upper_triangular_tag = lx.upper_triangular_tag
positive_semidefinite_tag = lx.positive_semidefinite_tag
negative_semidefinite_tag = lx.negative_semidefinite_tag


# ---------------------------------------------------------------------------
# GaussX tag queries
# ---------------------------------------------------------------------------


@ft.singledispatch
def is_kronecker(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether *operator* carries the Kronecker tag."""
    return False


@ft.singledispatch
def is_block_diagonal(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether *operator* carries the block-diagonal tag."""
    return False


@ft.singledispatch
def is_low_rank(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether *operator* carries the low-rank tag."""
    return False


@ft.singledispatch
def is_kronecker_sum(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether *operator* carries the Kronecker sum tag."""
    return False


@ft.singledispatch
def is_block_tridiagonal(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether *operator* carries the block-tridiagonal tag."""
    return False


# ---------------------------------------------------------------------------
# Re-export lineax tag queries for unified import
# ---------------------------------------------------------------------------

is_symmetric = lx.is_symmetric
is_diagonal = lx.is_diagonal
is_tridiagonal = lx.is_tridiagonal
is_lower_triangular = lx.is_lower_triangular
is_upper_triangular = lx.is_upper_triangular
is_positive_semidefinite = lx.is_positive_semidefinite
is_negative_semidefinite = lx.is_negative_semidefinite
