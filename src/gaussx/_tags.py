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
    """Singleton tag with a readable repr.

    Args:
        name: The text ``repr`` shows, the tag's public name.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


kronecker_tag = _Tag("kronecker_tag")
"""Operator is a Kronecker product."""

block_diagonal_tag = _Tag("block_diagonal_tag")
"""Operator is block diagonal."""

low_rank_tag = _Tag("low_rank_tag")
"""Operator has low-rank structure (e.g. A₀ + U D Vᵀ)."""

kronecker_sum_tag = _Tag("kronecker_sum_tag")
"""Operator is a Kronecker sum A ⊕ B = A ⊗ I_b + I_a ⊗ B."""

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
    """Check whether ``operator`` carries the Kronecker tag.

    Args:
        operator: Linear operator to inspect.

    Returns:
        ``True`` if the operator carries the Kronecker tag, else ``False``.

    Examples:
        A Kronecker product A ⊗ A, then a plain dense operator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(jnp.eye(2))
        >>> gaussx.is_kronecker(gaussx.Kronecker(A, A))
        True
        >>> gaussx.is_kronecker(A)
        False
    """
    return False


@ft.singledispatch
def is_block_diagonal(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether ``operator`` carries the block-diagonal tag.

    Args:
        operator: Linear operator to inspect.

    Returns:
        ``True`` if the operator carries the block-diagonal tag, else ``False``.

    Examples:
        A block-diagonal operator diag(A, A), then a plain dense operator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(jnp.eye(2))
        >>> gaussx.is_block_diagonal(gaussx.BlockDiag(A, A))
        True
        >>> gaussx.is_block_diagonal(A)
        False
    """
    return False


@ft.singledispatch
def is_low_rank(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether ``operator`` carries the low-rank tag.

    Args:
        operator: Linear operator to inspect.

    Returns:
        ``True`` if the operator carries the low-rank tag, else ``False``.

    Examples:
        A low-rank update A₀ + U Uᵀ, then a plain dense operator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(jnp.eye(2))
        >>> gaussx.is_low_rank(gaussx.LowRankUpdate(A, jnp.ones((2, 1))))
        True
        >>> gaussx.is_low_rank(A)
        False
    """
    return False


@ft.singledispatch
def is_kronecker_sum(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether ``operator`` carries the Kronecker sum tag.

    Args:
        operator: Linear operator to inspect.

    Returns:
        ``True`` if the operator carries the Kronecker sum tag, else ``False``.

    Examples:
        A Kronecker sum A ⊕ A, then a plain dense operator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(jnp.eye(2))
        >>> gaussx.is_kronecker_sum(gaussx.KroneckerSum(A, A))
        True
        >>> gaussx.is_kronecker_sum(A)
        False
    """
    return False


@ft.singledispatch
def is_block_tridiagonal(operator: lx.AbstractLinearOperator) -> bool:
    """Check whether ``operator`` carries the block-tridiagonal tag.

    Args:
        operator: Linear operator to inspect.

    Returns:
        ``True`` if the operator carries the block-tridiagonal tag, else
        ``False``.

    Examples:
        Two 2 × 2 diagonal blocks coupled by one sub-diagonal block, then a plain
        dense operator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> blocks = jnp.stack([jnp.eye(2), jnp.eye(2)])
        >>> operator = gaussx.BlockTriDiag(blocks, jnp.zeros((1, 2, 2)))
        >>> gaussx.is_block_tridiagonal(operator)
        True
        >>> gaussx.is_block_tridiagonal(lx.MatrixLinearOperator(jnp.eye(2)))
        False
    """
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
