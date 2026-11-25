"""Grounding: a singular operator made positive definite by fixing a few nodes.

A positive-semidefinite R whose null space is spanned by the D_c columns of V
(a random walk, a graph Laplacian) has a positive-definite block R_FF once D_c
nodes P are fixed, if the rows V_P are invertible. Padded with zeros on P, R_FF⁻¹
is a symmetric generalised inverse of R: exact, with no ridge to choose, and on
R's own sparsity pattern.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, Int

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._sparse import SparseOperator


_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


def ground_nodes(null_space: Float[Array, "N c"]) -> Int[Array, " c"]:
    """Nodes P whose rows V_P of a null-space basis are invertible.

        x ∈ span(V) and x_P = 0   ⇒   x = 0

    The pivot rows of an LU factorisation of V with partial pivoting: one node per
    connected component of a graph, the two ends of a second-order random walk.

    Args:
        null_space: Basis V of the null space, of full column rank.

    Returns:
        The D_c node indices P, as traced integers with no gradient.
    """
    num_null = null_space.shape[1]
    if num_null == 0:
        return jnp.zeros(0, dtype=int)
    frozen_basis = jax.lax.stop_gradient(null_space)
    _, _, permutation = jax.lax.linalg.lu(frozen_basis)
    return permutation[:num_null]


def node_mask(nodes: Int[Array, " c"], size: int) -> Bool[Array, " N"]:
    """Mask of the nodes P.

    Args:
        nodes: The node indices P.
        size: The number of nodes N.

    Returns:
        ``True`` at P.
    """
    return jnp.zeros(size, dtype=bool).at[nodes].set(True)


def add_diagonal(
    operator: lx.AbstractLinearOperator, diagonal: Float[Array, " N"]
) -> lx.AbstractLinearOperator:
    """A + diag(d), keeping a sparse or block-tridiagonal structure.

    Args:
        operator: The operator A; anything but a (tagged) `SparseOperator` or
            `BlockTriDiag` is densified.
        diagonal: The diagonal d.

    Returns:
        A + diag(d), tagged symmetric positive semidefinite.
    """
    operator = _untagged(operator)
    if isinstance(operator, SparseOperator):
        return operator.add_diagonal(diagonal, tags=_PSD)
    if isinstance(operator, BlockTriDiag):
        block_size = operator.diagonal.shape[-1]
        # d, one row per diagonal block
        blocks = einx.id("(block row) -> block row", diagonal, row=block_size)
        eye = jnp.eye(block_size, dtype=diagonal.dtype)
        # diag(d) as a stack of diagonal blocks
        added = einx.multiply("block row, row col -> block row col", blocks, eye)
        return BlockTriDiag(
            operator.diagonal + added,
            operator.sub_diagonal,
            symmetric=operator.symmetric,
            tags=_PSD,
        )
    return lx.MatrixLinearOperator(operator.as_matrix() + jnp.diag(diagonal), _PSD)


def ground(
    operator: lx.AbstractLinearOperator, nodes: Int[Array, " c"]
) -> lx.AbstractLinearOperator:
    """R with the rows and columns of the nodes P replaced by a unit diagonal.

        [R_FF  0]
        [0     I]   on (F, P), F the other nodes

    Positive definite when P determines the null space of R (`ground_nodes`), with
    determinant det R_FF.

    Args:
        operator: The structure R; anything but a (tagged) `SparseOperator` or
            `BlockTriDiag` is densified.
        nodes: The node indices P.

    Returns:
        The grounded operator, tagged symmetric positive semidefinite.
    """
    operator = _untagged(operator)
    fixed = node_mask(nodes, operator.in_size())
    if isinstance(operator, SparseOperator):
        pattern = operator.pattern
        touched = fixed[pattern.rows] | fixed[pattern.cols]
        # [R]_ij with i, j ∈ F, zero elsewhere
        kept = jnp.where(touched, 0.0, operator.values)
        kept_operator = SparseOperator(kept, pattern, tags=operator.tags)
        return add_diagonal(kept_operator, fixed.astype(kept.dtype))
    if isinstance(operator, BlockTriDiag):
        block_size = operator.diagonal.shape[-1]
        blocks = einx.id("(block row) -> block row", fixed, row=block_size)
        touched = einx.logical_or(
            "block row, block col -> block row col", blocks, blocks
        )
        diagonal = jnp.where(touched, 0.0, operator.diagonal)
        sub_diagonal = operator.sub_diagonal
        if sub_diagonal.shape[0] > 0:  # einx rejects an empty block axis
            # row t + 1 of the structure, column t
            touched = einx.logical_or(
                "block row, block col -> block row col", blocks[1:], blocks[:-1]
            )
            sub_diagonal = jnp.where(touched, 0.0, sub_diagonal)
        kept_operator = BlockTriDiag(
            diagonal, sub_diagonal, symmetric=operator.symmetric
        )
        return add_diagonal(kept_operator, fixed.astype(diagonal.dtype))
    matrix = operator.as_matrix()
    touched = einx.logical_or("row, col -> row col", fixed, fixed)
    kept_matrix = jnp.where(touched, 0.0, matrix)
    return lx.MatrixLinearOperator(
        kept_matrix + jnp.diag(fixed.astype(matrix.dtype)), _PSD
    )


def _untagged(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Strip every `lineax.TaggedLinearOperator` wrapper from an operator."""
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator
