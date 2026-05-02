"""Structured sub-matrix extraction with dispatch on operator type."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._operators._block_diag import BlockDiag


def submatrix(
    operator: lx.AbstractLinearOperator,
    row_idx: Int[Array, " R"],
    col_idx: Int[Array, " C"],
) -> Float[Array, "R C"]:
    """Extract ``A[row_idx, col_idx]`` without forming the full matrix.

    For structured operators, exploits the structure to avoid
    materializing the full ``(N, N)`` matrix when only a sub-block is
    needed (e.g., the conditional Gaussian extracts ``Sigma_AA``,
    ``Sigma_AB``, ``Sigma_BB`` from a joint covariance).

    Currently dispatches on:

    - :class:`lineax.DiagonalLinearOperator`
    - :class:`gaussx.BlockDiag`

    Falls back to ``operator.as_matrix()[ix_(row_idx, col_idx)]`` for
    other operators.

    Args:
        operator: Linear operator A, shape ``(N, N)``.
        row_idx: Row indices, shape ``(R,)``.
        col_idx: Column indices, shape ``(C,)``.

    Returns:
        Dense sub-matrix ``A[ix_(row_idx, col_idx)]`` of shape ``(R, C)``.
    """
    n = operator.in_size()
    row_idx = _normalize_indices(row_idx, n)
    col_idx = _normalize_indices(col_idx, n)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _submatrix_diagonal(operator, row_idx, col_idx)
    if isinstance(operator, BlockDiag):
        return _submatrix_block_diag(operator, row_idx, col_idx)
    return operator.as_matrix()[jnp.ix_(row_idx, col_idx)]


def _normalize_indices(
    idx: Int[Array, " K"],
    n: int,
) -> Int[Array, " K"]:
    """Map negative indices to their canonical non-negative positions.

    Mirrors NumPy/JAX indexing: ``idx[idx < 0] += n``.
    """
    idx = jnp.asarray(idx)
    return jnp.where(idx < 0, idx + n, idx)


def _submatrix_diagonal(
    operator: lx.DiagonalLinearOperator,
    row_idx: Int[Array, " R"],
    col_idx: Int[Array, " C"],
) -> Float[Array, "R C"]:
    """Diagonal sub-matrix: nonzero only on matched index pairs."""
    d = lx.diagonal(operator)
    # M[i, j] = d[row_idx[i]] if row_idx[i] == col_idx[j] else 0
    match = row_idx[:, None] == col_idx[None, :]
    return jnp.where(match, d[row_idx][:, None], jnp.zeros((), dtype=d.dtype))


def _submatrix_block_diag(
    operator: BlockDiag,
    row_idx: Int[Array, " R"],
    col_idx: Int[Array, " C"],
) -> Float[Array, "R C"]:
    """Sub-matrix of a block-diagonal operator without forming the full matrix.

    Materializes every block once (so it can be stacked and gathered
    from), then masks off-block entries to zero. The win over the dense
    fallback is that we never materialize the full ``(N, N)`` matrix —
    only the union of the per-block matrices, plus a single padded
    stack of size ``(num_blocks, max_block_size, max_block_size)``.
    Block sizes are static (operator structure is a PyTree leaf), so
    the bookkeeping is jit-compatible.

    For workloads where the requested indices stay within a few blocks
    a future optimization could materialize only the touched blocks.
    """
    block_sizes = [op.in_size() for op in operator.operators]
    offsets_list = [0]
    for s in block_sizes[:-1]:
        offsets_list.append(offsets_list[-1] + s)
    offsets = jnp.asarray(offsets_list)
    # Static, computed in Python — keeps this jit-compatible.
    max_size = max(block_sizes)

    def _block_index(
        idx: Int[Array, " K"],
    ) -> tuple[Int[Array, " K"], Int[Array, " K"]]:
        # Assign each global index to (block_id, local_idx).
        block_ids = jnp.searchsorted(offsets, idx, side="right") - 1
        return block_ids, idx - offsets[block_ids]

    row_block, row_local = _block_index(row_idx)
    col_block, col_local = _block_index(col_idx)
    same_block = row_block[:, None] == col_block[None, :]

    # Materialize each block once; pull entries from the right block.
    block_mats = [op.as_matrix() for op in operator.operators]
    padded = jnp.stack(
        [
            jnp.pad(
                m,
                ((0, max_size - m.shape[0]), (0, max_size - m.shape[1])),
            )
            for m in block_mats
        ]
    )  # (num_blocks, max_size, max_size)

    # Pull raw entries assuming row_block == col_block; mask afterwards.
    raw = padded[
        row_block[:, None],
        row_local[:, None],
        col_local[None, :],
    ]
    return jnp.where(same_block, raw, jnp.zeros((), dtype=raw.dtype))
