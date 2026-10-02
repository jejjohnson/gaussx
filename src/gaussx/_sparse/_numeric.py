"""Numeric sparse Cholesky and triangular solves over static index plans.

Two layouts. With a narrow band (RCM on a mesh) the matrix is block
tridiagonal and dense block kernels do the work. Otherwise every loop is a
``lax.scan`` over the columns of ``L`` whose per-step work is a fixed-size
window, gathered from the flat CSC arrays at the column pointers and masked
past the column's end; columns are bucketed by length so each bucket pads to
its own size. Either way the plans cost ``O(nnz(L))`` memory, not one index
per flop, and values are the only traced inputs: everything ``jit``s, and
``vmap``s over values.
"""

from __future__ import annotations

import functools as ft
from typing import NamedTuple

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._operators._block_tridiag import BlockTriDiag, LowerBlockTriDiag
from gaussx._sparse._symbolic import SymbolicCholesky


class _Bucket(NamedTuple):
    """Columns sharing one padded window size, in elimination order."""

    cols: np.ndarray
    size: int  # longest column of the bucket
    rows: int  # longest row of the bucket (update list), at least 1
    reach: int  # longest column among the rows of the bucket's columns


class _Plans(NamedTuple):
    """Host arrays of the scans, padded so every window stays in bounds."""

    rowidx: np.ndarray  # (nnz + max_col,), padded with the dummy row n
    col_start: np.ndarray  # (n + 1,), entry n is an empty dummy column
    col_end: np.ndarray  # (n + 1,)
    rowpos: np.ndarray  # (nnz_off + max_row,), positions of L[j, k] by row j
    rowend: np.ndarray  # (nnz_off + max_row,), end of column k for each of them
    row_start: np.ndarray  # (n,)
    row_end: np.ndarray  # (n,)
    buckets: tuple[_Bucket, ...]  # increasing window size; a topological order


@ft.lru_cache(maxsize=64)
def plans(sym: SymbolicCholesky) -> _Plans:
    """Padded index plans of ``sym`` (host, cached).

    Columns are bucketed by ``⌈log₂ m_j⌉``, with ``m_j`` the longest column in
    the elimination subtree of ``j``. ``m`` never decreases towards the root,
    so the buckets in increasing order, each in elimination order, are a
    topological order of the tree (descendants first) and their reverse puts
    ancestors first. Each bucket pads to its own longest column, so a few
    long columns (the top separators of a nested-dissection or AMD ordering)
    do not set the padding of all the short ones.
    """
    n, nnz, c, r = sym.n, sym.nnz, sym.max_col, sym.max_row
    counts = np.diff(sym.colptr).astype(np.int64)
    row_counts = np.diff(sym.rowptr).astype(np.int64)
    subtree = counts.copy()
    for j in range(n):
        if sym.parent[j] >= 0:
            subtree[sym.parent[j]] = max(subtree[sym.parent[j]], subtree[j])
    level = np.ceil(np.log2(np.maximum(subtree, 1))).astype(np.int64)
    reach = np.maximum.reduceat(counts[sym.rowidx], sym.colptr[:-1]) if n else counts
    buckets = []
    for value in np.unique(level):
        cols = np.flatnonzero(level == value)
        buckets.append(
            _Bucket(
                cols=cols.astype(np.int32),
                size=int(counts[cols].max()),
                rows=int(max(row_counts[cols].max(), 1)),
                reach=int(reach[cols].max()),
            )
        )
    rowend = sym.colptr[sym.colidx[sym.rowpos] + 1]
    return _Plans(
        rowidx=np.concatenate([sym.rowidx, np.full(c, n)]).astype(np.int32),
        col_start=np.append(sym.colptr[:-1], nnz).astype(np.int32),
        col_end=np.append(sym.colptr[1:], nnz).astype(np.int32),
        rowpos=np.concatenate([sym.rowpos, np.zeros(r)]).astype(np.int32),
        rowend=np.concatenate([rowend, np.zeros(r)]).astype(np.int32),
        row_start=sym.rowptr[:-1].astype(np.int32),
        row_end=sym.rowptr[1:].astype(np.int32),
        buckets=tuple(buckets),
    )


def _window(array: Array, start: Array, size: int) -> Array:
    return jax.lax.dynamic_slice(array, (start,), (size,))


def _column(
    p: _Plans, n: int, values: Array, j: Array, size: int
) -> tuple[Array, Array, Array, Array]:
    """Start, rows, values and mask of column ``j``'s window of ``size``."""
    start = jnp.asarray(p.col_start)[j]
    mask = jnp.arange(size) < jnp.asarray(p.col_end)[j] - start
    rows = jnp.where(mask, _window(jnp.asarray(p.rowidx), start, size), n)
    vals = jnp.where(mask, _window(values, start, size), 0)
    return start, rows, vals, mask


@ft.partial(jax.jit, static_argnums=0)
def numeric_cholesky(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Numeric factorisation ``A = L Lᵀ``: banded blocks or gathered windows.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of the permuted matrix on ``L``'s CSC pattern (zero
            where the factor fills in), shape ``(nnz_L,)``.

    Returns:
        The values of ``L`` on its CSC pattern, shape ``(nnz_L,)``.
    """
    if sym.banded:
        return _numeric_banded(sym, a)
    return _numeric_windows(sym, a)


def _numeric_windows(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    r"""Left-looking numeric factorisation ``A = L Lᵀ`` on ``sym``'s pattern.

    For each column ``j``, after every column it depends on,

    $$
    L_{j:,j} \propto A_{j:,j} - \sum_{k:\,L_{jk}\neq 0} L_{j:,k}\,L_{jk},
    $$

    followed by a square root of the diagonal and a scaling. From ``L_jk``
    down, column ``k``'s rows all lie in ``struct(L_{:,j})``, so each update is
    a window no longer than column ``j``; it is scattered into a dense
    workspace through ``L``'s own row indices (no per-flop index plan).
    """
    p = plans(sym)
    n, c = sym.n, sym.max_col
    dtype = a.dtype
    a_pad = jnp.concatenate([a, jnp.zeros(c, dtype)])
    rowidx = jnp.asarray(p.rowidx)
    rowpos = jnp.asarray(p.rowpos)
    rowend = jnp.asarray(p.rowend)
    row_start = jnp.asarray(p.row_start)
    row_end = jnp.asarray(p.row_end)

    def step(carry, j, size, r):
        L, w = carry
        t = jnp.arange(size)
        # Update list: the columns k < j with L[j, k] != 0.
        k_pos = _window(rowpos, row_start[j], r)
        k_end = _window(rowend, row_start[j], r)
        k_mask = jnp.arange(r) < row_end[j] - row_start[j]
        idx = einx.add("r, c -> r c", k_pos, t)
        mask = einx.logical_and(
            "r c, r -> r c", einx.less("r c, r -> r c", idx, k_end), k_mask
        )
        vals = jnp.where(mask, L[idx], 0)
        rows = jnp.where(mask, rowidx[idx], n)
        w = w.at[rows].add(einx.multiply("r c, r -> r c", vals, vals[:, 0]))
        # Column j: gather the accumulated updates, then clear the workspace.
        start, col_rows, _, col_mask = _column(p, n, L, j, size)
        col = jnp.where(col_mask, _window(a_pad, start, size) - w[col_rows], 0)
        w = w.at[col_rows].set(0)
        d = jnp.sqrt(col[0])
        col = jnp.where(t == 0, d, col / d)
        old = _window(L, start, size)
        L = jax.lax.dynamic_update_slice(L, jnp.where(col_mask, col, old), (start,))
        return (L, w), None

    carry = (jnp.zeros(sym.nnz + c, dtype), jnp.zeros(n + 1, dtype))
    for bucket in p.buckets:
        body = ft.partial(step, size=bucket.size, r=bucket.rows)
        carry, _ = jax.lax.scan(body, carry, jnp.asarray(bucket.cols))
    return carry[0][: sym.nnz]


@ft.partial(jax.jit, static_argnums=0)
def solve_lower(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"], b: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Forward substitution ``L y = b``."""
    if sym.banded:
        return _solve_banded(sym, L, b, upper=False)
    return _solve_lower_windows(sym, L, b)


@ft.partial(jax.jit, static_argnums=0)
def solve_upper(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"], y: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Back substitution ``Lᵀ x = y``."""
    if sym.banded:
        return _solve_banded(sym, L, y, upper=True)
    return _solve_upper_windows(sym, L, y)


def _solve_lower_windows(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"], b: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Column-oriented forward substitution, descendants first."""
    p = plans(sym)
    L_pad = jnp.concatenate([L, jnp.zeros(sym.max_col, L.dtype)])

    def step(y, j, size):
        _, rows, vals, _ = _column(p, sym.n, L_pad, j, size)
        yj = y[j] / vals[0]
        y = y.at[rows].add(jnp.where(jnp.arange(size) > 0, -vals * yj, 0))
        return y.at[j].set(yj), None

    dtype = jnp.result_type(L, b)
    y = jnp.append(b.astype(dtype), jnp.zeros((), dtype))
    for bucket in p.buckets:
        body = ft.partial(step, size=bucket.size)
        y, _ = jax.lax.scan(body, y, jnp.asarray(bucket.cols))
    return y[: sym.n]


def _solve_upper_windows(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"], y: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Column-oriented back substitution, ancestors first."""
    p = plans(sym)
    L_pad = jnp.concatenate([L, jnp.zeros(sym.max_col, L.dtype)])

    def step(x, j, size):
        _, rows, vals, _ = _column(p, sym.n, L_pad, j, size)
        below = jnp.sum(jnp.where(jnp.arange(size) > 0, vals * x[rows], 0))
        return x.at[j].set((x[j] - below) / vals[0]), None

    dtype = jnp.result_type(L, y)
    x = jnp.append(y.astype(dtype), jnp.zeros((), dtype))
    for bucket in reversed(p.buckets):
        body = ft.partial(step, size=bucket.size)
        x, _ = jax.lax.scan(body, x, jnp.asarray(bucket.cols), reverse=True)
    return x[: sym.n]


@ft.partial(jax.jit, static_argnums=0)
def lower_values(sym: SymbolicCholesky, values: Float[Array, " nnz"]) -> Array:
    """Scatter an operator's stored values onto the lower triangle of ``P Q Pᵀ``.

    Symmetric storage maps each stored entry once; general storage adds half
    of ``Q_ij`` and half of ``Q_ji``, so the factored matrix is ``½(Q + Qᵀ)``.
    """
    weight = jnp.asarray(sym.value_weight, dtype=values.dtype)
    return jax.ops.segment_sum(
        values * weight, jnp.asarray(sym.value_target), num_segments=sym.nnz
    )


# ---------------------------------------------------------------------------
# Banded layout: A and L are block tridiagonal in ``block_size`` blocks.
# ---------------------------------------------------------------------------


def to_blocks(
    sym: SymbolicCholesky, x: Float[Array, " nnz_L"]
) -> tuple[Float[Array, "N d d"], Float[Array, "Nm1 d d"]]:
    """Lower-triangle values on ``L``'s pattern as diagonal and sub-diagonal blocks.

    The rows past ``n`` that pad the last block get a unit diagonal, which
    leaves log-determinants, solves and the selected inverse unchanged.
    """
    size, blocks = sym.block_size, sym.num_blocks
    area = size * size
    flat = (
        jnp.zeros((2 * blocks - 1) * area, x.dtype)
        .at[jnp.asarray(sym.block_pos)]
        .set(x)
        .at[jnp.asarray(sym.pad_pos)]
        .set(1)
    )
    diagonal = rearrange(flat[: blocks * area], "(N i j) -> N i j", i=size, j=size)
    if blocks == 1:
        return diagonal, jnp.zeros((0, size, size), x.dtype)
    sub = rearrange(flat[blocks * area :], "(M i j) -> M i j", i=size, j=size)
    return diagonal, sub


def from_blocks(
    sym: SymbolicCholesky,
    diagonal: Float[Array, "N d d"],
    sub: Float[Array, "Nm1 d d"],
) -> Float[Array, " nnz_L"]:
    """Inverse of `to_blocks`: gather the entries on ``L``'s pattern."""
    flat = rearrange(diagonal, "N i j -> (N i j)")
    if sub.shape[0]:
        flat = jnp.concatenate([flat, rearrange(sub, "M i j -> (M i j)")])
    return flat[jnp.asarray(sym.block_pos)]


def _numeric_banded(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Block-tridiagonal Cholesky (``O(n b²)``, dense ``b × b`` blocks)."""
    from gaussx._primitives._cholesky import _cholesky_block_tridiag

    diagonal, sub = to_blocks(sym, a)
    # Only the lower triangle of each diagonal block was scattered.
    strict = 1 - jnp.eye(sym.block_size, dtype=a.dtype)
    upper = einx.multiply(
        "N i j, i j -> N i j", rearrange(diagonal, "N i j -> N j i"), strict
    )
    factor = _cholesky_block_tridiag(BlockTriDiag(diagonal + upper, sub))
    return from_blocks(sym, factor.diagonal, factor.sub_diagonal)


def _solve_banded(
    sym: SymbolicCholesky,
    L: Float[Array, " nnz_L"],
    b: Float[Array, " n"],
    *,
    upper: bool,
) -> Float[Array, " n"]:
    from gaussx._primitives._solve import (
        _solve_lower_block_tridiag,
        _solve_upper_block_tridiag,
    )

    factor = LowerBlockTriDiag(*to_blocks(sym, L))
    dtype = jnp.result_type(L, b)
    pad = sym.num_blocks * sym.block_size - sym.n
    rhs = jnp.concatenate([b.astype(dtype), jnp.zeros(pad, dtype)])
    if upper:
        x = _solve_upper_block_tridiag(factor.transpose(), rhs)
    else:
        x = _solve_lower_block_tridiag(factor, rhs)
    return x[: sym.n]
