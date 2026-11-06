"""Numeric sparse Cholesky and triangular solves over static index plans.

Two layouts for P A Pᵀ = L Lᵀ. With a narrow band (RCM on a mesh) the matrix is block
tridiagonal and dense block kernels do the work. Otherwise every loop is a
``lax.scan`` over the columns of L whose per-step work is a fixed-size
window, gathered from the flat CSC arrays at the column pointers and masked
past the column's end; columns are bucketed by length so each bucket pads to
its own size. Either way the plans cost O(nnz(L)) memory, not one index
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
    """Columns sharing one padded window size, in elimination order.

    Attributes:
        cols: The bucket's columns.
        size: The longest column of the bucket.
        rows: The longest row of the bucket (update list), at least 1.
        reach: The longest column among the rows of the bucket's columns.
    """

    cols: np.ndarray
    size: int
    rows: int
    reach: int


class _Plans(NamedTuple):
    """Host arrays of the scans, padded so every window stays in bounds.

    Attributes:
        rowidx: Row indices of L padded with the dummy row N, of length
            nnz(L) + max_col.
        col_start: Column starts, of length N + 1; entry N is an empty dummy column.
        col_end: Column ends, of length N + 1.
        rowpos: Positions of L_jk by row j, of length nnz_off + max_row, for
            nnz_off the entries of L below the diagonal.
        rowend: End of column k for each of them, of length nnz_off + max_row.
        row_start: Row starts into ``rowpos``, of length N.
        row_end: Row ends into ``rowpos``, of length N.
        buckets: The buckets by increasing window size; a topological order.
    """

    rowidx: np.ndarray
    col_start: np.ndarray
    col_end: np.ndarray
    rowpos: np.ndarray
    rowend: np.ndarray
    row_start: np.ndarray
    row_end: np.ndarray
    buckets: tuple[_Bucket, ...]


@ft.lru_cache(maxsize=64)
def plans(sym: SymbolicCholesky) -> _Plans:
    """Padded index plans of ``sym`` (host, cached).

        level_j = ⌈log₂ m_j⌉

    with m_j the longest column in the elimination subtree of j. m never decreases
    towards the root, so the buckets in increasing order, each in elimination
    order, are a topological order of the tree (descendants first) and their
    reverse puts ancestors first. Each bucket pads to its own longest column, so a
    few long columns (the top separators of a nested-dissection or AMD ordering) do
    not set the padding of all the short ones.

    Args:
        sym: The symbolic analysis.

    Returns:
        The padded `_Plans`.

    Examples:
        A 2 × 2 matrix with a fill-free off-diagonal entry:

        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.array([1]), np.array([0]), (2, 2), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern, ordering="natural")
        >>> plans(symbolic).rowidx.tolist()
        [0, 1, 1, 2, 2]
        >>> [bucket.size for bucket in plans(symbolic).buckets]
        [2]
    """
    n, nnz, max_col, max_row = sym.n, sym.nnz, sym.max_col, sym.max_row
    counts = np.diff(sym.colptr).astype(np.int64)
    row_counts = np.diff(sym.rowptr).astype(np.int64)
    # m_j, children before their parents
    subtree = counts.copy()
    for j in range(n):
        if sym.parent[j] >= 0:
            subtree[sym.parent[j]] = max(subtree[sym.parent[j]], subtree[j])
    # ⌈log₂ m_j⌉
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
        rowidx=np.concatenate([sym.rowidx, np.full(max_col, n)]).astype(np.int32),
        col_start=np.append(sym.colptr[:-1], nnz).astype(np.int32),
        col_end=np.append(sym.colptr[1:], nnz).astype(np.int32),
        rowpos=np.concatenate([sym.rowpos, np.zeros(max_row)]).astype(np.int32),
        rowend=np.concatenate([rowend, np.zeros(max_row)]).astype(np.int32),
        row_start=sym.rowptr[:-1].astype(np.int32),
        row_end=sym.rowptr[1:].astype(np.int32),
        buckets=tuple(buckets),
    )


def _window(array: Array, start: Array, size: int) -> Array:
    """The slice ``array[start : start + size]`` under a traced ``start``.

    Args:
        array: The flat array.
        start: The traced start of the slice.
        size: The static length of the slice.

    Returns:
        The slice, of length ``size``.
    """
    return jax.lax.dynamic_slice(array, (start,), (size,))


def _column(
    p: _Plans, n: int, values: Array, j: Array, size: int
) -> tuple[Array, Array, Array, Array]:
    """Start, rows, values and mask of column ``j``'s window of ``size``.

    Args:
        p: The padded plans.
        n: The matrix size N, also the dummy row past each column's end.
        values: Flat values on L's CSC pattern, padded.
        j: The traced column index.
        size: The static window length.

    Returns:
        Tuple ``(start, rows, vals, mask)``: past the column's end ``rows`` holds
        the dummy row N, ``vals`` holds 0 and ``mask`` is ``False``.
    """
    start = jnp.asarray(p.col_start)[j]
    slots = jnp.arange(size)
    col_end = jnp.asarray(p.col_end)[j]
    col_length = col_end - start
    mask = slots < col_length
    rows = jnp.where(mask, _window(jnp.asarray(p.rowidx), start, size), n)
    vals = jnp.where(mask, _window(values, start, size), 0)
    return start, rows, vals, mask


@ft.partial(jax.jit, static_argnums=0)
def numeric_cholesky(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Numeric factorisation P A Pᵀ = L Lᵀ: banded blocks or gathered windows.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern (zero where the factor
            fills in).

    Returns:
        The values of L on its CSC pattern.

    Examples:
        For A = 4 I in two dimensions, L = 2 I:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.arange(2), np.arange(2), (2, 2), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern)
        >>> L = numeric_cholesky(symbolic, jnp.array([4.0, 4.0]))
        >>> bool(jnp.allclose(L, jnp.array([2.0, 2.0])))
        True
    """
    if sym.banded:
        return _numeric_banded(sym, a)
    return _numeric_windows(sym, a)


def _numeric_windows(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Left-looking numeric factorisation P A Pᵀ = L Lᵀ on ``sym``'s pattern.

        L_j:,j ∝ [P A Pᵀ]_j:,j − ∑ L_j:,k L_jk

    summed over k with L_jk ≠ 0, for each column j after every column it depends
    on, followed by a square root of the diagonal and a scaling. From L_jk down,
    column k's rows all lie in struct(L_:,j), so each update is a window no longer
    than column j; it is scattered into a dense workspace through L's own row
    indices (no per-flop index plan).

    Algorithm:
        1. Sweep the buckets in order, descendants first, column by column.
        2. Gather the update list of column j and accumulate its products in the
           workspace.
        3. Subtract them from column j of P A Pᵀ, then clear the workspace.
        4. Take the square root of the diagonal and scale the rest by it.

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern.

    Returns:
        The values of L on its CSC pattern.
    """
    p = plans(sym)
    n, max_col = sym.n, sym.max_col
    dtype = a.dtype
    a_pad = jnp.concatenate([a, jnp.zeros(max_col, dtype)])
    rowidx = jnp.asarray(p.rowidx)
    rowpos = jnp.asarray(p.rowpos)
    rowend = jnp.asarray(p.rowend)
    row_start = jnp.asarray(p.row_start)
    row_end = jnp.asarray(p.row_end)

    def step(carry, j, size, max_updates):
        L, workspace = carry
        slots = jnp.arange(size)

        # --- update list: the columns k < j with L_jk ≠ 0 ---
        k_pos = _window(rowpos, row_start[j], max_updates)
        k_end = _window(rowend, row_start[j], max_updates)
        update_slots = jnp.arange(max_updates)
        num_updates = row_end[j] - row_start[j]
        k_mask = update_slots < num_updates
        idx = einx.add("update, entry -> update entry", k_pos, slots)
        in_column = einx.less("update entry, update -> update entry", idx, k_end)
        mask = einx.logical_and(
            "update entry, update -> update entry", in_column, k_mask
        )
        vals = jnp.where(mask, L[idx], 0)
        rows = jnp.where(mask, rowidx[idx], n)
        # L_jk, the first entry of each update window
        first_vals = vals[:, 0]
        # L_j:,k L_jk
        products = einx.multiply(
            "update entry, update -> update entry", vals, first_vals
        )
        # w ← w + L_j:,k L_jk
        workspace = workspace.at[rows].add(products)

        # --- column j: gather the accumulated updates, then clear the workspace ---
        start, col_rows, _, col_mask = _column(p, n, L, j, size)
        a_window = _window(a_pad, start, size)
        accumulated = workspace[col_rows]
        # [P A Pᵀ]_j:,j − ∑ L_j:,k L_jk
        difference = a_window - accumulated
        col = jnp.where(col_mask, difference, 0)
        workspace = workspace.at[col_rows].set(0)
        # L_jj = √([P A Pᵀ]_jj − ∑ L_jk²)
        diagonal = jnp.sqrt(col[0])
        is_diagonal = slots == 0
        # L_ij = ([P A Pᵀ]_ij − ∑ L_ik L_jk) / L_jj
        scaled = col / diagonal
        col = jnp.where(is_diagonal, diagonal, scaled)
        old = _window(L, start, size)
        L = jax.lax.dynamic_update_slice(L, jnp.where(col_mask, col, old), (start,))
        return (L, workspace), None

    carry = (jnp.zeros(sym.nnz + max_col, dtype), jnp.zeros(n + 1, dtype))
    for bucket in p.buckets:
        body = ft.partial(step, size=bucket.size, max_updates=bucket.rows)
        carry, _ = jax.lax.scan(body, carry, jnp.asarray(bucket.cols))
    return carry[0][: sym.nnz]


@ft.partial(jax.jit, static_argnums=0)
def solve_lower(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"], b: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Forward substitution L y = b.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.
        b: Right-hand side b, in the permuted order.

    Returns:
        The solution y, in the permuted order.

    Examples:
        For L = 2 I in two dimensions:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.arange(2), np.arange(2), (2, 2), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern)
        >>> L = jnp.array([2.0, 2.0])
        >>> y = solve_lower(symbolic, L, jnp.array([1.0, 3.0]))
        >>> bool(jnp.allclose(y, jnp.array([0.5, 1.5])))
        True
    """
    if sym.banded:
        return _solve_banded(sym, L, b, upper=False)
    return _solve_lower_windows(sym, L, b)


@ft.partial(jax.jit, static_argnums=0)
def solve_upper(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"], y: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Back substitution Lᵀ x = y.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.
        y: Right-hand side y, in the permuted order.

    Returns:
        The solution x, in the permuted order.

    Examples:
        For L = 2 I in two dimensions:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.arange(2), np.arange(2), (2, 2), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern)
        >>> L = jnp.array([2.0, 2.0])
        >>> x = solve_upper(symbolic, L, jnp.array([1.0, 3.0]))
        >>> bool(jnp.allclose(x, jnp.array([0.5, 1.5])))
        True
    """
    if sym.banded:
        return _solve_banded(sym, L, y, upper=True)
    return _solve_upper_windows(sym, L, y)


def _solve_lower_windows(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"], b: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Column-oriented forward substitution L y = b, descendants first.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.
        b: Right-hand side b, in the permuted order.

    Returns:
        The solution y, in the permuted order.
    """
    p = plans(sym)
    L_pad = jnp.concatenate([L, jnp.zeros(sym.max_col, L.dtype)])

    def step(y, j, size):
        _, rows, vals, _ = _column(p, sym.n, L_pad, j, size)
        # y_j = b_j / L_jj
        yj = y[j] / vals[0]
        is_below = jnp.arange(size) > 0
        # −L_ij y_j
        elimination = -vals * yj
        below = jnp.where(is_below, elimination, 0)
        # b_i ← b_i − L_ij y_j
        y = y.at[rows].add(below)
        return y.at[j].set(yj), None

    dtype = jnp.result_type(L, b)
    # The extra entry absorbs the dummy row N past each column's end.
    y = jnp.append(b.astype(dtype), jnp.zeros((), dtype))
    for bucket in p.buckets:
        body = ft.partial(step, size=bucket.size)
        y, _ = jax.lax.scan(body, y, jnp.asarray(bucket.cols))
    return y[: sym.n]


def _solve_upper_windows(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"], y: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Column-oriented back substitution Lᵀ x = y, ancestors first.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.
        y: Right-hand side y, in the permuted order.

    Returns:
        The solution x, in the permuted order.
    """
    p = plans(sym)
    L_pad = jnp.concatenate([L, jnp.zeros(sym.max_col, L.dtype)])

    def step(x, j, size):
        _, rows, vals, _ = _column(p, sym.n, L_pad, j, size)
        is_below = jnp.arange(size) > 0
        # L_ij x_i
        products = vals * x[rows]
        below_products = jnp.where(is_below, products, 0)
        # ∑_i>j L_ij x_i
        below = jnp.sum(below_products)
        # x_j = (y_j − ∑_i>j L_ij x_i) / L_jj
        return x.at[j].set((x[j] - below) / vals[0]), None

    dtype = jnp.result_type(L, y)
    # The extra entry absorbs the dummy row N past each column's end.
    x = jnp.append(y.astype(dtype), jnp.zeros((), dtype))
    for bucket in reversed(p.buckets):
        body = ft.partial(step, size=bucket.size)
        x, _ = jax.lax.scan(body, x, jnp.asarray(bucket.cols), reverse=True)
    return x[: sym.n]


@ft.partial(jax.jit, static_argnums=0)
def lower_values(sym: SymbolicCholesky, values: Float[Array, " nnz"]) -> Array:
    """Scatter an operator's stored values onto the lower triangle of P A Pᵀ.

    Symmetric storage maps each stored entry once; general storage adds half
    of A_ij and half of A_ji, so the factored matrix is ½(A + Aᵀ).

    Args:
        sym: The symbolic analysis.
        values: The operator's stored values.

    Returns:
        The lower triangle of P A Pᵀ on L's CSC pattern.

    Examples:
        General storage of an asymmetric off-diagonal pair averages it:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.array([0, 1]), np.array([1, 0]), (2, 2)
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern, ordering="natural")
        >>> values = jnp.array([2.0, 1.0, 3.0, 2.0])
        >>> lower = lower_values(symbolic, values)
        >>> bool(jnp.allclose(lower, jnp.array([2.0, 2.0, 2.0])))
        True
    """
    weight = jnp.asarray(sym.value_weight, dtype=values.dtype)
    # w · values
    weighted = values * weight
    return jax.ops.segment_sum(
        weighted, jnp.asarray(sym.value_target), num_segments=sym.nnz
    )


# --- banded layout: P A Pᵀ and L are block tridiagonal in ``block_size`` blocks ---


def to_blocks(
    sym: SymbolicCholesky, x: Float[Array, " nnz_L"]
) -> tuple[Float[Array, "N d d"], Float[Array, "Nm1 d d"]]:
    """Lower-triangle values on L's pattern as diagonal and sub-diagonal blocks.

    The rows past N that pad the last block get a unit diagonal, which leaves
    log-determinants, solves and the selected inverse unchanged.

    Args:
        sym: The symbolic analysis.
        x: Values on L's CSC pattern.

    Returns:
        Tuple ``(diagonal, sub)`` of the block stacks.

    Examples:
        A diagonal matrix has bandwidth 0, so its blocks are 1 × 1:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.arange(2), np.arange(2), (2, 2), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern)
        >>> diagonal, sub = to_blocks(symbolic, jnp.array([2.0, 2.0]))
        >>> diagonal.shape, sub.shape
        ((2, 1, 1), (1, 1, 1))

        One block leaves an empty sub-diagonal stack:

        >>> pattern = gaussx.SparsityPattern(
        ...     np.array([0]), np.array([0]), (1, 1), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern)
        >>> diagonal, sub = to_blocks(symbolic, jnp.array([2.0]))
        >>> diagonal.shape, sub.shape
        ((1, 1, 1), (0, 1, 1))
    """
    size, blocks = sym.block_size, sym.num_blocks
    area = size * size
    empty = jnp.zeros((2 * blocks - 1) * area, x.dtype)
    scattered = empty.at[jnp.asarray(sym.block_pos)].set(x)
    flat = scattered.at[jnp.asarray(sym.pad_pos)].set(1)
    diagonal_flat = flat[: blocks * area]
    diagonal = rearrange(
        diagonal_flat, "(block row col) -> block row col", row=size, col=size
    )
    # jnp, not einx: einx raises AxisSizeError on the zero-length sub-diagonal.
    if blocks == 1:
        return diagonal, jnp.zeros((0, size, size), x.dtype)
    sub_flat = flat[blocks * area :]
    sub = rearrange(sub_flat, "(block row col) -> block row col", row=size, col=size)
    return diagonal, sub


def from_blocks(
    sym: SymbolicCholesky,
    diagonal: Float[Array, "N d d"],
    sub: Float[Array, "Nm1 d d"],
) -> Float[Array, " nnz_L"]:
    """Inverse of `to_blocks`: gather the entries on L's pattern.

    Args:
        sym: The symbolic analysis.
        diagonal: The diagonal blocks.
        sub: The sub-diagonal blocks.

    Returns:
        The values on L's CSC pattern.

    Examples:
        The round trip through `to_blocks`:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.arange(2), np.arange(2), (2, 2), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern)
        >>> values = jnp.array([2.0, 3.0])
        >>> round_trip = from_blocks(symbolic, *to_blocks(symbolic, values))
        >>> bool(jnp.allclose(round_trip, values))
        True
    """
    flat = rearrange(diagonal, "block row col -> (block row col)")
    # einx raises AxisSizeError on a zero-length sub-diagonal: skip it.
    if sub.shape[0]:
        sub_flat = rearrange(sub, "block row col -> (block row col)")
        flat = jnp.concatenate([flat, sub_flat])
    return flat[jnp.asarray(sym.block_pos)]


def _numeric_banded(
    sym: SymbolicCholesky, a: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Block-tridiagonal Cholesky (O(N b²), dense b × b blocks).

        P A Pᵀ = L Lᵀ

    Args:
        sym: The symbolic analysis.
        a: Lower triangle of P A Pᵀ on L's CSC pattern.

    Returns:
        The values of L on its CSC pattern.
    """
    # lazy import, cycle: _primitives._cholesky -> _sparse._factor -> _sparse._numeric
    from gaussx._primitives._cholesky import _cholesky_block_tridiag

    diagonal, sub = to_blocks(sym, a)
    # Only the lower triangle of each diagonal block was scattered.
    # 1 − I
    strict = 1 - jnp.eye(sym.block_size, dtype=a.dtype)
    diagonal_transpose = rearrange(diagonal, "block row col -> block col row")
    # Strictly upper triangle of each diagonal block.
    upper = einx.multiply(
        "block row col, row col -> block row col", diagonal_transpose, strict
    )
    full_diagonal = diagonal + upper
    factor = _cholesky_block_tridiag(BlockTriDiag(full_diagonal, sub))
    return from_blocks(sym, factor.diagonal, factor.sub_diagonal)


def _solve_banded(
    sym: SymbolicCholesky,
    L: Float[Array, " nnz_L"],
    b: Float[Array, " n"],
    *,
    upper: bool,
) -> Float[Array, " n"]:
    """Triangular solve L y = b or Lᵀ x = b on the block-tridiagonal layout.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.
        b: Right-hand side, in the permuted order.
        upper: Solve with Lᵀ rather than L.

    Returns:
        The solution, in the permuted order.
    """
    # lazy import, cycle: _primitives._solve -> _primitives._cholesky ->
    #   _sparse._factor -> _sparse._numeric
    from gaussx._primitives._solve import (
        _solve_lower_block_tridiag,
        _solve_upper_block_tridiag,
    )

    diagonal, sub = to_blocks(sym, L)
    factor = LowerBlockTriDiag(diagonal, sub)
    dtype = jnp.result_type(L, b)
    pad = sym.num_blocks * sym.block_size - sym.n
    rhs = jnp.concatenate([b.astype(dtype), jnp.zeros(pad, dtype)])
    if upper:
        solution = _solve_upper_block_tridiag(factor.transpose(), rhs)
    else:
        solution = _solve_lower_block_tridiag(factor, rhs)
    return solution[: sym.n]
