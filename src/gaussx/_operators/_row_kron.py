"""Row-wise Kronecker (face-splitting) product of two sparse operators."""

from __future__ import annotations

import functools as ft

import jax.numpy as jnp
import numpy as np

from gaussx._operators._sparse import (
    _PLAN_CACHE_SIZE,
    SparseOperator,
    SparsityPattern,
    _positions,
)


def row_kron(left: SparseOperator, right: SparseOperator) -> SparseOperator:
    """Row-wise Kronecker (face-splitting) product A • B of two sparse operators.

        [A • B]_m,· = [A]_m,· ⊗ [B]_m,·,   (A • B)(x ⊗ y) = (A x) ⊙ (B y)

    Row m reads the product field x ⊗ y where row m of A reads x and row m of B
    reads y: the observation operator of a Kronecker-structured field at scattered
    points. `Kronecker` is the full product of every row with every row.

    The pattern is built on the host, once per pair of patterns, and holds
    ∑_m nnz([A]_m,·) nnz([B]_m,·) entries, plus the diagonal of a square product. The
    values may be traced.

    Args:
        left: A, shape ``(M, P)``.
        right: B, shape ``(M, Q)``.

    Returns:
        A • B as a general (non-symmetric) `SparseOperator`, shape ``(M, P Q)``.
        Column ``p * Q + q`` is row-major, as in `Kronecker` and ``jnp.kron``.

    Raises:
        ValueError: If A and B have different numbers of rows.

    Examples:
        Three observations of a field on a 2 × 3 grid, each at its own node of both
        axes:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> rows = np.arange(3)
        >>> space = gaussx.SparseOperator.from_coo(rows, [0, 1, 1], jnp.ones(3), (3, 2))
        >>> time = gaussx.SparseOperator.from_coo(rows, [2, 0, 1], jnp.ones(3), (3, 3))
        >>> projector = gaussx.row_kron(space, time)
        >>> projector.out_size(), projector.in_size()
        (3, 6)
        >>> projector.mv(jnp.arange(6.0)).tolist()
        [2.0, 3.0, 4.0]

        The operands must have one row per observation:

        >>> gaussx.row_kron(space, space.T)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: row_kron needs operands with the same number of rows, got 3 and 2.
    """
    if left.out_size() != right.out_size():
        raise ValueError(
            "row_kron needs operands with the same number of rows, got "
            f"{left.out_size()} and {right.out_size()}."
        )
    pattern, entry_left, entry_right, position = _row_kron_plan(
        left.pattern, right.pattern
    )
    left_values, right_values = left._full_values(), right._full_values()
    # [A]_m,p [B]_m,q, one per pair of entries of row m
    products = left_values[entry_left] * right_values[entry_right]
    # zero on a stored diagonal no pair reaches
    values = jnp.zeros(pattern.nnz, products.dtype).at[position].set(products)
    return SparseOperator(values, pattern)


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _row_kron_plan(
    left: SparsityPattern, right: SparsityPattern
) -> tuple[SparsityPattern, np.ndarray, np.ndarray, np.ndarray]:
    """Pattern of A • B and the pair of entries behind each of its entries.

    Args:
        left: Pattern of A, M × P.
        right: Pattern of B, M × Q.

    Returns:
        Tuple ``(pattern, entry_left, entry_right, position)``: the pattern of the
        product; for every pair of entries of one row, their positions in the full
        (row-major) layouts of A and B; and the pair's position in the product.
    """
    num_rows, num_right = right.shape
    left_rows, left_cols, _ = left._full
    right_rows, right_cols, _ = right._full
    counts = np.bincount(right_rows, minlength=num_rows)
    starts = np.cumsum(counts) - counts
    # each entry of A pairs with every entry of B in its row
    reps = counts[left_rows]
    entry_left = np.repeat(np.arange(left_rows.size), reps)
    offset = np.arange(entry_left.size) - np.repeat(np.cumsum(reps) - reps, reps)
    entry_right = starts[left_rows[entry_left]] + offset
    rows = left_rows[entry_left]
    cols = left_cols[entry_left].astype(np.int64) * num_right + right_cols[entry_right]
    pattern = SparsityPattern(rows, cols, (num_rows, left.shape[1] * num_right))
    as_int = ft.partial(np.asarray, dtype=np.int32)
    return (
        pattern,
        as_int(entry_left),
        as_int(entry_right),
        _positions(pattern, rows, cols),
    )
