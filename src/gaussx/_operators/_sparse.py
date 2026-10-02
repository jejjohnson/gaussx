"""Sparse operator on a static, host-side sparsity pattern.

The pattern (the index arrays) is plain NumPy, hashable and static under
``jax.jit``; only the non-zero ``values`` are traced. Every piece of symbolic
work -- canonical ordering, transposition, pattern unions, the pattern of a
congruence ``Aᵀ diag(w) A`` -- therefore happens once per pattern on the host,
and is reused across every ``jit`` call, Newton step and ``vmap``-ped dataset.

``jax.experimental.sparse`` is imported only in this module.
"""

from __future__ import annotations

import functools as ft
import hashlib
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental import sparse as jsparse
from jaxtyping import Array, ArrayLike, Float

from gaussx._operators._block_diag import _to_frozenset


_PLAN_CACHE_SIZE = 128


class SparsityPattern:
    r"""Static, hashable sparsity pattern of a ``(m, n)`` matrix.

    The index arrays live on the host (NumPy) and are canonicalised on
    construction:

    - duplicate ``(row, col)`` pairs are merged;
    - entries are sorted row-major (by row, then column);
    - for a square pattern the diagonal is always present, so ``diag``,
      ``add_diagonal`` and a later Cholesky never change the pattern;
    - with ``symmetric=True`` only the lower triangle (``row >= col``) is
      stored, and an upper-triangle pair ``(i, j)`` is stored as ``(j, i)``.

    The hash is a content hash (SHA-256 of the canonical index arrays, the
    shape and the symmetry flag), so it is stable across processes and can key
    a cache of symbolic analyses. The index arrays are read-only.

    Args:
        rows: Row indices, shape ``(k,)``.
        cols: Column indices, shape ``(k,)``.
        shape: Matrix shape ``(m, n)``.
        symmetric: Store the lower triangle of a symmetric matrix. Requires a
            square ``shape``.

    Raises:
        ValueError: If the index arrays differ in length, are not rank 1, are
            out of range, or ``symmetric=True`` with a non-square shape.

    Example:
        ```python
        import numpy as np

        # Path graph 0 - 1 - 2, each edge given once
        p = gaussx.SparsityPattern(
            np.array([1, 2]), np.array([0, 1]), (3, 3), symmetric=True
        )
        p.rows, p.cols  # ([0, 1, 1, 2, 2], [0, 0, 1, 1, 2]): diagonal added
        ```
    """

    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]
    symmetric: bool
    _digest: str

    def __init__(
        self,
        rows: ArrayLike,
        cols: ArrayLike,
        shape: tuple[int, int],
        *,
        symmetric: bool = False,
    ) -> None:
        rows_, cols_, _ = _canonicalise(rows, cols, shape, symmetric)
        self._set(rows_, cols_, shape, symmetric)

    @classmethod
    def _from_canonical(
        cls,
        rows: np.ndarray,
        cols: np.ndarray,
        shape: tuple[int, int],
        symmetric: bool,
    ) -> SparsityPattern:
        pattern = cls.__new__(cls)
        pattern._set(rows, cols, shape, symmetric)
        return pattern

    def _set(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        shape: tuple[int, int],
        symmetric: bool,
    ) -> None:
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        cols = np.ascontiguousarray(cols, dtype=np.int32)
        rows.flags.writeable = False
        cols.flags.writeable = False
        shape = (int(shape[0]), int(shape[1]))
        digest = hashlib.sha256()
        digest.update(repr((shape, bool(symmetric))).encode())
        digest.update(rows.astype("<i4").tobytes())
        digest.update(cols.astype("<i4").tobytes())
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "symmetric", bool(symmetric))
        object.__setattr__(self, "_digest", digest.hexdigest())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SparsityPattern is immutable.")

    @property
    def nnz(self) -> int:
        """Number of stored entries (the length of ``values``)."""
        return int(self.rows.shape[0])

    @property
    def digest(self) -> str:
        """Hex SHA-256 content hash, identical across processes."""
        return self._digest

    def __hash__(self) -> int:
        return int(self._digest[:15], 16)  # 60 bits: never truncated by hash()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, SparsityPattern):
            return NotImplemented
        return (
            self._digest == other._digest
            and self.shape == other.shape
            and self.symmetric == other.symmetric
            and np.array_equal(self.rows, other.rows)
            and np.array_equal(self.cols, other.cols)
        )

    def __repr__(self) -> str:
        return (
            f"SparsityPattern(shape={self.shape}, nnz={self.nnz}, "
            f"symmetric={self.symmetric})"
        )

    @ft.cached_property
    def _diagonal_positions(self) -> np.ndarray:
        """Positions of the stored diagonal entries."""
        return np.flatnonzero(self.rows == self.cols).astype(np.int32)

    @ft.cached_property
    def _full(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The full matrix's entries in row-major order.

        Returns ``(rows, cols, value_index)``: entry ``e`` of the full matrix is
        ``values[value_index[e]]``. ``value_index`` is ``None`` when the stored
        entries already are the full matrix (general storage).
        """
        if not self.symmetric:
            return self.rows, self.cols, None
        off = np.flatnonzero(self.rows != self.cols)
        rows = np.concatenate([self.rows, self.cols[off]])
        cols = np.concatenate([self.cols, self.rows[off]])
        index = np.concatenate([np.arange(self.nnz), off])
        order = np.lexsort((cols, rows))
        return (
            rows[order].astype(np.int32),
            cols[order].astype(np.int32),
            index[order].astype(np.int32),
        )


class SparseOperator(lx.AbstractLinearOperator):
    r"""Sparse matrix with a static `SparsityPattern` and traced values.

    Only ``values`` is a pytree leaf, so ``jit``, ``grad`` and ``vmap`` act on
    the non-zeros while the pattern stays a compile-time constant. Changing
    the values (``eqx.tree_at(lambda op: op.values, Q, new_values)``) never
    retraces; changing the pattern does.

    The primitives dispatch on it: `gaussx.diag` reads the diagonal from the
    pattern; `gaussx.solve` uses CG for large positive semidefinite operators
    (`AutoSolver` rules) and a dense solve otherwise; `gaussx.logdet` uses
    `SLQLogdet` for large PSD operators; `gaussx.eig` with ``rank=`` runs
    Lanczos on the matvec; `gaussx.cholesky` returns a sparse
    `SparseCholeskyFactor`. For exact solves, log-determinants and marginal
    variances through that factor, pass `SparseCholeskySolver`.

    Args:
        values: Stored non-zeros in the pattern's canonical order, shape
            ``(nnz,)``. For a symmetric pattern, the lower triangle only.
        pattern: The static sparsity pattern.
        tags: Lineax tags. ``lx.symmetric_tag`` is added for a symmetric
            pattern.

    Example:
        ```python
        import numpy as np

        # ICAR structure matrix of the path graph 0 - 1 - 2 (edges once)
        senders, receivers = np.array([1, 2]), np.array([0, 1])
        deg = np.bincount(np.r_[senders, receivers], minlength=3)
        R = gaussx.SparseOperator.from_coo(
            np.r_[np.arange(3), senders],
            np.r_[np.arange(3), receivers],
            jnp.asarray(np.r_[deg, -np.ones(2)]),
            (3, 3),
            symmetric=True,
        )
        R.mv(jnp.ones(3))  # [0, 0, 0]: constants are in the null space
        ```
    """

    values: Float[Array, " nnz"]
    pattern: SparsityPattern = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        values: Float[ArrayLike, " nnz"],
        pattern: SparsityPattern,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        values = jnp.asarray(values)
        if values.shape != (pattern.nnz,):
            raise ValueError(
                f"values must have shape ({pattern.nnz},) to match the pattern, "
                f"got {values.shape}."
            )
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(jnp.result_type(values.dtype, jnp.float32))
        self.values = values
        self.pattern = pattern
        tags = _to_frozenset(tags)
        if pattern.symmetric:
            tags = tags | {lx.symmetric_tag}
        self.tags = tags

    @classmethod
    def from_coo(
        cls,
        rows: ArrayLike,
        cols: ArrayLike,
        values: Float[ArrayLike, " k"],
        shape: tuple[int, int],
        *,
        symmetric: bool = False,
        tags: object | frozenset[object] = frozenset(),
    ) -> SparseOperator:
        """Build from coordinate (COO) triplets.

        ``rows`` and ``cols`` must be concrete (host) integer arrays; only
        ``values`` may be traced. Duplicate coordinates are summed, and the
        diagonal of a square matrix is added with zeros where missing.

        Args:
            rows: Row indices, shape ``(k,)``.
            cols: Column indices, shape ``(k,)``.
            values: Entry values, shape ``(k,)``.
            shape: Matrix shape ``(m, n)``.
            symmetric: The matrix is symmetric and each off-diagonal pair is
                given **once** (either ``(i, j)`` or ``(j, i)``); it is mirrored
                to the other triangle. Giving both would sum them. This is
                the edge-once storage of an undirected graph.
            tags: Lineax tags.

        Returns:
            The operator, with values in the pattern's canonical order.
        """
        rows_, cols_, inverse = _canonicalise(rows, cols, shape, symmetric)
        pattern = SparsityPattern._from_canonical(rows_, cols_, shape, symmetric)
        values = jnp.asarray(values)
        if values.shape != (inverse.shape[0],):
            raise ValueError(
                f"values must have shape ({inverse.shape[0]},) to match rows and "
                f"cols, got {values.shape}."
            )
        values = jax.ops.segment_sum(values, inverse, num_segments=pattern.nnz)
        return cls(values, pattern, tags=tags)

    def _full_values(self) -> Float[Array, " nnz_full"]:
        _, _, index = self.pattern._full
        return self.values if index is None else self.values[index]

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        # segment_sum over the row-sorted full pattern; it beat the BCOO
        # matvec at every size in the G1 benchmark
        # (tests/operators/test_sparse.py::test_matvec_benchmark).
        rows, cols, _ = self.pattern._full
        return jax.ops.segment_sum(
            self._full_values() * vector[cols],
            rows,
            num_segments=self.pattern.shape[0],
            indices_are_sorted=True,
        )

    def as_matrix(self) -> Float[Array, "m n"]:
        rows, cols, _ = self.pattern._full
        dense = jnp.zeros(self.pattern.shape, dtype=self.values.dtype)
        return dense.at[rows, cols].add(self._full_values())

    def to_bcoo(self) -> jsparse.BCOO:
        """The full matrix as a `jax.experimental.sparse.BCOO` array.

        A symmetric pattern is expanded to both triangles.

        Returns:
            A ``BCOO`` with sorted, unique indices.
        """
        rows, cols, _ = self.pattern._full
        indices = jnp.asarray(np.column_stack([rows, cols]))
        return jsparse.BCOO(
            (self._full_values(), indices),
            shape=self.pattern.shape,
            indices_sorted=True,
            unique_indices=True,
        )

    def transpose(self) -> SparseOperator:
        if self.pattern.symmetric:
            return self
        pattern, perm = _transpose_plan(self.pattern)
        return SparseOperator(
            self.values[perm], pattern, tags=lx.transpose_tags(self.tags)
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.pattern.shape[1],), self.values.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.pattern.shape[0],), self.values.dtype)

    def diagonal(self) -> Float[Array, " k"]:
        """The diagonal, read from the pattern (no densification).

        Returns:
            ``diag(A)``, shape ``(min(m, n),)``.
        """
        pos = self.pattern._diagonal_positions
        out = jnp.zeros(min(self.pattern.shape), dtype=self.values.dtype)
        return out.at[self.pattern.rows[pos]].add(self.values[pos])

    def add_diagonal(
        self,
        d: Float[Array, " n"],
        *,
        tags: object | frozenset[object] | None = None,
    ) -> SparseOperator:
        """``A + diag(d)`` on the same pattern.

        Args:
            d: Diagonal to add, shape ``(n,)``.
            tags: Tags of the result. By default only symmetry is kept,
                since an arbitrary ``d`` can break definiteness; pass
                ``lx.positive_semidefinite_tag`` when you know it holds.

        Returns:
            The shifted operator, with the identical pattern.
        """
        n = self._square_size("add_diagonal")
        d = jnp.asarray(d)
        if d.shape != (n,):
            raise ValueError(f"d must have shape ({n},), got {d.shape}.")
        pos = self.pattern._diagonal_positions
        values = self.values.at[pos].add(d)
        if tags is None:
            tags = self.tags & {lx.symmetric_tag}
        return SparseOperator(values, self.pattern, tags=tags)

    def union(
        self,
        other: SparseOperator,
        *,
        tags: object | frozenset[object] | None = None,
    ) -> SparseOperator:
        """``A + B`` on the union of the two patterns.

        The union pattern and the scatter positions are computed on the host
        once per pair of patterns (and cached); only the values are added in
        JAX. Two symmetric patterns stay symmetric; otherwise the symmetric
        operand is expanded to both triangles.

        Args:
            other: Operator of the same shape.
            tags: Tags of the result. Defaults to the tags both operands share
                (a sum of PSD operators is PSD), minus ``unit_diagonal_tag``.

        Returns:
            The sum, on the union pattern.
        """
        if self.pattern.shape != other.pattern.shape:
            raise ValueError(
                f"Shapes differ: {self.pattern.shape} vs {other.pattern.shape}."
            )
        pattern, pos_a, pos_b = _union_plan(self.pattern, other.pattern)
        values_a = self._storage_values(pattern.symmetric)
        values_b = other._storage_values(pattern.symmetric)
        dtype = jnp.result_type(values_a, values_b)
        values = (
            jnp.zeros(pattern.nnz, dtype=dtype)
            .at[pos_a]
            .add(values_a)
            .at[pos_b]
            .add(values_b)
        )
        if tags is None:
            tags = (self.tags & other.tags) - {lx.unit_diagonal_tag}
        return SparseOperator(values, pattern, tags=tags)

    def congruence(
        self,
        A: SparseOperator,
        w: Float[Array, " m"],
        *,
        tags: object | frozenset[object] | None = None,
    ) -> SparseOperator:
        r"""``Aᵀ diag(w) A`` on the union of ``self``'s pattern and ``AᵀA``'s.

        ``(AᵀWA)_{ij} = Σ_k A_{ki} w_k A_{kj}`` is non-zero only where columns
        ``i`` and ``j`` share a row of ``A``. That pattern, unioned with this
        operator's own, and the index triples ``(p, q, k)`` feeding each entry
        are computed on the host once per ``(pattern, A.pattern)``; the values
        are one ``segment_sum`` in JAX. Because the result already lives on
        ``self``'s pattern (padded with zeros), ``self.union(result)`` is an
        aligned add, and a projector whose rows only touch neighbours in
        ``self`` (a FEM projector on a mesh precision) leaves the pattern
        unchanged.

        Args:
            A: Operator of shape ``(m, n)``, with ``n`` the size of ``self``.
            w: Row weights, shape ``(m,)``.
            tags: Tags of the result. Defaults to symmetry only (``w`` may
                have either sign).

        Returns:
            The congruence, symmetric storage iff ``self``'s pattern is.
        """
        n = self._square_size("congruence")
        m, n_a = A.pattern.shape
        if n_a != n:
            raise ValueError(f"A must have {n} columns, got shape {A.pattern.shape}.")
        w = jnp.asarray(w)
        if w.shape != (m,):
            raise ValueError(f"w must have shape ({m},), got {w.shape}.")
        pattern, p, q, k, target = _congruence_plan(self.pattern, A.pattern)
        full = A._full_values()
        contributions = full[p] * w[k] * full[q]
        values = jax.ops.segment_sum(contributions, target, num_segments=pattern.nnz)
        if tags is None:
            tags = frozenset()
        return SparseOperator(values, pattern, tags=tags)

    def _storage_values(self, symmetric: bool) -> Float[Array, " nnz"]:
        """Values in symmetric (stored) or general (full) storage."""
        return self.values if symmetric else self._full_values()

    def _square_size(self, method: str) -> int:
        m, n = self.pattern.shape
        if m != n:
            raise ValueError(f"{method} needs a square operator, got {(m, n)}.")
        return n


# ---------------------------------------------------------------------------
# Host-side symbolic plans (cached per pattern, reused across jit / Newton
# steps / vmap).
# ---------------------------------------------------------------------------


def _canonicalise(
    rows: ArrayLike,
    cols: ArrayLike,
    shape: tuple[int, int],
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical (sorted, unique, diagonal-complete) pattern of COO indices.

    Returns the canonical ``rows`` and ``cols`` and, for each input entry, its
    position in them.
    """
    try:
        r = np.asarray(rows)
        c = np.asarray(cols)
    except jax.errors.TracerArrayConversionError as err:
        raise TypeError(
            "Sparsity indices must be concrete host arrays; only values may be traced."
        ) from err
    if r.ndim != 1 or c.ndim != 1 or r.shape != c.shape:
        raise ValueError(
            f"rows and cols must be rank-1 arrays of equal length, got "
            f"{r.shape} and {c.shape}."
        )
    if r.size and not (
        np.issubdtype(r.dtype, np.integer) and np.issubdtype(c.dtype, np.integer)
    ):
        raise TypeError(f"Indices must be integers, got {r.dtype} and {c.dtype}.")
    r = r.astype(np.int64)
    c = c.astype(np.int64)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if r.size and (
        r.min() < 0 or r.max() >= n_rows or c.min() < 0 or c.max() >= n_cols
    ):
        raise ValueError(f"Indices out of range for shape {(n_rows, n_cols)}.")
    if symmetric:
        if n_rows != n_cols:
            raise ValueError(f"symmetric=True needs a square shape, got {shape}.")
        r, c = np.maximum(r, c), np.minimum(r, c)
    n_given = r.size
    if n_rows == n_cols:
        diagonal = np.arange(n_rows)
        r = np.concatenate([r, diagonal])
        c = np.concatenate([c, diagonal])
    keys, inverse = np.unique(r * n_cols + c, return_inverse=True)
    return (
        (keys // n_cols).astype(np.int32),
        (keys % n_cols).astype(np.int32),
        inverse.ravel()[:n_given].astype(np.int32),
    )


def _positions(
    pattern: SparsityPattern, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Positions of the entries ``(rows, cols)`` in ``pattern`` (all present)."""
    n_cols = pattern.shape[1]
    keys = pattern.rows.astype(np.int64) * n_cols + pattern.cols
    return np.searchsorted(keys, rows.astype(np.int64) * n_cols + cols).astype(np.int32)


def _storage_indices(
    pattern: SparsityPattern, symmetric: bool
) -> tuple[np.ndarray, np.ndarray]:
    if symmetric:
        return pattern.rows, pattern.cols
    rows, cols, _ = pattern._full
    return rows, cols


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _transpose_plan(pattern: SparsityPattern) -> tuple[SparsityPattern, np.ndarray]:
    m, n = pattern.shape
    perm = np.lexsort((pattern.rows, pattern.cols)).astype(np.int32)
    transposed = SparsityPattern._from_canonical(
        pattern.cols[perm], pattern.rows[perm], (n, m), False
    )
    return transposed, perm


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _union_plan(
    a: SparsityPattern, b: SparsityPattern
) -> tuple[SparsityPattern, np.ndarray, np.ndarray]:
    symmetric = a.symmetric and b.symmetric
    rows_a, cols_a = _storage_indices(a, symmetric)
    rows_b, cols_b = _storage_indices(b, symmetric)
    rows, cols, _ = _canonicalise(
        np.concatenate([rows_a, rows_b]),
        np.concatenate([cols_a, cols_b]),
        a.shape,
        symmetric,
    )
    union = SparsityPattern._from_canonical(rows, cols, a.shape, symmetric)
    return union, _positions(union, rows_a, cols_a), _positions(union, rows_b, cols_b)


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _congruence_plan(
    base: SparsityPattern, a: SparsityPattern
) -> tuple[SparsityPattern, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pattern and index triples of ``Aᵀ diag(w) A`` on top of ``base``.

    Entry ``e`` of the full ``A`` (row-sorted) pairs with every entry of the
    same row; pair ``(p, q)`` contributes ``A_p w_k A_q`` to ``(col_p, col_q)``.
    """
    a_rows, a_cols, _ = a._full
    n_entries = a_rows.shape[0]
    counts = np.bincount(a_rows, minlength=a.shape[0])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    reps = counts[a_rows]
    p = np.repeat(np.arange(n_entries), reps)
    offset = np.arange(p.shape[0]) - np.repeat(np.cumsum(reps) - reps, reps)
    q = starts[a_rows[p]] + offset
    i, j = a_cols[p], a_cols[q]
    if base.symmetric:
        keep = i >= j
        p, q, i, j = p[keep], q[keep], i[keep], j[keep]
    rows, cols, _ = _canonicalise(
        np.concatenate([base.rows, i]),
        np.concatenate([base.cols, j]),
        base.shape,
        base.symmetric,
    )
    pattern = SparsityPattern._from_canonical(rows, cols, base.shape, base.symmetric)
    # ``p`` and ``q`` index the full layout, i.e. ``A._full_values()``.
    return (
        pattern,
        p.astype(np.int32),
        q.astype(np.int32),
        a_rows[p].astype(np.int32),
        _positions(pattern, i, j),
    )


# ---------------------------------------------------------------------------
# lineax registrations: structural predicates read the tags; ``linearise`` /
# ``materialise`` are the identity (needed by matrix-free solvers such as CG);
# ``diagonal`` is exact from the pattern.
# ---------------------------------------------------------------------------


@lx.is_symmetric.register(SparseOperator)
def _(operator: SparseOperator) -> bool:
    return bool(
        operator.tags
        & {
            lx.symmetric_tag,
            lx.diagonal_tag,
            lx.positive_semidefinite_tag,
            lx.negative_semidefinite_tag,
        }
    )


for _check, _tag in (
    (lx.is_diagonal, lx.diagonal_tag),
    (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
    (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
    (lx.is_tridiagonal, lx.tridiagonal_tag),
    (lx.is_lower_triangular, lx.lower_triangular_tag),
    (lx.is_upper_triangular, lx.upper_triangular_tag),
    (lx.has_unit_diagonal, lx.unit_diagonal_tag),
):
    _check.register(SparseOperator)(lambda operator, tag=_tag: tag in operator.tags)

lx.linearise.register(SparseOperator)(lambda operator: operator)
lx.materialise.register(SparseOperator)(lambda operator: operator)
lx.diagonal.register(SparseOperator)(lambda operator: operator.diagonal())
