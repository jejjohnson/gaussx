"""Sparse operator A ∈ ℝ^{M × N} on a static, host-side sparsity pattern.

A stores only the entries [A]_ij at the positions (i, j) of its pattern:
the row indices and the column indices are host-side NumPy arrays, hashable
and static under ``jax.jit``, and only the stored values are traced. Every
piece of symbolic work (canonical ordering, transposition, pattern unions,
the pattern of a congruence Aᵀ diag(w) A) therefore happens once per pattern
on the host, and is reused across every ``jit`` call, Newton step and
``vmap``-ped dataset.

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
    r"""Static, hashable sparsity pattern of an M × N matrix A.

        pattern = {(i, j) : [A]_ij is stored},   i < M, j < N

    The row indices and column indices live on the host (NumPy) and are
    canonicalised on construction:

    - duplicate (i, j) pairs are merged;
    - entries are sorted row-major (by row, then column);
    - for a square pattern the diagonal is always present, so ``diag``,
      ``add_diagonal`` and a later Cholesky never change the pattern;
    - with ``symmetric=True`` only the lower triangle (i ≥ j) is stored, and
      an upper-triangle pair (i, j) is stored as (j, i).

    The hash is a content hash (SHA-256 of the canonical index arrays, the
    shape and the symmetry flag), so it is stable across processes and can key
    a cache of symbolic analyses. The index arrays are read-only.

    Args:
        rows: Row index i of each given entry, one per entry.
        cols: Column index j of each given entry, one per entry.
        shape: Matrix shape (M, N).
        symmetric: Store the lower triangle of a symmetric A. Requires a
            square ``shape``.

    Raises:
        TypeError: If the index arrays are traced or are not integers.
        ValueError: If the index arrays differ in length, are not rank 1, are
            out of range, or ``symmetric=True`` with a non-square shape.

    Examples:
        The path graph 0 − 1 − 2, each edge given once; the diagonal is
        added:

        >>> import numpy as np
        >>> import gaussx
        >>> rows, cols = np.array([1, 2]), np.array([0, 1])
        >>> pattern = gaussx.SparsityPattern(rows, cols, (3, 3), symmetric=True)
        >>> pattern.rows.tolist(), pattern.cols.tolist()
        ([0, 1, 1, 2, 2], [0, 0, 1, 1, 2])
        >>> pattern
        SparsityPattern(shape=(3, 3), nnz=5, symmetric=True)

        Equal content gives an equal pattern and an equal hash:

        >>> again = gaussx.SparsityPattern(rows, cols, (3, 3), symmetric=True)
        >>> again == pattern, hash(again) == hash(pattern)
        (True, True)
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
        canonical_rows, canonical_cols, _ = _canonicalise(rows, cols, shape, symmetric)
        self._set(canonical_rows, canonical_cols, shape, symmetric)

    @classmethod
    def _from_canonical(
        cls,
        rows: np.ndarray,
        cols: np.ndarray,
        shape: tuple[int, int],
        symmetric: bool,
    ) -> SparsityPattern:
        """Wrap index arrays that are already canonical, skipping the checks.

        Args:
            rows: Canonical row indices.
            cols: Canonical column indices.
            shape: Matrix shape (M, N).
            symmetric: Whether the indices are a lower-triangle storage.

        Returns:
            The pattern, built without ``__init__``.
        """
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
        """Freeze the index arrays and store them with their content hash.

        Args:
            rows: Canonical row indices.
            cols: Canonical column indices.
            shape: Matrix shape (M, N).
            symmetric: Whether the indices are a lower-triangle storage.
        """
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        cols = np.ascontiguousarray(cols, dtype=np.int32)
        rows.flags.writeable = False
        cols.flags.writeable = False
        shape = (int(shape[0]), int(shape[1]))
        # SHA-256 of (shape, symmetric, row indices, column indices), with
        # little-endian int32 bytes so the digest is the same on every host
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
        # the first 15 hex digits, 60 bits: never truncated by hash()
        return int(self._digest[:15], 16)

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
        """Positions in the storage of the stored diagonal entries (i, i)."""
        return np.flatnonzero(self.rows == self.cols).astype(np.int32)

    @ft.cached_property
    def _full(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The full matrix's entries in row-major order.

        A symmetric storage holds the lower triangle only; the full pattern
        adds the mirror (j, i) of every stored off-diagonal (i, j), since
        [A]_ji = [A]_ij.

        Returns:
            Tuple ``(rows, cols, value_index)``: entry e of the full matrix
            is ``values[value_index[e]]``. ``value_index`` is ``None`` when
            the stored entries already are the full matrix (general storage).
        """
        if not self.symmetric:
            return self.rows, self.cols, None
        # stored off-diagonal entries, i > j
        off_diagonal = np.flatnonzero(self.rows != self.cols)
        # the stored entries, then their mirrors (j, i)
        rows = np.concatenate([self.rows, self.cols[off_diagonal]])
        cols = np.concatenate([self.cols, self.rows[off_diagonal]])
        # a mirror reads the value of the entry it mirrors
        index = np.concatenate([np.arange(self.nnz), off_diagonal])
        # sort row-major: by row, then column
        order = np.lexsort((cols, rows))
        return (
            rows[order].astype(np.int32),
            cols[order].astype(np.int32),
            index[order].astype(np.int32),
        )


class SparseOperator(lx.AbstractLinearOperator):
    r"""Sparse matrix A with a static `SparsityPattern` and traced values.

        [A x]_i = ∑_j [A]_ij x_j,   summed over the stored entries of row i

    Only the stored values are a pytree leaf, so ``jit``, ``grad`` and
    ``vmap`` act on the non-zeros while the pattern stays a compile-time
    constant. Changing the values
    (``eqx.tree_at(lambda op: op.values, Q, new_values)``) never retraces;
    changing the pattern does.

    The primitives dispatch on it: `gaussx.diag` reads the diagonal from the
    pattern; `gaussx.solve` uses CG for large positive semidefinite operators
    (`AutoSolver` rules) and a dense solve otherwise; `gaussx.logdet` uses
    `SLQLogdet` for large PSD operators; `gaussx.eig` with ``rank=`` runs
    Lanczos on the matvec; `gaussx.cholesky` returns a sparse
    `SparseCholeskyFactor`. For exact solves, log-determinants and marginal
    variances through that factor, pass `SparseCholeskySolver`.

    Args:
        values: Stored values of A, one per pattern entry, in the pattern's
            canonical order; for a symmetric pattern, the lower triangle
            only. Integer values are promoted to floating point.
        pattern: The static sparsity pattern of A.
        tags: Extra lineax tags for A; ``lx.symmetric_tag`` is added for a
            symmetric pattern.

    Raises:
        ValueError: If ``values`` does not hold one value per pattern entry.

    Examples:
        The ICAR structure matrix of the path graph 0 − 1 − 2, edges given
        once:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> rows = np.array([0, 1, 2, 1, 2])
        >>> cols = np.array([0, 1, 2, 0, 1])
        >>> values = jnp.array([1.0, 2.0, 1.0, -1.0, -1.0])
        >>> R = gaussx.SparseOperator.from_coo(
        ...     rows, cols, values, (3, 3), symmetric=True
        ... )
        >>> R.as_matrix().tolist()
        [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]

        Constants are in its null space:

        >>> bool(jnp.allclose(R.mv(jnp.ones(3)), 0.0))
        True

        The diagonal comes from the pattern, without densifying:

        >>> gaussx.diag(R).tolist()
        [1.0, 2.0, 1.0]
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
        """Build A from coordinate (COO) triplets (i, j, [A]_ij).

            [A]_ij = ∑ of the values given at (i, j)

        ``rows`` and ``cols`` must be concrete (host) integer arrays; only
        ``values`` may be traced. Duplicate coordinates are summed, and the
        diagonal of a square matrix is added with zeros where missing.

        Args:
            rows: Row index i of each triplet.
            cols: Column index j of each triplet.
            values: Value of each triplet.
            shape: Matrix shape (M, N).
            symmetric: A is symmetric and each off-diagonal pair is given
                **once** (either (i, j) or (j, i)); it is mirrored to the
                other triangle. Giving both would sum them. This is the
                edge-once storage of an undirected graph.
            tags: Extra lineax tags for A.

        Returns:
            The operator, with values in the pattern's canonical order.

        Raises:
            TypeError: If the indices are traced or are not integers.
            ValueError: If the indices are malformed or out of range, or
                ``values`` does not hold one value per triplet.

        Examples:
            Duplicate triplets are summed, and the missing diagonal entry
            is a stored zero:

            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> rows, cols = np.array([0, 0, 1]), np.array([1, 1, 1])
            >>> values = jnp.array([1.0, 2.0, 5.0])
            >>> A = gaussx.SparseOperator.from_coo(rows, cols, values, (2, 2))
            >>> A.as_matrix().tolist()
            [[0.0, 3.0], [0.0, 5.0]]
            >>> A.pattern.nnz
            3
        """
        canonical_rows, canonical_cols, inverse = _canonicalise(
            rows, cols, shape, symmetric
        )
        pattern = SparsityPattern._from_canonical(
            canonical_rows, canonical_cols, shape, symmetric
        )
        values = jnp.asarray(values)
        if values.shape != (inverse.shape[0],):
            raise ValueError(
                f"values must have shape ({inverse.shape[0]},) to match rows and "
                f"cols, got {values.shape}."
            )
        # [A]_ij = ∑ of the triplet values that land on pattern entry (i, j)
        values = jax.ops.segment_sum(values, inverse, num_segments=pattern.nnz)
        return cls(values, pattern, tags=tags)

    def _full_values(self) -> Float[Array, " nnz_full"]:
        """Values of the full pattern, both triangles for a symmetric storage.

        Returns:
            The value of each full-pattern entry, in row-major order.
        """
        _, _, index = self.pattern._full
        return self.values if index is None else self.values[index]

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        """Apply A to a vector, one segment sum over the stored entries.

            [A x]_i = ∑_j [A]_ij x_j

        Args:
            vector: Input x, of length N.

        Returns:
            A x, of length M.
        """
        # segment_sum over the row-sorted full pattern; it beat the BCOO
        # matvec at every size in the G1 benchmark
        # (tests/operators/test_sparse.py::test_matvec_benchmark).
        rows, cols, _ = self.pattern._full
        # [A]_ij x_j for every stored entry (i, j)
        entry_products = self._full_values() * vector[cols]
        # [A x]_i = ∑_j [A]_ij x_j, one segment per row
        return jax.ops.segment_sum(
            entry_products,
            rows,
            num_segments=self.pattern.shape[0],
            indices_are_sorted=True,
        )

    def as_matrix(self) -> Float[Array, "m n"]:
        """Dense A, M × N, zero off the pattern.

        Returns:
            The materialised matrix.
        """
        rows, cols, _ = self.pattern._full
        # 0 ∈ ℝ^{M × N}
        dense = jnp.zeros(self.pattern.shape, dtype=self.values.dtype)
        # [A]_ij at every stored entry (i, j)
        return dense.at[rows, cols].add(self._full_values())

    def to_bcoo(self) -> jsparse.BCOO:
        """The full A as a `jax.experimental.sparse.BCOO` array.

        A symmetric pattern is expanded to both triangles.

        Returns:
            A ``BCOO`` with sorted, unique indices.

        Examples:
            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> rows, cols = np.array([0, 1, 1]), np.array([0, 0, 1])
            >>> values = jnp.array([2.0, -1.0, 2.0])
            >>> A = gaussx.SparseOperator.from_coo(
            ...     rows, cols, values, (2, 2), symmetric=True
            ... )
            >>> A.to_bcoo().todense().tolist()
            [[2.0, -1.0], [-1.0, 2.0]]
        """
        rows, cols, _ = self.pattern._full
        # one (i, j) row per full-pattern entry
        indices = jnp.asarray(np.column_stack([rows, cols]))
        return jsparse.BCOO(
            (self._full_values(), indices),
            shape=self.pattern.shape,
            indices_sorted=True,
            unique_indices=True,
        )

    def transpose(self) -> SparseOperator:
        """Aᵀ on the transposed pattern, [Aᵀ]_ji = [A]_ij.

        A symmetric storage is its own transpose; otherwise the values are
        permuted into the row-major order of the transposed pattern, which
        is planned once per pattern on the host.

        Returns:
            The transposed operator, with transposed tags.
        """
        if self.pattern.symmetric:
            return self
        pattern, perm = _transpose_plan(self.pattern)
        # [Aᵀ]_ji = [A]_ij, reordered row-major for Aᵀ
        transposed_values = self.values[perm]
        return SparseOperator(
            transposed_values, pattern, tags=lx.transpose_tags(self.tags)
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self.pattern.shape[1],), self.values.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (M,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self.pattern.shape[0],), self.values.dtype)

    def diagonal(self) -> Float[Array, " k"]:
        """The diagonal of A, read from the pattern (no densification).

            diag(A)_i = [A]_ii,   i < min(M, N)

        Returns:
            diag(A), of length min(M, N).

        Examples:
            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> rows, cols = np.array([0, 1, 1]), np.array([0, 0, 1])
            >>> values = jnp.array([2.0, -1.0, 3.0])
            >>> A = gaussx.SparseOperator.from_coo(rows, cols, values, (2, 2))
            >>> A.diagonal().tolist()
            [2.0, 3.0]
        """
        diagonal_positions = self.pattern._diagonal_positions
        # 0 ∈ ℝ^{min(M, N)}
        zeros = jnp.zeros(min(self.pattern.shape), dtype=self.values.dtype)
        # diag(A)_i = [A]_ii at every stored diagonal entry
        return zeros.at[self.pattern.rows[diagonal_positions]].add(
            self.values[diagonal_positions]
        )

    def add_diagonal(
        self,
        d: Float[Array, " n"],
        *,
        tags: object | frozenset[object] | None = None,
    ) -> SparseOperator:
        """A + diag(d) on the same pattern.

            [A + diag(d)]_ii = [A]_ii + d_i

        A square pattern always stores its diagonal, so the shift never
        changes the pattern.

        Args:
            d: Diagonal shift d ∈ ℝ^N.
            tags: Tags of the result. By default only symmetry is kept,
                since an arbitrary d can break definiteness; pass
                ``lx.positive_semidefinite_tag`` when you know it holds.

        Returns:
            The shifted operator, with the identical pattern.

        Raises:
            ValueError: If A is not square or d is not of length N.

        Examples:
            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> rows, cols = np.array([1]), np.array([0])
            >>> values = jnp.array([-1.0])
            >>> A = gaussx.SparseOperator.from_coo(rows, cols, values, (2, 2))
            >>> shifted = A.add_diagonal(jnp.array([2.0, 3.0]))
            >>> shifted.as_matrix().tolist()
            [[2.0, 0.0], [-1.0, 3.0]]
            >>> shifted.pattern == A.pattern
            True
        """
        n = self._square_size("add_diagonal")
        d = jnp.asarray(d)
        if d.shape != (n,):
            raise ValueError(f"d must have shape ({n},), got {d.shape}.")
        diagonal_positions = self.pattern._diagonal_positions
        # [A]_ii + d_i
        values = self.values.at[diagonal_positions].add(d)
        if tags is None:
            tags = self.tags & {lx.symmetric_tag}
        return SparseOperator(values, self.pattern, tags=tags)

    def union(
        self,
        other: SparseOperator,
        *,
        tags: object | frozenset[object] | None = None,
    ) -> SparseOperator:
        """A + B on the union of the two patterns.

            [A + B]_ij = [A]_ij + [B]_ij,   zero where a pattern has no entry

        The union pattern and the scatter positions are computed on the host
        once per pair of patterns (and cached); only the values are added in
        JAX. Two symmetric patterns stay symmetric; otherwise the symmetric
        operand is expanded to both triangles.

        Args:
            other: The second summand B, of the same shape as A.
            tags: Tags of the result. Defaults to the tags both operands share
                (a sum of PSD operators is PSD), minus ``unit_diagonal_tag``.

        Returns:
            The sum, on the union pattern.

        Raises:
            ValueError: If A and B differ in shape.

        Examples:
            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> A = gaussx.SparseOperator.from_coo(
            ...     np.array([1]), np.array([0]), jnp.array([1.0]), (2, 2)
            ... )
            >>> B = gaussx.SparseOperator.from_coo(
            ...     np.array([0]), np.array([1]), jnp.array([4.0]), (2, 2)
            ... )
            >>> A.union(B).as_matrix().tolist()
            [[0.0, 4.0], [1.0, 0.0]]
        """
        if self.pattern.shape != other.pattern.shape:
            raise ValueError(
                f"Shapes differ: {self.pattern.shape} vs {other.pattern.shape}."
            )
        pattern, positions_a, positions_b = _union_plan(self.pattern, other.pattern)
        values_a = self._storage_values(pattern.symmetric)
        values_b = other._storage_values(pattern.symmetric)
        dtype = jnp.result_type(values_a, values_b)
        # 0 on the union pattern
        zeros = jnp.zeros(pattern.nnz, dtype=dtype)
        # [A]_ij scattered onto the union, then [A]_ij + [B]_ij
        values = zeros.at[positions_a].add(values_a).at[positions_b].add(values_b)
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
        r"""Aᵀ diag(w) A on the union of this pattern and the pattern of AᵀA.

            [Aᵀ diag(w) A]_ij = ∑_k [A]_ki w_k [A]_kj

        Here A is the argument, not this operator, whose pattern alone (not
        its values) shapes the result. An entry (i, j) is non-zero only where
        columns i and j share a row k of A. That pattern, unioned with this
        operator's own, and the index triples (p, q, k) feeding each entry
        are computed on the host once per pair of patterns; the values are
        one ``segment_sum`` in JAX. Because the result already lives on this
        operator's pattern (padded with zeros), ``self.union(result)`` is an
        aligned add, and a projector whose rows only touch neighbours in this
        pattern (a FEM projector on a mesh precision) leaves the pattern
        unchanged.

        Args:
            A: Sparse operator, K × N, with N the size of this operator.
            w: Row weights w ∈ ℝ^K.
            tags: Tags of the result. Defaults to no tags (w may have either
                sign); a symmetric pattern still adds ``lx.symmetric_tag``.

        Returns:
            The congruence, in symmetric storage iff this pattern is.

        Raises:
            ValueError: If this operator is not square, A does not have N
                columns, or w is not of length K.

        Examples:
            >>> import jax.numpy as jnp
            >>> import numpy as np
            >>> import gaussx
            >>> base = gaussx.SparseOperator.from_coo(
            ...     np.array([], dtype=int), np.array([], dtype=int),
            ...     jnp.zeros(0), (2, 2), symmetric=True,
            ... )
            >>> A = gaussx.SparseOperator.from_coo(
            ...     np.array([0, 0]), np.array([0, 1]), jnp.array([1.0, 2.0]), (1, 2)
            ... )
            >>> base.congruence(A, jnp.array([3.0])).as_matrix().tolist()
            [[3.0, 6.0], [6.0, 12.0]]
        """
        n = self._square_size("congruence")
        num_rows_a, num_cols_a = A.pattern.shape
        if num_cols_a != n:
            raise ValueError(f"A must have {n} columns, got shape {A.pattern.shape}.")
        w = jnp.asarray(w)
        if w.shape != (num_rows_a,):
            raise ValueError(f"w must have shape ({num_rows_a},), got {w.shape}.")
        pattern, p, q, k, target = _congruence_plan(self.pattern, A.pattern)
        full_values = A._full_values()
        # [A]_ki w_k [A]_kj for every triple (p, q, k)
        contributions = full_values[p] * w[k] * full_values[q]
        # ∑_k [A]_ki w_k [A]_kj, one segment per result entry (i, j)
        values = jax.ops.segment_sum(contributions, target, num_segments=pattern.nnz)
        if tags is None:
            tags = frozenset()
        return SparseOperator(values, pattern, tags=tags)

    def _storage_values(self, symmetric: bool) -> Float[Array, " nnz"]:
        """Values in symmetric (stored) or general (full) storage.

        Args:
            symmetric: Whether the target storage is a lower triangle.

        Returns:
            The stored values, or the full-pattern values.
        """
        return self.values if symmetric else self._full_values()

    def _square_size(self, method: str) -> int:
        """Size N of a square A, for a method that needs one.

        Args:
            method: Name of the calling method, for the error message.

        Returns:
            N, with A ∈ ℝ^{N × N}.

        Raises:
            ValueError: If A is not square.
        """
        num_rows, num_cols = self.pattern.shape
        if num_rows != num_cols:
            raise ValueError(
                f"{method} needs a square operator, got {(num_rows, num_cols)}."
            )
        return num_cols


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

    Algorithm:
        1. Validate the indices: concrete, rank 1, integer and in range.
        2. For a symmetric pattern, fold every (i, j) to the lower triangle
           (max(i, j), min(i, j)).
        3. For a square pattern, append the diagonal (i, i), i < N.
        4. Sort and merge the row-major linear indices i N + j.

    Args:
        rows: Row indices i, one per given entry.
        cols: Column indices j, one per given entry.
        shape: Matrix shape (M, N).
        symmetric: Whether to store the lower triangle only.

    Returns:
        Tuple ``(rows, cols, inverse)``: the canonical row and column indices
        and, for each given entry, its position in them.

    Raises:
        TypeError: If the indices are traced or are not integers.
        ValueError: If the indices are malformed or out of range, or
            ``symmetric`` with a non-square shape.
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
        # (i, j) ↦ (max(i, j), min(i, j)), the lower triangle
        r, c = np.maximum(r, c), np.minimum(r, c)
    n_given = r.size
    if n_rows == n_cols:
        # (i, i) for i < N, so the diagonal is always stored
        diagonal = np.arange(n_rows)
        r = np.concatenate([r, diagonal])
        c = np.concatenate([c, diagonal])
    # sorted unique row-major linear indices i N + j
    keys, inverse = np.unique(r * n_cols + c, return_inverse=True)
    # i = ⌊key / N⌋, j = key mod N; the inverse of the given entries only
    return (
        (keys // n_cols).astype(np.int32),
        (keys % n_cols).astype(np.int32),
        inverse.ravel()[:n_given].astype(np.int32),
    )


def _positions(
    pattern: SparsityPattern, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Positions of the entries (i, j) in a pattern, all of them present.

    Args:
        pattern: Pattern that contains every queried entry.
        rows: Row indices i of the queried entries.
        cols: Column indices j of the queried entries.

    Returns:
        The storage position of each queried entry.
    """
    n_cols = pattern.shape[1]
    # row-major linear indices i N + j of the pattern, already sorted
    keys = pattern.rows.astype(np.int64) * n_cols + pattern.cols
    # binary search for i N + j of each queried entry
    return np.searchsorted(keys, rows.astype(np.int64) * n_cols + cols).astype(np.int32)


def _storage_indices(
    pattern: SparsityPattern, symmetric: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices in symmetric (stored) or general (full) storage.

    Args:
        pattern: The pattern to read.
        symmetric: Whether the target storage is a lower triangle.

    Returns:
        Tuple ``(rows, cols)`` of the stored or the full-pattern entries.
    """
    if symmetric:
        return pattern.rows, pattern.cols
    rows, cols, _ = pattern._full
    return rows, cols


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _transpose_plan(pattern: SparsityPattern) -> tuple[SparsityPattern, np.ndarray]:
    """Pattern of Aᵀ and the permutation that reorders A's values for it.

    Args:
        pattern: General (non-symmetric) pattern of A, M × N.

    Returns:
        Tuple ``(transposed, perm)``: the N × M pattern of Aᵀ, and
        ``values[perm]``, the values of A in Aᵀ's row-major order.
    """
    m, n = pattern.shape
    # sort A's entries by column, then row: row-major order for Aᵀ
    perm = np.lexsort((pattern.rows, pattern.cols)).astype(np.int32)
    # (i, j) ↦ (j, i)
    transposed = SparsityPattern._from_canonical(
        pattern.cols[perm], pattern.rows[perm], (n, m), False
    )
    return transposed, perm


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _union_plan(
    a: SparsityPattern, b: SparsityPattern
) -> tuple[SparsityPattern, np.ndarray, np.ndarray]:
    """Union of two patterns and where each one's entries land in it.

    Args:
        a: Pattern of A.
        b: Pattern of B, of the same shape.

    Returns:
        Tuple ``(union, positions_a, positions_b)``: the union pattern,
        symmetric iff both are, and the union position of each storage
        entry of A and of B.
    """
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
    """Pattern and index triples of Aᵀ diag(w) A on top of a base pattern.

        [Aᵀ diag(w) A]_ij = ∑_k [A]_ki w_k [A]_kj

    Algorithm:
        1. Pair every entry p of the full, row-sorted A with every entry q
           of the same row k; pair (p, q) contributes [A]_p w_k [A]_q to
           (i, j) = (col_p, col_q).
        2. For a symmetric base, keep the lower-triangle pairs i ≥ j.
        3. Union the (i, j) with the base pattern and look up the position
           of each pair's target entry.

    Args:
        base: Pattern the result is laid on (this operator's).
        a: Pattern of A, K × N.

    Returns:
        Tuple ``(pattern, p, q, k, target)``: the result pattern; for every
        pair, the full-layout positions p and q in A (indices into
        ``A._full_values()``), the shared row k, and the position of its
        target entry in the result.
    """
    a_rows, a_cols, _ = a._full
    n_entries = a_rows.shape[0]
    # number of stored entries in each row k of A
    counts = np.bincount(a_rows, minlength=a.shape[0])
    # position of the first entry of each row
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    # entry p pairs with every entry of its own row
    reps = counts[a_rows]
    p = np.repeat(np.arange(n_entries), reps)
    # rank of each partner q within its row
    offset = np.arange(p.shape[0]) - np.repeat(np.cumsum(reps) - reps, reps)
    q = starts[a_rows[p]] + offset
    # target entry (i, j) = (col_p, col_q)
    i, j = a_cols[p], a_cols[q]
    if base.symmetric:
        # lower triangle i ≥ j only
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
