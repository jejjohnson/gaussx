"""Host-side symbolic Cholesky analysis, computed once per sparsity pattern.

Everything here is NumPy / SciPy on the host. The analysis of P A Pᵀ = L Lᵀ depends
only on the `SparsityPattern` and the ordering, never on the values, so it is cached
per ``(pattern, ordering, backend)`` and reused across ``jit`` calls, Newton steps,
θ-points and ``vmap``-ped datasets.
"""

from __future__ import annotations

import functools as ft
from typing import Any, Literal

import numpy as np

from gaussx._operators._sparse import SparsityPattern, _positions


Ordering = Literal["rcm", "natural", "amd"]
Backend = Literal["jax", "cholmod"]

_ORDERINGS = ("rcm", "natural", "amd")
_BACKENDS = ("jax", "cholmod")
_CACHE_SIZE = 64


class SymbolicCholesky:
    """Symbolic Cholesky factor of a symmetric sparsity pattern.

        P A Pᵀ = L Lᵀ,   [P A Pᵀ]_kl = [A]_perm[k],perm[l]
        struct(L_:,j) = union of struct([P A Pᵀ]_j:,j) and, over parent(c) = j,
            of struct(L_:,c) without c
        parent(j) = min{i > j : L_ij ≠ 0}

    It holds the elimination tree and the pattern of L in compressed sparse
    column (CSC) form, with each column's rows sorted so the diagonal comes first,
    together with the static index plans the numeric phase, the triangular solves
    and the Takahashi recursion gather through.

    Build it with `gaussx.symbolic_cholesky` (cached), not directly. It is
    hashable and compared by ``(pattern, ordering, backend, banded)``, so it sits in
    a static field and causes no retrace for equal inputs.

    Args:
        pattern: Square sparsity pattern of A to analyse.
        ordering: Name of the ordering that produced ``perm``.
        backend: Numeric backend, ``"jax"`` or ``"cholmod"``.
        perm: Fill-reducing permutation P, as P A Pᵀ = ``A[perm][:, perm]``.
        banded: Force the banded (``True``) or windowed (``False``) layout;
            ``None`` picks by storage cost.

    Attributes:
        pattern: The pattern it was computed for.
        ordering: The fill-reducing ordering used.
        backend: The numeric backend (``"jax"`` or ``"cholmod"``).
        n: Matrix size N.
        perm: P A Pᵀ = ``A[perm][:, perm]``, of length N.
        iperm: The inverse permutation, ``iperm[perm] = arange(n)``.
        parent: Elimination tree, ``parent[j]`` (``-1`` at a root).
        colptr: CSC column pointers of L, of length N + 1.
        rowidx: CSC row indices of L, of length nnz(L).
        colidx: Column of each entry of L, of length nnz(L).
        nnz: Number of entries of L (diagonal included): the fill.
        nnz_lower: Number of entries in the lower triangle of A.
        max_col: Longest column of L (padded window of the column plans).
        max_row: Longest row of L below the diagonal (update-list padding).
        rowptr: CSR row pointers of L's strictly lower entries, of length N + 1.
        rowpos: CSC position of each strictly lower entry, sorted by row.
        value_target: Position in L's CSC arrays of each stored value of A.
        value_weight: Weight of each stored value: ½ off the diagonal for general
            storage, else 1.
        banded: Whether the numeric phase and Takahashi run on dense
            ``block_size × block_size`` blocks of a block-tridiagonal layout
            (chosen when the blocks cost at most four times the storage of
            L, as after RCM on a mesh) rather than on gathered windows.
        block_size: The bandwidth of L (at least 1).
        num_blocks: Number of diagonal blocks of the block-tridiagonal layout.
        block_pos: Position of each entry of L in the flat blocks.
        pad_pos: Position in the flat blocks of the unit diagonal that pads the
            last block.

    Examples:
        The path graph 0 − 1 − 2 − 3, in its natural order, does not fill in:

        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.arange(1, 4), np.arange(3), (4, 4), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern, ordering="natural")
        >>> symbolic
        SymbolicCholesky(n=4, nnz_L=7, ordering='natural', backend='jax', banded=True)
        >>> symbolic.parent.tolist()
        [1, 2, 3, -1]
        >>> symbolic.fill_ratio
        1.0
    """

    def __init__(
        self,
        pattern: SparsityPattern,
        ordering: Ordering,
        backend: Backend,
        perm: np.ndarray,
        *,
        banded: bool | None = None,
    ) -> None:
        n = pattern.shape[0]
        perm = np.asarray(perm, dtype=np.int64)
        iperm = np.empty(n, dtype=np.int64)
        iperm[perm] = np.arange(n)
        self.pattern = pattern
        self.ordering = ordering
        self.backend = backend
        self.n = n
        self.perm = perm.astype(np.int32)
        self.iperm = iperm.astype(np.int32)

        # --- lower triangle of P A Pᵀ (structure only), grouped by column ---
        permuted_rows = iperm[pattern.rows]
        permuted_cols = iperm[pattern.cols]
        lower_cols, lower_rows = (
            np.minimum(permuted_rows, permuted_cols),
            np.maximum(permuted_rows, permuted_cols),
        )
        entry_keys = np.unique(lower_cols * n + lower_rows)
        self.nnz_lower = int(entry_keys.shape[0])
        a_cols, a_rows = entry_keys // n, entry_keys % n
        a_ptr = np.searchsorted(a_cols, np.arange(n + 1))

        # --- column patterns of L by the elimination-tree union ---
        # In column order: every child c < j is finished before its parent j
        # needs it.
        parent = np.full(n, -1, dtype=np.int64)
        children: list[list[int]] = [[] for _ in range(n)]
        structs: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n
        for j in range(n):
            pieces = [a_rows[a_ptr[j] : a_ptr[j + 1]]]
            pieces.extend(structs[child][1:] for child in children[j])
            # struct(L_:,j) = union of struct([P A Pᵀ]_j:,j), struct(L_:,c) without c
            struct = pieces[0] if len(pieces) == 1 else _sorted_union(pieces)
            structs[j] = struct
            if struct.shape[0] > 1:
                # parent(j) = min{i > j : L_ij ≠ 0}, the first row below the diagonal
                parent[j] = struct[1]
                children[struct[1]].append(j)
        counts = np.array([s.shape[0] for s in structs], dtype=np.int64)
        colptr = np.concatenate([[0], np.cumsum(counts)])
        rowidx = np.concatenate(structs) if n else np.empty(0, dtype=np.int64)
        colidx = np.repeat(np.arange(n), counts)
        nnz = int(colptr[-1])

        self.parent = parent.astype(np.int32)
        self.colptr = colptr.astype(np.int32)
        self.rowidx = rowidx.astype(np.int32)
        self.colidx = colidx.astype(np.int32)
        self.nnz = nnz
        self.max_col = int(counts.max(initial=1))

        # --- row structure of L below the diagonal ---
        # The left-looking update list of each column j, the columns k < j with
        # L_jk ≠ 0, as CSR over the strictly lower entries: positions of L_jk
        # sorted by row j.
        below_diagonal = np.flatnonzero(rowidx != colidx)
        row_order = below_diagonal[
            np.lexsort((colidx[below_diagonal], rowidx[below_diagonal]))
        ]
        row_counts = np.bincount(rowidx[row_order], minlength=n)
        self.rowptr = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
        self.rowpos = row_order.astype(np.int32)
        self.max_row = int(max(row_counts.max(initial=0), 1))

        # --- scatter plan from the stored values to the lower triangle ---
        # lower(P A Pᵀ) = segment_sum(w · values, target) on L's pattern. Full
        # storage contributes ½ from each of (i, j) and (j, i), so the factored
        # matrix is the symmetric part ½(A + Aᵀ).
        self.value_target = _lower_positions(self, lower_cols, lower_rows)
        weight = np.ones(pattern.nnz)
        if not pattern.symmetric:
            weight[pattern.rows != pattern.cols] = 0.5
        self.value_weight = weight

        # --- banded layout ---
        # With bandwidth b, P A Pᵀ and L are block tridiagonal in b × b blocks,
        # and dense block kernels (BLAS) replace the gathers. RCM keeps a mesh's
        # fill inside a band, so the blocks cost a small multiple of the storage
        # of L; an arrow-shaped pattern (a dense row) does not.
        bandwidth = int((self.rowidx - self.colidx).max(initial=0))
        size = max(bandwidth, 1)
        blocks = -(-n // size) if n else 0
        if banded is None:
            banded = 2 * blocks * size * size <= 4 * max(nnz, 1)
        self.banded = bool(banded)
        self.block_size = size
        self.num_blocks = blocks
        block_rows, block_cols = self.rowidx // size, self.colidx // size
        local = (self.rowidx % size) * size + self.colidx % size
        offset = np.where(block_rows == block_cols, 0, blocks * size * size)
        self.block_pos = (offset + block_cols * size * size + local).astype(np.int64)
        pad = np.arange(n, blocks * size)
        self.pad_pos = ((pad // size) * size * size + (pad % size) * (size + 1)).astype(
            np.int64
        )

    # --- identity ---

    @property
    def _key(self) -> tuple[SparsityPattern, str, str, bool]:
        """What identifies the analysis: pattern, ordering, backend and layout."""
        return (self.pattern, self.ordering, self.backend, self.banded)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolicCholesky):
            return NotImplemented
        return self is other or self._key == other._key

    def __repr__(self) -> str:
        return (
            f"SymbolicCholesky(n={self.n}, nnz_L={self.nnz}, "
            f"ordering={self.ordering!r}, backend={self.backend!r}, "
            f"banded={self.banded})"
        )

    @property
    def fill_ratio(self) -> float:
        """How much the factor fills in, nnz(L) / nnz(tril(A))."""
        return self.nnz / max(self.nnz_lower, 1)

    # --- plans for the selected inverse ---

    @ft.cached_property
    def inverse_plan(self) -> tuple[SparsityPattern, np.ndarray]:
        """Pattern of L + Lᵀ in the original order, and the gather into it.

        Returns:
            Tuple ``(pattern, index)``: the symmetric (lower-triangle) pattern of
            Pᵀ (L + Lᵀ) P, and ``index`` such that its canonical values are
            ``z[index]`` for ``z`` on L's CSC pattern.
        """
        original_rows = self.perm[self.rowidx].astype(np.int64)
        original_cols = self.perm[self.colidx].astype(np.int64)
        lower = SparsityPattern(
            np.maximum(original_rows, original_cols),
            np.minimum(original_rows, original_cols),
            (self.n, self.n),
            symmetric=True,
        )
        positions = _positions(
            lower,
            np.maximum(original_rows, original_cols),
            np.minimum(original_rows, original_cols),
        )
        index = np.empty(self.nnz, dtype=np.int32)
        index[positions] = np.arange(self.nnz)
        return lower, index


def _lower_positions(
    sym: SymbolicCholesky, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """CSC positions in L of the permuted lower entries ``(hi, lo)``.

    Args:
        sym: The analysis so far: its ``colidx``, ``rowidx`` and ``n``.
        lo: Column of each entry, the smaller permuted index.
        hi: Row of each entry, the larger permuted index.

    Returns:
        The position of each entry in L's CSC arrays.
    """
    if lo.size == 0:
        return np.empty(0, dtype=np.int32)
    keys = sym.colidx.astype(np.int64) * sym.n + sym.rowidx
    positions = np.searchsorted(keys, lo * sym.n + hi)
    return positions.astype(np.int32)


def _sorted_union(pieces: list[np.ndarray]) -> np.ndarray:
    """Sorted union of index arrays (cheaper than ``np.unique`` per column).

    Args:
        pieces: The index arrays.

    Returns:
        Their sorted, duplicate-free union.
    """
    merged = np.sort(np.concatenate(pieces))
    keep = np.empty(merged.shape[0], dtype=bool)
    keep[0] = True
    np.not_equal(merged[1:], merged[:-1], out=keep[1:])
    return merged[keep]


def _check_pattern(pattern: SparsityPattern) -> None:
    """Reject a non-square pattern.

    Args:
        pattern: The pattern of the matrix to factor.

    Raises:
        ValueError: If the pattern is not square.
    """
    m, n = pattern.shape
    if m != n:
        raise ValueError(f"A Cholesky factor needs a square pattern, got {(m, n)}.")


def _symmetric_graph(pattern: SparsityPattern) -> Any:
    """The union of ``pattern`` and its transpose as a SciPy CSR graph.

    Args:
        pattern: Square sparsity pattern.

    Returns:
        The graph, a ``scipy.sparse.csr_matrix`` with its duplicates summed.
    """
    import scipy.sparse as sp

    n = pattern.shape[0]
    rows = np.concatenate([pattern.rows, pattern.cols])
    cols = np.concatenate([pattern.cols, pattern.rows])
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    graph.sum_duplicates()
    return graph


def _ordering(pattern: SparsityPattern, ordering: Ordering) -> np.ndarray:
    """The fill-reducing permutation of a pattern.

    Args:
        pattern: Square sparsity pattern.
        ordering: ``"natural"``, ``"rcm"`` or anything else for CHOLMOD's AMD; the
            caller has validated it.

    Returns:
        The permutation ``perm``.

    Raises:
        ImportError: For ``"amd"`` without ``scikit-sparse``.
    """
    n = pattern.shape[0]
    if ordering == "natural":
        return np.arange(n)
    if ordering == "rcm":
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        return np.asarray(
            reverse_cuthill_mckee(_symmetric_graph(pattern), symmetric_mode=True)
        )
    # lazy import, cycle: _sparse._cholmod -> _sparse._symbolic
    from gaussx._sparse._cholmod import cholmod_amd_ordering

    return cholmod_amd_ordering(pattern)


def symbolic_cholesky(
    pattern: SparsityPattern,
    *,
    ordering: Ordering = "rcm",
    backend: Backend = "jax",
) -> SymbolicCholesky:
    """Symbolic analysis of a sparse Cholesky factorisation (host, cached).

        P A Pᵀ = L Lᵀ

    Computes a fill-reducing permutation P, the elimination tree and the pattern
    of the factor L, plus the static, padded index plans that
    `gaussx.sparse_cholesky` and the Takahashi selected inverse gather through. It
    depends only on the pattern, so it runs once on the host and is cached per
    ``(pattern, ordering, backend)`` (the pattern's content hash): later calls,
    under ``jit`` or not, are a dictionary lookup.

    A symmetric pattern stores the lower triangle; a general pattern is
    symmetrised structurally, and its factor is that of the symmetric part
    ½(A + Aᵀ).

    Algorithm:
        1. Validate the options and that the pattern is square.
        2. Compute the permutation P from ``ordering``.
        3. Build the elimination tree and the columns of L by the tree union.
        4. Build the index plans and choose the layout by storage cost.

    Args:
        pattern: Square sparsity pattern of the matrix A to factor.
        ordering: ``"rcm"`` (reverse Cuthill-McKee from
            ``scipy.sparse.csgraph``; minimises bandwidth), ``"natural"`` (no
            permutation) or ``"amd"`` (approximate minimum degree, from
            CHOLMOD; needs ``scikit-sparse``). RCM fill grows like the
            bandwidth times N; AMD and nested dissection are much sparser
            on 2-D meshes.
        backend: ``"jax"`` (numeric factorisation as a ``lax.scan``; ``jit``,
            ``grad`` and ``vmap`` work) or ``"cholmod"`` (numeric
            factorisation by CHOLMOD on the host through
            ``jax.pure_callback``; CPU only, ``vmap`` runs sequentially; needs
            ``scikit-sparse``). Triangular solves, Takahashi and the gradients
            are the same JAX code for both.

    Returns:
        The cached `SymbolicCholesky`.

    Raises:
        ValueError: For a non-square pattern or an unknown option.
        ImportError: For ``"amd"`` or ``"cholmod"`` without ``scikit-sparse``.

    Examples:
        The path graph 0 − 1 − 2 − 3 − 4 (lower triangle, edges once) does not
        fill in, and the analysis is cached per pattern:

        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.arange(1, 5), np.arange(4), (5, 5), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern)
        >>> symbolic.nnz
        9
        >>> gaussx.symbolic_cholesky(pattern) is symbolic
        True

        An unknown option is rejected:

        >>> gaussx.symbolic_cholesky(
        ...     pattern, ordering="metis"
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: Unknown ordering 'metis'; expected one of
        ('rcm', 'natural', 'amd').

        >>> gaussx.symbolic_cholesky(pattern, backend="cuda")
        Traceback (most recent call last):
        ...
        ValueError: Unknown backend 'cuda'; expected one of ('jax', 'cholmod').

        So is a pattern that is not square:

        >>> wide = gaussx.SparsityPattern(np.array([0]), np.array([0]), (2, 3))
        >>> gaussx.symbolic_cholesky(wide)
        Traceback (most recent call last):
        ...
        ValueError: A Cholesky factor needs a square pattern, got (2, 3).
    """
    if ordering not in _ORDERINGS:
        raise ValueError(
            f"Unknown ordering {ordering!r}; expected one of {_ORDERINGS}."
        )
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {_BACKENDS}.")
    _check_pattern(pattern)
    return _symbolic_cached(pattern, ordering, backend)


@ft.lru_cache(maxsize=_CACHE_SIZE)
def _symbolic_cached(
    pattern: SparsityPattern, ordering: Ordering, backend: Backend
) -> SymbolicCholesky:
    """The analysis behind `symbolic_cholesky`, cached on its validated arguments.

    Args:
        pattern: Square sparsity pattern.
        ordering: The fill-reducing ordering.
        backend: The numeric backend.

    Returns:
        The `SymbolicCholesky`.

    Raises:
        ImportError: For ``"amd"`` or ``"cholmod"`` without ``scikit-sparse``.
    """
    if backend == "cholmod":
        # lazy import, cycle: _sparse._cholmod -> _sparse._symbolic
        from gaussx._sparse._cholmod import require_cholmod

        require_cholmod()
    return SymbolicCholesky(pattern, ordering, backend, _ordering(pattern, ordering))
