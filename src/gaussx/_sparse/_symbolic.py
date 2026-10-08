"""Host-side symbolic Cholesky analysis, computed once per sparsity pattern.

Everything here is NumPy / SciPy on the host. The analysis depends only on the
`SparsityPattern` and the ordering, never on the values, so it is cached per
``(pattern, ordering, backend)`` and reused across ``jit`` calls, Newton steps,
θ-points and ``vmap``-ped datasets.
"""

from __future__ import annotations

import functools as ft
from typing import Literal

import numpy as np

from gaussx._operators._sparse import SparsityPattern, _positions


Ordering = Literal["rcm", "natural", "amd"]
Backend = Literal["jax", "cholmod"]

_ORDERINGS = ("rcm", "natural", "amd")
_BACKENDS = ("jax", "cholmod")
_CACHE_SIZE = 64


class SymbolicCholesky:
    r"""Symbolic Cholesky factor of a symmetric sparsity pattern.

    For the permuted matrix ``A = P Q Pᵀ`` (``A[k, l] = Q[perm[k], perm[l]]``)
    it holds the elimination tree and the pattern of ``L`` (``A = L Lᵀ``) in
    compressed sparse column (CSC) form, with each column's rows sorted so the
    diagonal comes first, together with the static index plans the numeric
    phase, the triangular solves and the Takahashi recursion gather through.
    The column patterns follow the elimination tree,

    $$
    \operatorname{struct}(L_{:,j}) = \operatorname{struct}(A_{j:,j})\ \cup
    \bigcup_{\operatorname{parent}(c)=j}\operatorname{struct}(L_{:,c})\setminus\{c\},
    \qquad \operatorname{parent}(j) = \min\{i>j : L_{ij}\neq 0\}.
    $$

    Build it with `gaussx.symbolic_cholesky` (cached), not directly. It is
    hashable and compared by ``(pattern, ordering, backend, banded)``, so it sits in
    a static field and causes no retrace for equal inputs.

    Args:
        pattern: Square sparsity pattern to analyse.
        ordering: Name of the ordering that produced ``perm``.
        backend: Numeric backend, ``"jax"`` or ``"cholmod"``.
        perm: Fill-reducing permutation, ``A = Q[perm][:, perm]``.
        banded: Force the banded (``True``) or windowed (``False``) layout;
            ``None`` picks by storage cost.

    Attributes:
        pattern: The pattern it was computed for.
        ordering: The fill-reducing ordering used.
        backend: The numeric backend (``"jax"`` or ``"cholmod"``).
        n: Matrix size.
        perm: ``A = Q[perm][:, perm]``, shape ``(n,)``.
        iperm: The inverse permutation, ``iperm[perm] = arange(n)``.
        parent: Elimination tree, ``parent[j]`` (``-1`` at a root).
        colptr: CSC column pointers of ``L``, shape ``(n + 1,)``.
        rowidx: CSC row indices of ``L``, shape ``(nnz_L,)``.
        colidx: Column of each entry of ``L``, shape ``(nnz_L,)``.
        nnz: Number of entries of ``L`` (diagonal included): the fill.
        nnz_lower: Number of entries in the lower triangle of ``Q``.
        max_col: Longest column of ``L`` (padded window of the column plans).
        max_row: Longest row of ``L`` below the diagonal (update-list padding).
        banded: Whether the numeric phase and Takahashi run on dense
            ``block_size × block_size`` blocks of a block-tridiagonal layout
            (chosen when the blocks cost at most four times the storage of
            ``L``, as after RCM on a mesh) rather than on gathered windows.
        block_size: The bandwidth of ``L`` (at least 1).
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

        # Lower triangle of A = P Q Pᵀ (structure only), grouped by column.
        r = iperm[pattern.rows]
        c = iperm[pattern.cols]
        lo, hi = np.minimum(r, c), np.maximum(r, c)
        keys = np.unique(lo * n + hi)
        self.nnz_lower = int(keys.shape[0])
        a_cols, a_rows = keys // n, keys % n
        a_ptr = np.searchsorted(a_cols, np.arange(n + 1))

        # Column patterns of L by the elimination-tree union, in column order:
        # every child c < j is finished before its parent j needs it.
        parent = np.full(n, -1, dtype=np.int64)
        children: list[list[int]] = [[] for _ in range(n)]
        structs: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n
        for j in range(n):
            pieces = [a_rows[a_ptr[j] : a_ptr[j + 1]]]
            pieces.extend(structs[child][1:] for child in children[j])
            struct = pieces[0] if len(pieces) == 1 else _sorted_union(pieces)
            structs[j] = struct
            if struct.shape[0] > 1:
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

        # Row structure of L below the diagonal (the left-looking update list
        # of each column j: the columns k < j with L[j, k] != 0), as CSR over
        # the strictly lower entries: positions of L[j, k] sorted by row j.
        off = np.flatnonzero(rowidx != colidx)
        order = off[np.lexsort((colidx[off], rowidx[off]))]
        row_counts = np.bincount(rowidx[order], minlength=n)
        self.rowptr = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
        self.rowpos = order.astype(np.int32)
        self.max_row = int(max(row_counts.max(initial=0), 1))

        # Scatter plan from the operator's stored values to the lower
        # triangle of A on L's pattern: A_lower = segment_sum(w · values, target).
        # Full storage contributes ½ from each of (i, j) and (j, i), so the
        # factored matrix is the symmetric part ½(Q + Qᵀ).
        self.value_target = _lower_positions(self, lo, hi)
        weight = np.ones(pattern.nnz)
        if not pattern.symmetric:
            weight[pattern.rows != pattern.cols] = 0.5
        self.value_weight = weight

        # Banded layout: with bandwidth b, A and L are block tridiagonal in
        # b × b blocks, and dense block kernels (BLAS) replace the gathers.
        # RCM keeps a mesh's fill inside a band, so the blocks cost a small
        # multiple of the storage of L; an arrow-shaped pattern (a dense row)
        # does not.
        b = int((self.rowidx - self.colidx).max(initial=0))
        size = max(b, 1)
        blocks = -(-n // size) if n else 0
        if banded is None:
            banded = 2 * blocks * size * size <= 4 * max(nnz, 1)
        self.banded = bool(banded)
        self.block_size = size
        self.num_blocks = blocks
        kr, kc = self.rowidx // size, self.colidx // size
        local = (self.rowidx % size) * size + self.colidx % size
        offset = np.where(kr == kc, 0, blocks * size * size)
        self.block_pos = (offset + kc * size * size + local).astype(np.int64)
        pad = np.arange(n, blocks * size)
        self.pad_pos = ((pad // size) * size * size + (pad % size) * (size + 1)).astype(
            np.int64
        )

    # -- identity -----------------------------------------------------------

    @property
    def _key(self) -> tuple[SparsityPattern, str, str, bool]:
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
        """``nnz(L) / nnz(tril(Q))``: how much the factor fills in."""
        return self.nnz / max(self.nnz_lower, 1)

    # -- plans for the selected inverse -------------------------------------

    @ft.cached_property
    def inverse_plan(self) -> tuple[SparsityPattern, np.ndarray]:
        """Pattern of ``L + Lᵀ`` in the original order, and the gather into it.

        Returns ``(pattern, index)``: the symmetric (lower-triangle) pattern of
        ``Pᵀ (L + Lᵀ) P``, and ``index`` such that its canonical values are
        ``z[index]`` for ``z`` on ``L``'s CSC pattern.
        """
        r = self.perm[self.rowidx].astype(np.int64)
        c = self.perm[self.colidx].astype(np.int64)
        lower = SparsityPattern(
            np.maximum(r, c), np.minimum(r, c), (self.n, self.n), symmetric=True
        )
        pos = _positions(lower, np.maximum(r, c), np.minimum(r, c))
        index = np.empty(self.nnz, dtype=np.int32)
        index[pos] = np.arange(self.nnz)
        return lower, index


def _lower_positions(
    sym: SymbolicCholesky, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """CSC positions in ``L`` of the permuted lower entries ``(hi, lo)``."""
    if lo.size == 0:
        return np.empty(0, dtype=np.int32)
    keys = sym.colidx.astype(np.int64) * sym.n + sym.rowidx
    pos = np.searchsorted(keys, lo * sym.n + hi)
    return pos.astype(np.int32)


def _sorted_union(pieces: list[np.ndarray]) -> np.ndarray:
    """Sorted union of index arrays (cheaper than ``np.unique`` per column)."""
    merged = np.sort(np.concatenate(pieces))
    keep = np.empty(merged.shape[0], dtype=bool)
    keep[0] = True
    np.not_equal(merged[1:], merged[:-1], out=keep[1:])
    return merged[keep]


def _check_pattern(pattern: SparsityPattern) -> None:
    m, n = pattern.shape
    if m != n:
        raise ValueError(f"A Cholesky factor needs a square pattern, got {(m, n)}.")


def _symmetric_graph(pattern: SparsityPattern):
    """The union of ``pattern`` and its transpose as a SciPy CSR graph."""
    import scipy.sparse as sp

    n = pattern.shape[0]
    rows = np.concatenate([pattern.rows, pattern.cols])
    cols = np.concatenate([pattern.cols, pattern.rows])
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    graph.sum_duplicates()
    return graph


def _ordering(pattern: SparsityPattern, ordering: Ordering) -> np.ndarray:
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
    r"""Symbolic analysis of a sparse Cholesky factorisation (host, cached).

    Computes a fill-reducing permutation, the elimination tree and the
    pattern of the factor ``L`` of ``P Q Pᵀ = L Lᵀ``, plus the static,
    padded index plans that `gaussx.sparse_cholesky` and the Takahashi
    selected inverse gather through. It depends only on the pattern, so it
    runs once on the host and is cached per ``(pattern, ordering, backend)``
    (the pattern's content hash): later calls, under ``jit`` or not, are a
    dictionary lookup.

    A symmetric pattern stores the lower triangle; a general pattern is
    symmetrised structurally, and its factor is that of the symmetric part
    ``½(Q + Qᵀ)``.

    Args:
        pattern: Square sparsity pattern of the matrix to factor.
        ordering: ``"rcm"`` (reverse Cuthill-McKee from
            ``scipy.sparse.csgraph``; minimises bandwidth), ``"natural"`` (no
            permutation) or ``"amd"`` (approximate minimum degree, from
            CHOLMOD; needs ``scikit-sparse``). RCM fill grows like the
            bandwidth times ``n``; AMD and nested dissection are much sparser
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
        ```python
        import numpy as np
        import gaussx

        # Path graph 0 - 1 - 2 - 3 - 4 (lower triangle, edges once)
        p = gaussx.SparsityPattern(
            np.arange(1, 5), np.arange(4), (5, 5), symmetric=True
        )
        sym = gaussx.symbolic_cholesky(p)
        assert sym.nnz == 9  # a tridiagonal matrix does not fill in
        assert gaussx.symbolic_cholesky(p) is sym  # cached per pattern
        ```
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
    if backend == "cholmod":
        # lazy import, cycle: _sparse._cholmod -> _sparse._symbolic
        from gaussx._sparse._cholmod import require_cholmod

        require_cholmod()
    return SymbolicCholesky(pattern, ordering, backend, _ordering(pattern, ordering))
