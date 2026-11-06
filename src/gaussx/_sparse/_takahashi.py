"""Takahashi selected inverse Z = (P A Pᵀ)⁻¹ on the pattern of L + Lᵀ."""

from __future__ import annotations

import functools as ft

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange, repeat
from gaussx._operators._block_tridiag import LowerBlockTriDiag
from gaussx._sparse._numeric import _column, _window, from_blocks, plans, to_blocks
from gaussx._sparse._symbolic import SymbolicCholesky


@ft.partial(jax.jit, static_argnums=0)
def takahashi(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Entries of Z = (P A Pᵀ)⁻¹ on the pattern of L (lower triangle of L + Lᵀ).

    The banded layout runs the block recursion of `gaussx.selected_inverse`
    on the factor's blocks; otherwise `_takahashi_windows`.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.

    Returns:
        Z on L's CSC pattern.

    Examples:
        For A = 4 I in two dimensions, L = 2 I and Z = ¼ I:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> pattern = gaussx.SparsityPattern(
        ...     np.arange(2), np.arange(2), (2, 2), symmetric=True
        ... )
        >>> symbolic = gaussx.symbolic_cholesky(pattern)
        >>> Z = takahashi(symbolic, jnp.array([2.0, 2.0]))
        >>> bool(jnp.allclose(Z, jnp.array([0.25, 0.25])))
        True
    """
    if sym.banded:
        # lazy import, cycle: _linalg._selected_inverse -> _primitives._cholesky ->
        #   _sparse._factor -> _sparse._takahashi
        from gaussx._linalg._selected_inverse import _block_takahashi

        diagonal, sub = to_blocks(sym, L)
        factor = LowerBlockTriDiag(diagonal, sub)
        band = _block_takahashi(factor)
        return from_blocks(sym, band.diagonal, band.sub_diagonal)
    return _takahashi_windows(sym, L)


def _takahashi_windows(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    """Entries of Z = (P A Pᵀ)⁻¹ on the pattern of L, by gathered windows.

        Z_ij = δ_ij / L_jj² − (1 / L_jj) ∑ L_kj Z_ki

    summed over k > j in struct(L_:,j), for i ∈ struct(L_:,j). The backward
    recursion over columns, ancestors first, only reads Z on pattern(L + Lᵀ): the
    rows of column j below the diagonal form a clique of the filled graph, so every
    Z_ki it needs is an entry of an ancestor's column.

    Algorithm:
        1. Sweep the buckets in reverse, ancestors first, column by column.
        2. Gather the clique's block of Z through a row-to-slot workspace (no
           per-flop index plan), and symmetrise it.
        3. Compute the column below the diagonal by a dense matvec, then Z_jj.

    The work is O(∑ⱼ |struct(L_:,j)|²) in all.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.

    Returns:
        Z on L's CSC pattern.
    """
    p = plans(sym)
    n, max_col = sym.n, sym.max_col
    dtype = L.dtype
    L_pad = jnp.concatenate([L, jnp.zeros(max_col, dtype)])
    rowidx = jnp.asarray(p.rowidx)
    col_start = jnp.asarray(p.col_start)
    col_end = jnp.asarray(p.col_end)

    def step(carry, j, size, reach):
        Z, slot = carry
        # The below-diagonal slots of the column.
        num_below = size - 1
        if num_below == 0:
            # Diagonal-only columns.
            start, _, vals, mask = _column(p, n, L_pad, j, size)
            # Z_jj = 1 / L_jj²
            col = 1 / vals[:1] ** 2
            old = _window(Z, start, size)
            Z = jax.lax.dynamic_update_slice(Z, jnp.where(mask, col, old), (start,))
            return (Z, slot), None
        slots = jnp.arange(num_below, dtype=jnp.int32)
        start, rows, vals, mask = _column(p, n, L_pad, j, size)
        # The clique S (dummy row n past the column's end).
        below = rows[1:]
        slot = slot.at[below].set(slots)

        # --- gather the clique's block of Z ---
        # Column k of S, from its diagonal down: Z_ik for i ≥ k, and every i in
        # S ≥ k is in it. Place each at (slot(k), slot(i)).
        idx = einx.add(
            "clique, entry -> clique entry", col_start[below], jnp.arange(reach)
        )
        k_mask = einx.less("clique entry, clique -> clique entry", idx, col_end[below])
        k_rows = jnp.where(k_mask, rowidx[idx], n)
        k_vals = jnp.where(k_mask, Z[idx], 0)
        loc = slot[k_rows]
        in_clique = below[loc] == k_rows
        valid = k_mask & in_clique
        row_slot = repeat(slots, "clique -> clique entry", entry=reach)
        empty_block = jnp.zeros((num_below, num_below), dtype)
        kept_vals = jnp.where(valid, k_vals, 0)
        block = empty_block.at[row_slot, loc].add(kept_vals)
        # Symmetrise: block holds Z[S_t, S_s] for S_t ≥ S_s, diagonal once.
        # Z_S,S = B + Bᵀ − diag(diag(B)), for the gathered block B
        block_transpose = rearrange(block, "row col -> col row")
        doubled = block + block_transpose
        diagonal = jnp.diag(block)
        diagonal_matrix = jnp.diag(diagonal)
        block = doubled - diagonal_matrix

        # --- the column of Z ---
        # l_S = L_S,j / L_jj
        scaled = vals[1:] / vals[0]
        # Z_S,S l_S
        block_scaled = einsum(block, scaled, "row col, col -> row")
        # Z_S,j = −Z_S,S l_S
        z_below = -block_scaled
        # 1 / L_jj²
        inverse_square = 1 / vals[0] ** 2
        # l_Sᵀ Z_S,j
        cross = scaled * z_below
        correction = jnp.sum(cross)
        # Z_jj = 1 / L_jj² − l_Sᵀ Z_S,j
        z_diag = inverse_square - correction
        z_diag_single = rearrange(z_diag, " -> 1")
        col = jnp.concatenate([z_diag_single, z_below])
        old = _window(Z, start, size)
        Z = jax.lax.dynamic_update_slice(Z, jnp.where(mask, col, old), (start,))
        return (Z, slot), None

    carry = (jnp.zeros(sym.nnz + max_col, dtype), jnp.zeros(n + 1, jnp.int32))
    # Ancestors first.
    for bucket in reversed(p.buckets):
        body = ft.partial(step, size=bucket.size, reach=bucket.reach)
        carry, _ = jax.lax.scan(body, carry, jnp.asarray(bucket.cols), reverse=True)
    return carry[0][: sym.nnz]
