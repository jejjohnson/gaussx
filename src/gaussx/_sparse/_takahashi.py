"""Takahashi selected inverse on the pattern of ``L + Lᵀ``."""

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
    """Entries of ``Z = A⁻¹`` on the pattern of ``L`` (lower triangle of ``L + Lᵀ``).

    The banded layout runs the block recursion of `gaussx.selected_inverse`
    on the factor's blocks; otherwise `_takahashi_windows`.

    Args:
        sym: The symbolic analysis.
        L: The factor's values on its CSC pattern.

    Returns:
        ``Z`` on ``L``'s CSC pattern, shape ``(nnz_L,)``.
    """
    if sym.banded:
        # lazy import, cycle: _linalg._selected_inverse -> _primitives._cholesky ->
        #   _sparse._factor -> _sparse._takahashi
        from gaussx._linalg._selected_inverse import _block_takahashi

        sigma = _block_takahashi(LowerBlockTriDiag(*to_blocks(sym, L)))
        return from_blocks(sym, sigma.diagonal, sigma.sub_diagonal)
    return _takahashi_windows(sym, L)


def _takahashi_windows(
    sym: SymbolicCholesky, L: Float[Array, " nnz_L"]
) -> Float[Array, " nnz_L"]:
    r"""Entries of ``Z = A⁻¹`` on the pattern of ``L`` (lower triangle of ``L + Lᵀ``).

    The backward recursion over columns, ancestors first,

    $$
    Z_{ij} = \frac{\delta_{ij}}{L_{jj}^2} - \frac{1}{L_{jj}}
    \sum_{k>j,\ k\in\operatorname{struct}(L_{:,j})} L_{kj}\,Z_{ki},
    \qquad i \in \operatorname{struct}(L_{:,j}),
    $$

    only reads ``Z`` on ``pattern(L + Lᵀ)``: the rows of column ``j`` below
    the diagonal form a clique of the filled graph, so every ``Z_ki`` it needs
    is an entry of an ancestor's column. Each step gathers that clique's block of
    ``Z`` through a row-to-slot workspace (no per-flop index plan) and does a
    dense matvec, for ``O(Σ_j |struct(L_{:,j})|²)`` work in all.
    """
    p = plans(sym)
    n, c = sym.n, sym.max_col
    dtype = L.dtype
    L_pad = jnp.concatenate([L, jnp.zeros(c, dtype)])
    rowidx = jnp.asarray(p.rowidx)
    col_start = jnp.asarray(p.col_start)
    col_end = jnp.asarray(p.col_end)

    def step(carry, j, size, reach):
        Z, slot = carry
        m = size - 1  # below-diagonal slots of the column
        if m == 0:  # diagonal-only columns: Z_jj = 1 / L_jj²
            start, _, vals, mask = _column(p, n, L_pad, j, size)
            col = 1 / vals[:1] ** 2
            old = _window(Z, start, size)
            Z = jax.lax.dynamic_update_slice(Z, jnp.where(mask, col, old), (start,))
            return (Z, slot), None
        slots = jnp.arange(m, dtype=jnp.int32)
        start, rows, vals, mask = _column(p, n, L_pad, j, size)
        below = rows[1:]  # the clique S (dummy row n past the column's end)
        slot = slot.at[below].set(slots)
        # Column k of S, from its diagonal down: Z[i, k] for i >= k, and every
        # i in S >= k is in it. Place each at (slot(k), slot(i)).
        idx = einx.add("s, c -> s c", col_start[below], jnp.arange(reach))
        k_mask = einx.less("s c, s -> s c", idx, col_end[below])
        k_rows = jnp.where(k_mask, rowidx[idx], n)
        k_vals = jnp.where(k_mask, Z[idx], 0)
        loc = slot[k_rows]
        valid = k_mask & (below[loc] == k_rows)
        row_slot = repeat(slots, "s -> s c", c=reach)
        block = (
            jnp.zeros((m, m), dtype).at[row_slot, loc].add(jnp.where(valid, k_vals, 0))
        )
        # Symmetrise: block holds Z[S_t, S_s] for S_t >= S_s, diagonal once.
        block = block + rearrange(block, "s t -> t s") - jnp.diag(jnp.diag(block))
        l = vals[1:] / vals[0]
        z_below = -einsum(block, l, "s t, t -> s")
        z_diag = 1 / vals[0] ** 2 - jnp.sum(l * z_below)
        col = jnp.concatenate([z_diag[None], z_below])
        old = _window(Z, start, size)
        Z = jax.lax.dynamic_update_slice(Z, jnp.where(mask, col, old), (start,))
        return (Z, slot), None

    carry = (jnp.zeros(sym.nnz + c, dtype), jnp.zeros(n + 1, jnp.int32))
    for bucket in reversed(p.buckets):  # ancestors first
        body = ft.partial(step, size=bucket.size, reach=bucket.reach)
        carry, _ = jax.lax.scan(body, carry, jnp.asarray(bucket.cols), reverse=True)
    return carry[0][: sym.nnz]
