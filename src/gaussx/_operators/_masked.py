"""Masked linear operator — row/column sub-selection of a base operator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._operators._block_diag import _to_frozenset


class MaskedOperator(lx.AbstractLinearOperator):
    """Row/column-masked view of a base operator.

    Given a base operator A of shape ``(N, N)`` and boolean masks,
    produces the sub-matrix ``A[row_mask][:, col_mask]``.

    Matvec is computed without materializing the sub-matrix:
    zero-pad input to full size, apply base matvec, then extract
    masked rows.

    Args:
        base: The underlying ``(N, N)`` linear operator.
        row_mask: Boolean mask of length N selecting output rows.
        col_mask: Boolean mask of length N selecting input columns.
    """

    base: lx.AbstractLinearOperator
    row_mask: Bool[Array, " N"]
    col_mask: Bool[Array, " N"]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        base: lx.AbstractLinearOperator,
        row_mask: Bool[Array, " N"],
        col_mask: Bool[Array, " N"],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if base.in_size() != base.out_size():
            raise ValueError(
                f"Base operator must be square, got in_size={base.in_size()}, "
                f"out_size={base.out_size()}."
            )
        n = base.in_size()
        if row_mask.shape != (n,) or col_mask.shape != (n,):
            raise ValueError(
                f"Masks must have shape ({n},), got row_mask={row_mask.shape}, "
                f"col_mask={col_mask.shape}."
            )
        self.base = base
        self.row_mask = jnp.asarray(row_mask, dtype=bool)
        self.col_mask = jnp.asarray(col_mask, dtype=bool)
        self._in_size = int(jnp.sum(col_mask))
        self._out_size = int(jnp.sum(row_mask))
        struct = base.out_structure()
        leaves = jax.tree.leaves(struct)
        self._dtype = str(leaves[0].dtype)
        self.tags = _to_frozenset(tags)

    def mv(self, vector: Float[Array, " m"]) -> Float[Array, " k"]:
        # Scatter input into full-size vector at col_mask positions
        n = self.base.in_size()
        col_indices = jnp.where(self.col_mask, size=self._in_size)[0]
        full_v = jnp.zeros(n, dtype=vector.dtype).at[col_indices].set(vector)
        # Apply base operator
        full_out = self.base.mv(full_v)
        # Gather output at row_mask positions
        row_indices = jnp.where(self.row_mask, size=self._out_size)[0]
        return full_out[row_indices]

    def as_matrix(self) -> Float[Array, "k m"]:
        full = self.base.as_matrix()
        row_indices = jnp.where(self.row_mask, size=self._out_size)[0]
        col_indices = jnp.where(self.col_mask, size=self._in_size)[0]
        return full[jnp.ix_(row_indices, col_indices)]

    def transpose(self) -> MaskedOperator:
        return MaskedOperator(
            self.base.T,
            self.col_mask,
            self.row_mask,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))
