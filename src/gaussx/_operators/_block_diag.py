"""Block diagonal linear operator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


def _resolve_dtype(*operators: lx.AbstractLinearOperator) -> str:
    """Infer a common dtype string from the sub-operators' output structures."""
    dtypes = []
    for op in operators:
        struct = op.out_structure()
        leaves = jax.tree.leaves(struct)
        dtypes.extend(leaf.dtype for leaf in leaves)
    return str(jnp.result_type(*dtypes))


class BlockDiag(lx.AbstractLinearOperator):
    """Block diagonal operator ``diag(A₁, A₂, …, Aₖ)``.

    Each sub-operator acts on its own slice of the input vector.
    Matvec, transpose, logdet, solve, and cholesky all decompose
    per-block.

    Args:
        *operators: One or more ``lineax.AbstractLinearOperator`` instances
            forming the diagonal blocks.
    """

    operators: tuple[lx.AbstractLinearOperator, ...]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        *operators: lx.AbstractLinearOperator,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if len(operators) == 0:
            raise ValueError("BlockDiag requires at least one operator.")
        self.operators = operators
        self._in_size = sum(op.in_size() for op in operators)
        self._out_size = sum(op.out_size() for op in operators)
        self._dtype = _resolve_dtype(*operators)
        from gaussx._tags import block_diagonal_tag

        self.tags = _to_frozenset(tags) | {block_diagonal_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        # Use jax.lax.dynamic_slice with static offsets for JIT compatibility
        results = []
        offset = 0
        for op in self.operators:
            size = op.in_size()
            block = jax.lax.dynamic_slice(vector, (offset,), (size,))
            results.append(op.mv(block))
            offset += size
        return jnp.concatenate(results)

    def as_matrix(self) -> Float[Array, "m n"]:
        matrices = [op.as_matrix() for op in self.operators]
        return jax.scipy.linalg.block_diag(*matrices)

    def transpose(self) -> BlockDiag:
        return BlockDiag(
            *(op.T for op in self.operators),
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def _to_frozenset(x: object | frozenset[object]) -> frozenset[object]:
    """Convert a single tag or frozenset of tags to a frozenset."""
    if isinstance(x, frozenset):
        return x
    try:
        return frozenset(x)  # type: ignore[arg-type]
    except TypeError:
        return frozenset([x])
