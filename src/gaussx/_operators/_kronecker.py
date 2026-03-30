"""Kronecker product linear operator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from einops import rearrange
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset


class Kronecker(lx.AbstractLinearOperator):
    """Kronecker product operator ``A₁ ⊗ A₂ ⊗ … ⊗ Aₖ``.

    Matvec uses Roth's column lemma for efficient computation without
    materializing the full Kronecker product. For two factors A (m x n)
    and B (p x q), the product (A kron B) vec(X) is computed as
    vec(B X A^T) where X is reshaped to (q, n).

    Complexity: O(sum n_i^3) instead of O((prod n_i)^2) for the naive approach.

    Args:
        *operators: Two or more ``lineax.AbstractLinearOperator`` instances.
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
        if len(operators) < 2:
            raise ValueError("Kronecker requires at least two operators.")
        self.operators = operators
        self._in_size = _prod(op.in_size() for op in operators)
        self._out_size = _prod(op.out_size() for op in operators)
        self._dtype = _resolve_dtype(*operators)
        from gaussx._tags import kronecker_tag

        self.tags = _to_frozenset(tags) | {kronecker_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        return _kronecker_mv(self.operators, vector)

    def as_matrix(self) -> Float[Array, "m n"]:
        result = self.operators[0].as_matrix()
        for op in self.operators[1:]:
            result = jnp.kron(result, op.as_matrix())
        return result

    def transpose(self) -> Kronecker:
        return Kronecker(
            *(op.T for op in self.operators),
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def _kronecker_mv(
    operators: tuple[lx.AbstractLinearOperator, ...],
    vector: Float[Array, " n"],
) -> Float[Array, " m"]:
    """Roth's column lemma for multi-factor Kronecker matvec.

    For (A₁ ⊗ A₂ ⊗ … ⊗ Aₖ) v, we process factors right-to-left.
    At each step, reshape the current vector, apply one factor via
    matmul, and flatten back.
    """
    x = vector
    # Process factors from right to left
    for i in range(len(operators) - 1, -1, -1):
        op = operators[i]
        n_in = op.in_size()
        # Reshape to (rest, n_in), apply op to each row
        x = rearrange(x, "(r c) -> r c", c=n_in)
        # Apply operator: each row gets multiplied by op^T
        # (A ⊗ B)vec(X) = vec(B X A^T) — we process one factor at a time
        mat = op.as_matrix()
        # x has shape (rest, n_in), mat has shape (n_out, n_in)
        # We want x @ mat^T = (rest, n_out)
        x = x @ mat.T
        # Flatten back, but now transpose the ordering for the next factor
        x = rearrange(x, "r c -> (c r)")
    return x


def _prod(iterable) -> int:
    """Product of an iterable of integers."""
    result = 1
    for x in iterable:
        result *= x
    return result
