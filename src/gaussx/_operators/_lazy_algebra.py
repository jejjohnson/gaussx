"""Lazy algebra operators: Sum, Scaled, Product."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset


class SumOperator(lx.AbstractLinearOperator):
    """Lazy sum ``(A + B + …) v = A v + B v + …``.

    Defers materialization so that structured sub-operators keep their
    efficient matvec. All operators must have the same input and output
    sizes.

    Args:
        *operators: Two or more ``lineax.AbstractLinearOperator`` instances
            with matching shapes.
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
            raise ValueError("SumOperator requires at least two operators.")
        in0 = operators[0].in_size()
        out0 = operators[0].out_size()
        for i, op in enumerate(operators[1:], 1):
            if op.in_size() != in0 or op.out_size() != out0:
                raise ValueError(
                    f"Shape mismatch: operator 0 has shape ({out0}, {in0}) "
                    f"but operator {i} has shape ({op.out_size()}, {op.in_size()})."
                )
        self.operators = operators
        self._in_size = in0
        self._out_size = out0
        self._dtype = _resolve_dtype(*operators)
        self.tags = _to_frozenset(tags)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        result = self.operators[0].mv(vector)
        for op in self.operators[1:]:
            result = result + op.mv(vector)
        return result

    def as_matrix(self) -> Float[Array, "m n"]:
        result = self.operators[0].as_matrix()
        for op in self.operators[1:]:
            result = result + op.as_matrix()
        return result

    def transpose(self) -> SumOperator:
        return SumOperator(
            *(op.T for op in self.operators),
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


class ScaledOperator(lx.AbstractLinearOperator):
    """Lazy scalar multiply ``(c A) v = c (A v)``.

    Args:
        operator: A ``lineax.AbstractLinearOperator``.
        scalar: A scalar multiplier.
    """

    operator: lx.AbstractLinearOperator
    scalar: Float[Array, ""]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        operator: lx.AbstractLinearOperator,
        scalar: float | Float[Array, ""],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        self.operator = operator
        self.scalar = jnp.asarray(scalar).reshape(())
        self._in_size = operator.in_size()
        self._out_size = operator.out_size()
        self._dtype = jnp.result_type(
            jnp.dtype(_resolve_dtype(operator)),
            self.scalar,
        ).name
        self.tags = _to_frozenset(tags)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.scalar * self.operator.mv(vector)

    def as_matrix(self) -> Float[Array, "m n"]:
        return self.scalar * self.operator.as_matrix()

    def transpose(self) -> ScaledOperator:
        return ScaledOperator(
            self.operator.T,
            self.scalar,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


class ProductOperator(lx.AbstractLinearOperator):
    """Lazy matmul ``(A B) v = A (B v)``.

    The inner dimension must match: ``left.in_size() == right.out_size()``.

    Args:
        left: The left operator A.
        right: The right operator B.
    """

    left: lx.AbstractLinearOperator
    right: lx.AbstractLinearOperator
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        left: lx.AbstractLinearOperator,
        right: lx.AbstractLinearOperator,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if left.in_size() != right.out_size():
            raise ValueError(
                f"Inner dimension mismatch: left.in_size()={left.in_size()} "
                f"!= right.out_size()={right.out_size()}."
            )
        self.left = left
        self.right = right
        self._in_size = right.in_size()
        self._out_size = left.out_size()
        self._dtype = _resolve_dtype(left, right)
        self.tags = _to_frozenset(tags)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.left.mv(self.right.mv(vector))

    def as_matrix(self) -> Float[Array, "m n"]:
        return self.left.as_matrix() @ self.right.as_matrix()

    def transpose(self) -> ProductOperator:
        return ProductOperator(
            self.right.T,
            self.left.T,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))
