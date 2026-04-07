"""Kernel matrix operator with efficient first-order autodiff."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PyTree

from gaussx._operators._block_diag import _to_frozenset


# ---------------------------------------------------------------------------
# Custom-JVP matvec
# ---------------------------------------------------------------------------


def _make_kernel_mv(kernel_fn: Callable) -> Callable:
    """Build a custom-JVP matvec function closed over *kernel_fn* (static).

    We close over ``kernel_fn`` so that ``jax.custom_jvp`` never tries to
    differentiate or trace through it as a positional argument.
    """

    def _kernel_mv_impl(
        params: PyTree,
        X1: Float[Array, "N D"],
        X2: Float[Array, "M D"],
        v: Float[Array, " M"],
    ) -> Float[Array, " N"]:
        """Compute ``K(X1, X2; params) @ v`` via scan."""

        def row_dot(x_i: Float[Array, " D"]) -> Float[Array, ""]:
            k_row = jax.vmap(lambda x_j: kernel_fn(params, x_i, x_j))(X2)
            return jnp.dot(k_row, v)

        def body_fn(
            carry: None, x_i: Float[Array, " D"]
        ) -> tuple[None, Float[Array, ""]]:
            return carry, row_dot(x_i)

        _, Kv = jax.lax.scan(body_fn, None, X1)
        return Kv

    @jax.custom_jvp
    def kernel_mv(
        params: PyTree,
        X1: Float[Array, "N D"],
        X2: Float[Array, "M D"],
        v: Float[Array, " M"],
    ) -> Float[Array, " N"]:
        return _kernel_mv_impl(params, X1, X2, v)

    @kernel_mv.defjvp
    def kernel_mv_jvp(
        primals: tuple[
            PyTree,
            Float[Array, "N D"],
            Float[Array, "M D"],
            Float[Array, " M"],
        ],
        tangents: tuple[
            PyTree,
            Float[Array, "N D"],
            Float[Array, "M D"],
            Float[Array, " M"],
        ],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        params, X1, X2, v = primals
        dparams, dX1, dX2, dv = tangents

        def row_jvp(
            x1_and_tangent: tuple[Float[Array, " D"], Float[Array, " D"]],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            x1_i, dx1_i = x1_and_tangent

            def kernel_with_jvp(
                x2_and_tangent: tuple[Float[Array, " D"], Float[Array, " D"]],
            ) -> tuple[Float[Array, ""], Float[Array, ""]]:
                x2_j, dx2_j = x2_and_tangent
                return jax.jvp(
                    kernel_fn,
                    (params, x1_i, x2_j),
                    (dparams, dx1_i, dx2_j),
                )

            k_row, dk_row = jax.vmap(kernel_with_jvp)((X2, dX2))
            primal = jnp.dot(k_row, v)
            tangent = jnp.dot(dk_row, v) + jnp.dot(k_row, dv)
            return primal, tangent

        primal_out, tangent_out = jax.vmap(row_jvp)((X1, dX1))
        return primal_out, tangent_out

    return kernel_mv


# ---------------------------------------------------------------------------
# Operator class
# ---------------------------------------------------------------------------


class KernelOperator(lx.AbstractLinearOperator):
    r"""Kernel matrix operator with efficient first-order autodiff.

    Represents the matrix ``K`` where ``K[i, j] = kernel_fn(params, X1[i], X2[j])``.
    The matvec ``K @ v`` is computed via scan (O(N) memory), and a
    ``jax.custom_jvp`` keeps first-order autodiff efficient without
    materializing Jacobians.

    Args:
        kernel_fn: Kernel function ``k(params, x, x') -> scalar``.  The first
            argument is a pytree of hyperparameters.
        X1: First set of data points, shape ``(N, D)``.
        X2: Second set of data points, shape ``(M, D)``.
        params: Pytree of kernel hyperparameters (differentiable).
        tags: Optional lineax structural tags.
    """

    kernel_fn: Callable = eqx.field(static=True)
    X1: Float[Array, "N D"]
    X2: Float[Array, "M D"]
    params: Any  # pytree of kernel hyperparameters
    _nrows: int = eqx.field(static=True)
    _ncols: int = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)
    _kernel_mv: Callable = eqx.field(static=True)

    def __init__(
        self,
        kernel_fn: Callable,
        X1: Float[Array, "N D"],
        X2: Float[Array, "M D"],
        params: Any,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        self.kernel_fn = kernel_fn
        self.X1 = X1
        self.X2 = X2
        self.params = params
        self._nrows = X1.shape[0]
        self._ncols = X2.shape[0]
        normalized_tags = _to_frozenset(tags)
        if lx.positive_semidefinite_tag in normalized_tags:
            if self._nrows != self._ncols:
                raise ValueError(
                    "positive_semidefinite_tag is only valid for square operators."
                )
            normalized_tags = normalized_tags | {lx.symmetric_tag}
        self.tags = normalized_tags
        self._kernel_mv = _make_kernel_mv(kernel_fn)

    def mv(self, vector: Float[Array, " M"]) -> Float[Array, " N"]:
        """Compute ``K @ v`` via scan with custom JVP support."""
        return self._kernel_mv(self.params, self.X1, self.X2, vector)

    def as_matrix(self) -> Float[Array, "N M"]:
        """Materialize the full kernel matrix."""
        return jax.vmap(
            lambda x_i: jax.vmap(lambda x_j: self.kernel_fn(self.params, x_i, x_j))(
                self.X2
            )
        )(self.X1)

    def transpose(self) -> KernelOperator:
        """Return the transpose operator (X1, X2 swapped, kernel transposed)."""
        if lx.symmetric_tag in self.tags:
            return self
        return KernelOperator(
            lambda p, x_i, x_j: self.kernel_fn(p, x_j, x_i),
            self.X2,
            self.X1,
            self.params,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._ncols,), self.X1.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._nrows,), self.X1.dtype)
