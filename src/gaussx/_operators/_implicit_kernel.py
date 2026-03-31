"""Implicit kernel linear operator — matrix-free kernel matvec."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset


class ImplicitKernelOperator(lx.AbstractLinearOperator):
    r"""Matrix-free kernel operator: ``(K + sigma^2 I) v`` via sequential scan.

    Computes the kernel matvec without materializing the ``N x N`` kernel
    matrix, using ``O(N)`` memory instead of ``O(N^2)``.  Each element of
    the output is computed as::

        y_i = \sum_j k(x_i, x_j) v_j + sigma^2 v_i

    The scan-based implementation is compatible with CG / BBMM solvers
    that only need matvec access.

    Args:
        kernel_fn: Kernel function ``k(x, x') -> scalar``.
        X: Training points, shape ``(N, D)``.
        noise_var: Diagonal noise variance ``sigma^2``.
    """

    kernel_fn: Callable = eqx.field(static=True)
    X: Float[Array, "N D"]
    noise_var: float = eqx.field(static=True)
    _size: int = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        kernel_fn: Callable[[Float[Array, " D"], Float[Array, " D"]], Float[Array, ""]],
        X: Float[Array, "N D"],
        noise_var: float = 0.0,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        self.kernel_fn = kernel_fn
        self.X = X
        self.noise_var = noise_var
        self._size = X.shape[0]
        normalized_tags = _to_frozenset(tags)
        if lx.positive_semidefinite_tag in normalized_tags:
            normalized_tags = normalized_tags | {lx.symmetric_tag}
        self.tags = normalized_tags

    def mv(self, vector: Float[Array, " N"]) -> Float[Array, " N"]:
        """Compute ``(K + sigma^2 I) @ v`` via scan over data points."""

        def row_dot(x_i: Float[Array, " D"]) -> Float[Array, ""]:
            # k(x_i, x_j) for all j, then dot with v
            k_row = jax.vmap(lambda x_j: self.kernel_fn(x_i, x_j))(self.X)
            return jnp.dot(k_row, vector)

        # Use lax.scan so each row's kernel vector is produced and reduced
        # immediately, avoiding an N x N intermediate from nested vmap.
        def body_fn(
            carry: None, x_i: Float[Array, " D"]
        ) -> tuple[None, Float[Array, ""]]:
            return carry, row_dot(x_i)

        _, Kv = jax.lax.scan(body_fn, None, self.X)
        if self.noise_var != 0.0:
            Kv = Kv + self.noise_var * vector
        return Kv

    def as_matrix(self) -> Float[Array, "N N"]:
        """Materialize the full kernel matrix (for debugging/testing)."""
        K = jax.vmap(
            lambda x_i: jax.vmap(lambda x_j: self.kernel_fn(x_i, x_j))(self.X)
        )(self.X)
        if self.noise_var != 0.0:
            K = K + self.noise_var * jnp.eye(self._size)
        return K

    def transpose(self) -> ImplicitKernelOperator:
        if lx.symmetric_tag in self.tags:
            return self
        return ImplicitKernelOperator(
            lambda x_i, x_j: self.kernel_fn(x_j, x_i),
            self.X,
            noise_var=self.noise_var,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), self.X.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), self.X.dtype)
