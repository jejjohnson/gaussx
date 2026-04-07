"""Implicit cross-kernel linear operator — matrix-free rectangular kernel matvec."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset


class ImplicitCrossKernelOperator(lx.AbstractLinearOperator):
    r"""Matrix-free rectangular kernel operator ``K(X, Z) \cdot v``.

    Computes the cross-kernel matvec without materializing the full
    ``N x M`` kernel matrix, using a batched scan that keeps peak memory
    at ``O(batch\_size \times M)`` per step.

    Forward matvec (``mv``)::

        y_i = \sum_j k(x_i, z_j) \cdot v_j

    maps an ``M``-vector to an ``N``-vector.

    Adjoint / transpose computes ``K^T u = K(Z, X) u``, mapping an
    ``N``-vector to an ``M``-vector.

    Args:
        kernel_fn: Kernel function ``k(x, z) -> scalar``.
        X_data: Data points, shape ``(N, D)``.
        X_inducing: Inducing points, shape ``(M, D)``.
        batch_size: Number of rows of ``X_data`` processed per scan step.
    """

    kernel_fn: Callable = eqx.field(static=True)
    X_data: Float[Array, "N D"]
    X_inducing: Float[Array, "M D"]
    batch_size: int = eqx.field(static=True)
    _n: int = eqx.field(static=True)
    _m: int = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        kernel_fn: Callable[[Float[Array, " D"], Float[Array, " D"]], Float[Array, ""]],
        X_data: Float[Array, "N D"],
        X_inducing: Float[Array, "M D"],
        batch_size: int = 1024,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        self.kernel_fn = kernel_fn
        self.X_data = X_data
        self.X_inducing = X_inducing
        self.batch_size = batch_size
        self._n = X_data.shape[0]
        self._m = X_inducing.shape[0]
        self.tags = _to_frozenset(tags)

    def mv(self, vector: Float[Array, " M"]) -> Float[Array, " N"]:
        """Compute ``K(X_data, X_inducing) @ v`` via batched scan.

        Peak memory per step is ``O(batch_size * M)``.
        """
        n = self._n
        bs = self.batch_size
        # Pad X_data to a multiple of batch_size
        n_padded = ((n + bs - 1) // bs) * bs
        pad_amount = n_padded - n
        X_padded = jnp.pad(self.X_data, ((0, pad_amount), (0, 0)), mode="constant")
        # Reshape into (num_batches, batch_size, D)
        num_batches = n_padded // bs
        X_batched = X_padded.reshape(num_batches, bs, -1)

        def batch_matvec(
            carry: None, X_batch: Float[Array, "batch_size D"]
        ) -> tuple[None, Float[Array, " batch_size"]]:
            # K_batch[i, j] = k(X_batch[i], X_inducing[j])
            K_batch = jax.vmap(
                lambda x_i: jax.vmap(lambda z_j: self.kernel_fn(x_i, z_j))(
                    self.X_inducing
                )
            )(X_batch)
            return carry, K_batch @ vector

        _, results = jax.lax.scan(batch_matvec, None, X_batched)
        # Flatten and trim padding
        return results.reshape(-1)[:n]

    def transpose(self) -> ImplicitCrossKernelOperator:
        """Return ``K(X_inducing, X_data)`` — the adjoint operator."""
        return ImplicitCrossKernelOperator(
            lambda x_i, x_j: self.kernel_fn(x_j, x_i),
            self.X_inducing,
            self.X_data,
            batch_size=self.batch_size,
            tags=lx.transpose_tags(self.tags),
        )

    def as_matrix(self) -> Float[Array, "N M"]:
        """Materialize the full ``N x M`` cross-kernel matrix.

        Uses batched vmap for consistency with the scan-based matvec.
        """
        return jax.vmap(
            lambda x_i: jax.vmap(lambda z_j: self.kernel_fn(x_i, z_j))(self.X_inducing)
        )(self.X_data)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._m,), self.X_data.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._n,), self.X_data.dtype)


def implicit_cross_kernel(
    kernel_fn: Callable[[Float[Array, " D"], Float[Array, " D"]], Float[Array, ""]],
    X_data: Float[Array, "N D"],
    X_inducing: Float[Array, "M D"],
    batch_size: int = 1024,
    *,
    tags: object | frozenset[object] = frozenset(),
) -> ImplicitCrossKernelOperator:
    """Create a matrix-free rectangular cross-kernel operator.

    Convenience wrapper around :class:`ImplicitCrossKernelOperator`.

    Args:
        kernel_fn: Kernel function ``k(x, z) -> scalar``.
        X_data: Data points, shape ``(N, D)``.
        X_inducing: Inducing points, shape ``(M, D)``.
        batch_size: Rows of ``X_data`` processed per scan step.
        tags: Lineax structural tags.

    Returns:
        An ``ImplicitCrossKernelOperator`` representing ``K(X_data, X_inducing)``.
    """
    return ImplicitCrossKernelOperator(
        kernel_fn, X_data, X_inducing, batch_size=batch_size, tags=tags
    )
