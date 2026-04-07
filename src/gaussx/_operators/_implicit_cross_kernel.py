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
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size}."
            )
        self.kernel_fn = kernel_fn
        self.X_data = X_data
        self.X_inducing = X_inducing
        self.batch_size = batch_size
        self._n = X_data.shape[0]
        self._m = X_inducing.shape[0]
        normalized_tags = _to_frozenset(tags)
        if lx.positive_semidefinite_tag in normalized_tags:
            if self._n != self._m:
                raise ValueError(
                    "positive_semidefinite_tag is only valid for square operators."
                )
            normalized_tags = normalized_tags | {lx.symmetric_tag}
        self.tags = normalized_tags

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

    def transpose(self) -> _TransposedCrossKernelOperator:
        """Return the adjoint operator ``K^T``.

        Uses a dedicated adjoint matvec that scans over batches of
        ``X_data`` and accumulates ``K_batch^T @ u_batch`` into an
        ``(M,)`` result, keeping peak memory at ``O(batch_size x M)``.
        """
        if lx.symmetric_tag in self.tags:
            return self  # type: ignore[return-value]
        return _TransposedCrossKernelOperator(self, tags=lx.transpose_tags(self.tags))

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


class _TransposedCrossKernelOperator(lx.AbstractLinearOperator):
    """Adjoint of :class:`ImplicitCrossKernelOperator`.

    Computes ``K^T u`` by scanning over batches of ``X_data`` and
    accumulating ``K_batch^T @ u_batch`` into an ``(M,)`` vector.
    Peak memory per step: ``O(batch_size x M)``.
    """

    _parent: ImplicitCrossKernelOperator
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        parent: ImplicitCrossKernelOperator,
        *,
        tags: frozenset[object],
    ) -> None:
        self._parent = parent
        self.tags = tags

    def mv(self, vector: Float[Array, " N"]) -> Float[Array, " M"]:
        r"""Compute ``K^T @ u`` via batched scan.

        Scans over batches of ``X_data``, building each
        ``(batch_size, M)`` kernel block and accumulating
        ``K_batch^T @ u_batch`` into an ``(M,)`` result.
        Peak memory per step: ``O(batch_size \times M)``.
        """
        parent = self._parent
        n = parent._n
        bs = parent.batch_size
        n_padded = ((n + bs - 1) // bs) * bs
        pad_amount = n_padded - n

        X_padded = jnp.pad(parent.X_data, ((0, pad_amount), (0, 0)), mode="constant")
        u_padded = jnp.pad(vector, (0, pad_amount), mode="constant")
        num_batches = n_padded // bs
        X_batched = X_padded.reshape(num_batches, bs, -1)
        u_batched = u_padded.reshape(num_batches, bs)

        def batch_adjoint(
            acc: Float[Array, " M"],
            xu: tuple[Float[Array, "batch_size D"], Float[Array, " batch_size"]],
        ) -> tuple[Float[Array, " M"], None]:
            X_batch, u_batch = xu
            # K_batch shape: (batch_size, M)
            K_batch = jax.vmap(
                lambda x_i: jax.vmap(lambda z_j: parent.kernel_fn(x_i, z_j))(
                    parent.X_inducing
                )
            )(X_batch)
            # K_batch^T @ u_batch -> (M,)
            acc = acc + K_batch.T @ u_batch
            return acc, None

        result, _ = jax.lax.scan(
            batch_adjoint,
            jnp.zeros(parent._m, dtype=parent.X_data.dtype),
            (X_batched, u_batched),
        )
        return result

    def as_matrix(self) -> Float[Array, "M N"]:
        """Materialize ``K^T`` as a dense matrix."""
        return self._parent.as_matrix().T

    def transpose(self) -> ImplicitCrossKernelOperator:
        """Return the original (non-transposed) operator."""
        return self._parent

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._parent._n,), self._parent.X_data.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._parent._m,), self._parent.X_data.dtype)


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
