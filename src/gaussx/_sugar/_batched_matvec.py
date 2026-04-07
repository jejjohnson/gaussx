"""Batched kernel matvec utilities — memory-efficient K @ v and K^T @ u."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def batched_kernel_matvec(
    kernel_fn: Callable[[Float[Array, " D"], Float[Array, " D"]], Float[Array, ""]],
    X: Float[Array, "N D"],
    Z: Float[Array, "M D"],
    v: Float[Array, " M"],
    batch_size: int = 1024,
) -> Float[Array, " N"]:
    r"""Compute ``K(X, Z) @ v`` in memory-efficient batches.

    Each batch evaluates a ``(batch_size, M)`` kernel sub-matrix and
    immediately contracts with ``v``, keeping peak memory at
    ``O(batch_size * M)`` instead of ``O(N * M)``.

    Args:
        kernel_fn: Pairwise kernel function ``k(x, z) -> scalar``.
        X: First set of points, shape ``(N, D)``.
        Z: Second set of points, shape ``(M, D)``.
        v: Vector to multiply, shape ``(M,)``.
        batch_size: Rows of ``X`` processed per scan step.

    Returns:
        Result of ``K(X, Z) @ v``, shape ``(N,)``.
    """
    n = X.shape[0]
    bs = batch_size
    n_padded = ((n + bs - 1) // bs) * bs
    pad_amount = n_padded - n
    X_padded = jnp.pad(X, ((0, pad_amount), (0, 0)), mode="constant")
    num_batches = n_padded // bs
    X_batched = X_padded.reshape(num_batches, bs, -1)

    def scan_body(
        carry: None, X_batch: Float[Array, "batch_size D"]
    ) -> tuple[None, Float[Array, " batch_size"]]:
        K_batch = jax.vmap(lambda x_i: jax.vmap(lambda z_j: kernel_fn(x_i, z_j))(Z))(
            X_batch
        )
        return carry, K_batch @ v

    _, results = jax.lax.scan(scan_body, None, X_batched)
    return results.reshape(-1)[:n]


def batched_kernel_rmatvec(
    kernel_fn: Callable[[Float[Array, " D"], Float[Array, " D"]], Float[Array, ""]],
    X: Float[Array, "N D"],
    Z: Float[Array, "M D"],
    u: Float[Array, " N"],
    batch_size: int = 1024,
) -> Float[Array, " M"]:
    r"""Compute ``K(X, Z)^T @ u`` in memory-efficient batches.

    Scans over batches of ``X``, building each ``(batch_size, M)``
    kernel block and accumulating ``K_batch^T @ u_batch`` into an
    ``(M,)`` result.  Peak memory per step: ``O(batch_size * M)``.

    Args:
        kernel_fn: Pairwise kernel function ``k(x, z) -> scalar``.
        X: First set of points, shape ``(N, D)``.
        Z: Second set of points, shape ``(M, D)``.
        u: Vector to multiply, shape ``(N,)``.
        batch_size: Rows of ``X`` processed per scan step.

    Returns:
        Result of ``K(X, Z)^T @ u``, shape ``(M,)``.
    """
    n = X.shape[0]
    m = Z.shape[0]
    bs = batch_size
    n_padded = ((n + bs - 1) // bs) * bs
    pad_amount = n_padded - n
    X_padded = jnp.pad(X, ((0, pad_amount), (0, 0)), mode="constant")
    u_padded = jnp.pad(u, (0, pad_amount), mode="constant")
    num_batches = n_padded // bs
    X_batched = X_padded.reshape(num_batches, bs, -1)
    u_batched = u_padded.reshape(num_batches, bs)

    def scan_body(
        acc: Float[Array, " M"],
        xu: tuple[Float[Array, "batch_size D"], Float[Array, " batch_size"]],
    ) -> tuple[Float[Array, " M"], None]:
        X_batch, u_batch = xu
        K_batch = jax.vmap(lambda x_i: jax.vmap(lambda z_j: kernel_fn(x_i, z_j))(Z))(
            X_batch
        )
        acc = acc + K_batch.T @ u_batch
        return acc, None

    result, _ = jax.lax.scan(
        scan_body,
        jnp.zeros(m, dtype=X.dtype),
        (X_batched, u_batched),
    )
    return result
