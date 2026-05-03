"""Implicit cross-kernel linear operator — matrix-free rectangular kernel matvec."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from einops import rearrange
from jaxtyping import Array, Float, PyTree

from gaussx._operators._block_diag import _to_frozenset
from gaussx._operators._utils import vmap_over_batch_dims


# ---------------------------------------------------------------------------
# Custom-JVP matvec for params-aware cross-kernel
# ---------------------------------------------------------------------------


def _make_cross_kernel_mv(kernel_fn: Callable, batch_size: int) -> Callable:
    """Build a custom-JVP cross-kernel matvec closed over *kernel_fn*.

    Forward:  ``K(X_data, X_inducing; params) @ v``  via scan.
    JVP: efficient first-order autodiff via vmap (transposes cleanly for VJP).
    """

    def _impl(
        params: PyTree,
        X_data: Float[Array, "N D"],
        X_inducing: Float[Array, "M D"],
        v: Float[Array, " M"],
    ) -> Float[Array, " N"]:
        n = X_data.shape[0]
        bs = batch_size
        n_padded = ((n + bs - 1) // bs) * bs
        pad_amount = n_padded - n
        X_padded = jnp.pad(X_data, ((0, pad_amount), (0, 0)), mode="constant")
        X_batched = rearrange(X_padded, "(B bs) D -> B bs D", bs=bs)

        def batch_jvp_rows(
            X_batch: Float[Array, "batch_size D"],
        ) -> Float[Array, " batch_size"]:
            K_batch = jax.vmap(
                lambda x_i: jax.vmap(lambda z_j: kernel_fn(params, x_i, z_j))(
                    X_inducing
                )
            )(X_batch)
            return K_batch @ v

        _, Kv = jax.lax.scan(
            lambda carry, X_batch: (carry, batch_jvp_rows(X_batch)),
            None,
            X_batched,
        )
        return rearrange(Kv, "B bs -> (B bs)")[:n]

    @jax.custom_jvp
    def cross_kernel_mv(
        params: PyTree,
        X_data: Float[Array, "N D"],
        X_inducing: Float[Array, "M D"],
        v: Float[Array, " M"],
    ) -> Float[Array, " N"]:
        return _impl(params, X_data, X_inducing, v)

    @cross_kernel_mv.defjvp
    def cross_kernel_mv_jvp(
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
        params_p, X_data, X_inducing, v = primals
        dparams, dX_data, dX_inducing, dv = tangents

        n = X_data.shape[0]
        bs = batch_size
        n_padded = ((n + bs - 1) // bs) * bs
        pad_amount = n_padded - n
        X_padded = jnp.pad(X_data, ((0, pad_amount), (0, 0)), mode="constant")
        dX_padded = jnp.pad(dX_data, ((0, pad_amount), (0, 0)), mode="constant")
        X_batched = rearrange(X_padded, "(B bs) D -> B bs D", bs=bs)
        dX_batched = rearrange(dX_padded, "(B bs) D -> B bs D", bs=bs)

        def row_jvp(
            xdx: tuple[Float[Array, " D"], Float[Array, " D"]],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            x_i, dx_i = xdx

            def col_jvp(
                zdz: tuple[Float[Array, " D"], Float[Array, " D"]],
            ) -> tuple[Float[Array, ""], Float[Array, ""]]:
                z_j, dz_j = zdz
                return jax.jvp(
                    kernel_fn,
                    (params_p, x_i, z_j),
                    (dparams, dx_i, dz_j),
                )

            k_row, dk_row = jax.vmap(col_jvp)((X_inducing, dX_inducing))
            primal = jnp.dot(k_row, v)
            tangent = jnp.dot(dk_row, v) + jnp.dot(k_row, dv)
            return primal, tangent

        def batch_jvp(
            xdx_batch: tuple[
                Float[Array, "batch_size D"],
                Float[Array, "batch_size D"],
            ],
        ) -> tuple[Float[Array, " batch_size"], Float[Array, " batch_size"]]:
            return jax.vmap(row_jvp)(xdx_batch)

        outputs = jax.vmap(batch_jvp)((X_batched, dX_batched))
        primal_out, tangent_out = outputs
        return rearrange(primal_out, "B bs -> (B bs)")[:n], rearrange(
            tangent_out, "B bs -> (B bs)"
        )[:n]

    return cross_kernel_mv


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

    Supports two kernel signatures:

    - **No params** (default): ``k(x, z) -> scalar``.
    - **With params**: ``k(params, x, z) -> scalar``.  Pass a pytree of
      differentiable hyperparameters and a ``jax.custom_jvp`` keeps
      first-order autodiff efficient.

    Batched inputs are supported: ``X_data`` and ``X_inducing`` may
    carry leading batch dimensions ``(*batch, N, D)`` /
    ``(*batch, M, D)`` (with matching ``*batch``). In that case ``mv``
    expects a vector of shape ``(*batch, M)`` and returns ``(*batch, N)``;
    the transposed operator follows the symmetric pattern. ``as_matrix()``
    returns a ``(*batch, N, M)`` tensor; ``in_structure()`` /
    ``out_structure()`` report the batched shapes.

    Args:
        kernel_fn: Kernel function (see above for signature).
        X_data: Data points, shape ``(*batch, N, D)``. Leading batch
            dimensions are optional.
        X_inducing: Inducing points, shape ``(*batch, M, D)`` with
            ``*batch`` matching ``X_data``.
        batch_size: Number of rows of ``X_data`` processed per scan step.
        params: Optional pytree of kernel hyperparameters.
    """

    kernel_fn: Callable = eqx.field(static=True)
    X_data: Float[Array, "*batch N D"]
    X_inducing: Float[Array, "*batch M D"]
    params: Any
    batch_size: int = eqx.field(static=True)
    _n: int = eqx.field(static=True)
    _m: int = eqx.field(static=True)
    _batch_shape: tuple[int, ...] = eqx.field(static=True)
    _has_params: bool = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)
    _kernel_mv: Callable = eqx.field(static=True)

    def __init__(
        self,
        kernel_fn: Callable,
        X_data: Float[Array, "N D"],
        X_inducing: Float[Array, "M D"],
        batch_size: int = 1024,
        *,
        params: Any | None = None,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size}."
            )
        if X_data.shape[:-2] != X_inducing.shape[:-2]:
            raise ValueError("X_data and X_inducing must have matching batch shapes.")
        self.kernel_fn = kernel_fn
        self.X_data = X_data
        self.X_inducing = X_inducing
        self.params = params
        self.batch_size = batch_size
        self._n = X_data.shape[-2]
        self._m = X_inducing.shape[-2]
        self._batch_shape = X_data.shape[:-2]
        self._has_params = params is not None
        normalized_tags = _to_frozenset(tags)
        if lx.positive_semidefinite_tag in normalized_tags:
            if self._n != self._m:
                raise ValueError(
                    "positive_semidefinite_tag is only valid for square operators."
                )
            normalized_tags = normalized_tags | {lx.symmetric_tag}
        self.tags = normalized_tags
        if self._has_params:
            self._kernel_mv = _make_cross_kernel_mv(kernel_fn, batch_size)
        else:
            self._kernel_mv = None  # type: ignore[assignment]

    def mv(self, vector: Float[Array, "*batch M"]) -> Float[Array, "*batch N"]:
        """Compute ``K(X_data, X_inducing) @ v`` via batched scan.

        Peak memory per step is ``O(batch_size * M)``.
        """
        if vector.shape[:-1] != self._batch_shape:
            raise ValueError(
                "vector must have leading batch dimensions matching X_data/X_inducing."
            )

        def mv_single(
            X_data: Float[Array, "N D"],
            X_inducing: Float[Array, "M D"],
            v: Float[Array, " M"],
        ) -> Float[Array, " N"]:
            if self._has_params:
                return self._kernel_mv(self.params, X_data, X_inducing, v)

            n = self._n
            bs = self.batch_size
            n_padded = ((n + bs - 1) // bs) * bs
            pad_amount = n_padded - n
            X_padded = jnp.pad(X_data, ((0, pad_amount), (0, 0)), mode="constant")
            X_batched = rearrange(X_padded, "(B bs) D -> B bs D", bs=bs)

            def batch_matvec(
                carry: None, X_batch: Float[Array, "batch_size D"]
            ) -> tuple[None, Float[Array, " batch_size"]]:
                K_batch = jax.vmap(
                    lambda x_i: jax.vmap(lambda z_j: self.kernel_fn(x_i, z_j))(
                        X_inducing
                    )
                )(X_batch)
                return carry, K_batch @ v

            _, results = jax.lax.scan(batch_matvec, None, X_batched)
            return rearrange(results, "B bs -> (B bs)")[:n]

        if not self._batch_shape:
            return mv_single(self.X_data, self.X_inducing, vector)
        batched_mv = vmap_over_batch_dims(mv_single, len(self._batch_shape))
        return batched_mv(self.X_data, self.X_inducing, vector)

    def transpose(self) -> _TransposedCrossKernelOperator:
        """Return the adjoint operator ``K^T``.

        Uses a dedicated adjoint matvec that scans over batches of
        ``X_data`` and accumulates ``K_batch^T @ u_batch`` into an
        ``(M,)`` result, keeping peak memory at ``O(batch_size x M)``.
        """
        if lx.symmetric_tag in self.tags:
            return self  # type: ignore[return-value]
        return _TransposedCrossKernelOperator(self, tags=lx.transpose_tags(self.tags))

    def as_matrix(self) -> Float[Array, "*batch N M"]:
        """Materialize the full ``N x M`` cross-kernel matrix."""
        if self._has_params:
            matrix_fn = lambda X_data, X_inducing: jax.vmap(
                lambda x_i: jax.vmap(lambda z_j: self.kernel_fn(self.params, x_i, z_j))(
                    X_inducing
                )
            )(X_data)
        else:
            matrix_fn = lambda X_data, X_inducing: jax.vmap(
                lambda x_i: jax.vmap(lambda z_j: self.kernel_fn(x_i, z_j))(X_inducing)
            )(X_data)
        if not self._batch_shape:
            return matrix_fn(self.X_data, self.X_inducing)
        batched_matrix_fn = vmap_over_batch_dims(matrix_fn, len(self._batch_shape))
        return batched_matrix_fn(self.X_data, self.X_inducing)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((*self._batch_shape, self._m), self.X_data.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((*self._batch_shape, self._n), self.X_data.dtype)


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

    def mv(self, vector: Float[Array, "*batch N"]) -> Float[Array, "*batch M"]:
        r"""Compute ``K^T @ u`` via batched scan.

        Scans over batches of ``X_data``, building each
        ``(batch_size, M)`` kernel block and accumulating
        ``K_batch^T @ u_batch`` into an ``(M,)`` result.
        Peak memory per step: ``O(batch_size \times M)``.
        """
        parent = self._parent
        if vector.shape[:-1] != parent._batch_shape:
            raise ValueError(
                "vector must have leading batch dimensions matching the parent "
                "operator."
            )

        def mv_single(
            X_data: Float[Array, "N D"],
            X_inducing: Float[Array, "M D"],
            u: Float[Array, " N"],
        ) -> Float[Array, " M"]:
            n = parent._n
            bs = parent.batch_size
            n_padded = ((n + bs - 1) // bs) * bs
            pad_amount = n_padded - n

            X_padded = jnp.pad(X_data, ((0, pad_amount), (0, 0)), mode="constant")
            u_padded = jnp.pad(u, (0, pad_amount), mode="constant")
            X_batched = rearrange(X_padded, "(B bs) D -> B bs D", bs=bs)
            u_batched = rearrange(u_padded, "(B bs) -> B bs", bs=bs)

            if parent._has_params:
                kfn = parent.kernel_fn
                params = parent.params

                def batch_adjoint_params(
                    acc: Float[Array, " M"],
                    xu: tuple[
                        Float[Array, "batch_size D"],
                        Float[Array, " batch_size"],
                    ],
                ) -> tuple[Float[Array, " M"], None]:
                    X_batch, u_batch = xu
                    K_batch = jax.vmap(
                        lambda x_i: jax.vmap(lambda z_j: kfn(params, x_i, z_j))(
                            X_inducing
                        )
                    )(X_batch)
                    acc = acc + K_batch.T @ u_batch
                    return acc, None

                result, _ = jax.lax.scan(
                    batch_adjoint_params,
                    jnp.zeros(parent._m, dtype=X_data.dtype),
                    (X_batched, u_batched),
                )
            else:

                def batch_adjoint(
                    acc: Float[Array, " M"],
                    xu: tuple[
                        Float[Array, "batch_size D"],
                        Float[Array, " batch_size"],
                    ],
                ) -> tuple[Float[Array, " M"], None]:
                    X_batch, u_batch = xu
                    K_batch = jax.vmap(
                        lambda x_i: jax.vmap(lambda z_j: parent.kernel_fn(x_i, z_j))(
                            X_inducing
                        )
                    )(X_batch)
                    acc = acc + K_batch.T @ u_batch
                    return acc, None

                result, _ = jax.lax.scan(
                    batch_adjoint,
                    jnp.zeros(parent._m, dtype=X_data.dtype),
                    (X_batched, u_batched),
                )
            return result

        if not parent._batch_shape:
            return mv_single(parent.X_data, parent.X_inducing, vector)
        batched_mv = vmap_over_batch_dims(mv_single, len(parent._batch_shape))
        return batched_mv(parent.X_data, parent.X_inducing, vector)

    def as_matrix(self) -> Float[Array, "*batch M N"]:
        """Materialize ``K^T`` as a dense matrix."""
        return jnp.swapaxes(self._parent.as_matrix(), -1, -2)

    def transpose(self) -> ImplicitCrossKernelOperator:
        """Return the original (non-transposed) operator."""
        return self._parent

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (*self._parent._batch_shape, self._parent._n),
            self._parent.X_data.dtype,
        )

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (*self._parent._batch_shape, self._parent._m),
            self._parent.X_data.dtype,
        )


def implicit_cross_kernel(
    kernel_fn: Callable,
    X_data: Float[Array, "N D"],
    X_inducing: Float[Array, "M D"],
    batch_size: int = 1024,
    *,
    params: Any | None = None,
    tags: object | frozenset[object] = frozenset(),
) -> ImplicitCrossKernelOperator:
    """Create a matrix-free rectangular cross-kernel operator.

    Convenience wrapper around :class:`ImplicitCrossKernelOperator`.

    Args:
        kernel_fn: Kernel function ``k(x, z) -> scalar`` or
            ``k(params, x, z) -> scalar``.
        X_data: Data points, shape ``(N, D)``.
        X_inducing: Inducing points, shape ``(M, D)``.
        batch_size: Rows of ``X_data`` processed per scan step.
        params: Optional pytree of kernel hyperparameters.
        tags: Lineax structural tags.

    Returns:
        An ``ImplicitCrossKernelOperator`` representing ``K(X_data, X_inducing)``.
    """
    return ImplicitCrossKernelOperator(
        kernel_fn, X_data, X_inducing, batch_size=batch_size, params=params, tags=tags
    )
