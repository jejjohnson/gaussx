"""Implicit kernel linear operator — matrix-free kernel matvec."""

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
# Custom-JVP matvec for params-aware ImplicitKernelOperator
# ---------------------------------------------------------------------------


def _make_implicit_kernel_mv(kernel_fn: Callable) -> Callable:
    """Build a custom-JVP matvec closed over *kernel_fn* (static).

    Mirrors the pattern from ``_kernel.py``.  The scan computes one
    kernel-row at a time so peak memory is ``O(N)`` rather than ``O(N^2)``.
    """

    def _impl(
        params: PyTree,
        X: Float[Array, "N D"],
        v: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        def row_dot(x_i: Float[Array, " D"]) -> Float[Array, ""]:
            k_row = jax.vmap(lambda x_j: kernel_fn(params, x_i, x_j))(X)
            return jnp.dot(k_row, v)

        def body_fn(
            carry: None, x_i: Float[Array, " D"]
        ) -> tuple[None, Float[Array, ""]]:
            return carry, row_dot(x_i)

        _, Kv = jax.lax.scan(body_fn, None, X)
        return Kv

    @jax.custom_jvp
    def implicit_kernel_mv(
        params: PyTree,
        X: Float[Array, "N D"],
        v: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        return _impl(params, X, v)

    @implicit_kernel_mv.defjvp
    def implicit_kernel_mv_jvp(
        primals: tuple[PyTree, Float[Array, "N D"], Float[Array, " N"]],
        tangents: tuple[PyTree, Float[Array, "N D"], Float[Array, " N"]],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        params, X, v = primals
        dparams, dX, dv = tangents

        def row_jvp(
            x_and_tangent: tuple[Float[Array, " D"], Float[Array, " D"]],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            x_i, dx_i = x_and_tangent

            def kernel_with_jvp(
                x_and_t: tuple[Float[Array, " D"], Float[Array, " D"]],
            ) -> tuple[Float[Array, ""], Float[Array, ""]]:
                x_j, dx_j = x_and_t
                return jax.jvp(
                    kernel_fn,
                    (params, x_i, x_j),
                    (dparams, dx_i, dx_j),
                )

            k_row, dk_row = jax.vmap(kernel_with_jvp)((X, dX))
            primal = jnp.dot(k_row, v)
            tangent = jnp.dot(dk_row, v) + jnp.dot(k_row, dv)
            return primal, tangent

        primal_out, tangent_out = jax.vmap(row_jvp)((X, dX))
        return primal_out, tangent_out

    return implicit_kernel_mv


class ImplicitKernelOperator(lx.AbstractLinearOperator):
    r"""Matrix-free kernel operator: ``(K + sigma^2 I) v`` via sequential scan.

    Computes the kernel matvec without materializing the ``N x N`` kernel
    matrix, using ``O(N)`` memory instead of ``O(N^2)``.  Each element of
    the output is computed as::

        y_i = \sum_j k(x_i, x_j) v_j + sigma^2 v_i

    The scan-based implementation is compatible with CG / BBMM solvers
    that only need matvec access.

    Supports two kernel signatures:

    - **No params** (default): ``k(x, x') -> scalar``.  Hyperparameters
      are closed over in the lambda.
    - **With params**: ``k(params, x, x') -> scalar``.  Pass a pytree of
      differentiable hyperparameters via the ``params`` argument and a
      ``jax.custom_jvp`` keeps first-order autodiff efficient.

    Args:
        kernel_fn: Kernel function (see above for signature).
        X: Training points, shape ``(N, D)``.
        noise_var: Diagonal noise variance ``sigma^2``.
        params: Optional pytree of kernel hyperparameters.
    """

    kernel_fn: Callable = eqx.field(static=True)
    X: Float[Array, "N D"]
    noise_var: float = eqx.field(static=True)
    params: Any
    _size: int = eqx.field(static=True)
    _has_params: bool = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)
    _kernel_mv: Callable = eqx.field(static=True)

    def __init__(
        self,
        kernel_fn: Callable,
        X: Float[Array, "N D"],
        noise_var: float = 0.0,
        *,
        params: Any | None = None,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        self.kernel_fn = kernel_fn
        self.X = X
        self.noise_var = noise_var
        self._size = X.shape[0]
        self._has_params = params is not None
        self.params = params
        normalized_tags = _to_frozenset(tags)
        if lx.positive_semidefinite_tag in normalized_tags:
            normalized_tags = normalized_tags | {lx.symmetric_tag}
        self.tags = normalized_tags
        if self._has_params:
            self._kernel_mv = _make_implicit_kernel_mv(kernel_fn)
        else:
            self._kernel_mv = None  # type: ignore[assignment]

    def mv(self, vector: Float[Array, " N"]) -> Float[Array, " N"]:
        """Compute ``(K + sigma^2 I) @ v`` via scan over data points."""
        if self._has_params:
            Kv = self._kernel_mv(self.params, self.X, vector)
        else:

            def row_dot(x_i: Float[Array, " D"]) -> Float[Array, ""]:
                k_row = jax.vmap(lambda x_j: self.kernel_fn(x_i, x_j))(self.X)
                return jnp.dot(k_row, vector)

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
        if self._has_params:
            K = jax.vmap(
                lambda x_i: jax.vmap(lambda x_j: self.kernel_fn(self.params, x_i, x_j))(
                    self.X
                )
            )(self.X)
        else:
            K = jax.vmap(
                lambda x_i: jax.vmap(lambda x_j: self.kernel_fn(x_i, x_j))(self.X)
            )(self.X)
        if self.noise_var != 0.0:
            K = K + self.noise_var * jnp.eye(self._size)
        return K

    def transpose(self) -> ImplicitKernelOperator:
        if lx.symmetric_tag in self.tags:
            return self
        if self._has_params:
            return ImplicitKernelOperator(
                lambda p, x_i, x_j: self.kernel_fn(p, x_j, x_i),
                self.X,
                noise_var=self.noise_var,
                params=self.params,
                tags=lx.transpose_tags(self.tags),
            )
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
