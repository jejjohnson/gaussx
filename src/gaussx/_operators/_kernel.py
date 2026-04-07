"""Kernel matrix operator with efficient hyperparameter gradients via custom VJP."""

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
# Custom-VJP matvec
# ---------------------------------------------------------------------------


def _make_kernel_mv(kernel_fn: Callable) -> Callable:
    """Build a custom-VJP matvec function closed over *kernel_fn* (static).

    We close over ``kernel_fn`` so that ``jax.custom_vjp`` never tries to
    differentiate or trace through it as a positional argument.
    """

    @jax.custom_vjp
    def kernel_mv(
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

    # -- forward ---------------------------------------------------------
    def kernel_mv_fwd(
        params: PyTree,
        X1: Float[Array, "N D"],
        X2: Float[Array, "M D"],
        v: Float[Array, " M"],
    ) -> tuple[
        Float[Array, " N"],
        tuple[
            PyTree,
            Float[Array, "N D"],
            Float[Array, "M D"],
            Float[Array, " M"],
        ],
    ]:
        result = kernel_mv(params, X1, X2, v)
        return result, (params, X1, X2, v)

    # -- backward --------------------------------------------------------
    def kernel_mv_bwd(
        res: tuple[
            PyTree,
            Float[Array, "N D"],
            Float[Array, "M D"],
            Float[Array, " M"],
        ],
        g: Float[Array, " N"],
    ) -> tuple[
        PyTree,
        Float[Array, "N D"],
        Float[Array, "M D"],
        Float[Array, " M"],
    ]:
        params_res, X1, X2, v = res
        u = g  # cotangent vector, shape (N,)

        # ∂L/∂v = K^T @ u  (scan-based, same structure)
        def row_dot_t(x_j: Float[Array, " D"]) -> Float[Array, ""]:
            k_col = jax.vmap(lambda x_i: kernel_fn(params_res, x_i, x_j))(X1)
            return jnp.dot(k_col, u)

        def body_fn_t(
            carry: None, x_j: Float[Array, " D"]
        ) -> tuple[None, Float[Array, ""]]:
            return carry, row_dot_t(x_j)

        _, grad_v = jax.lax.scan(body_fn_t, None, X2)

        # ∂L/∂params, ∂L/∂X1, ∂L/∂X2 via bilinear trick.
        # We scan over rows (i) and vmap over columns (j) so that each
        # iteration only holds one row of gradients in memory.
        #   ∂L/∂θ   = Σ_ij u_i v_j (∂k/∂θ)(params, x1_i, x2_j)
        #   ∂L/∂x1_i = u_i Σ_j v_j (∂k/∂x1)(params, x1_i, x2_j)
        #   ∂L/∂x2  += u_i Σ ... (accumulated across rows)
        dk_all = jax.grad(kernel_fn, argnums=(0, 1, 2))

        # Initialize accumulators
        zero_params = jax.tree.map(jnp.zeros_like, params_res)
        zero_X2 = jnp.zeros_like(X2)

        # Per-pair gradient: dk/d(params, x1, x2)
        def _pair_grad(
            x1_i: Float[Array, " D"], x2_j: Float[Array, " D"]
        ) -> tuple[PyTree, Float[Array, " D"], Float[Array, " D"]]:
            return dk_all(params_res, x1_i, x2_j)

        # vmap over j for a single x1_i
        _row_grads = jax.vmap(_pair_grad, in_axes=(None, 0))

        def _weighted_row(
            carry: tuple[PyTree, Float[Array, "M D"]],
            x1_i_ui: tuple[Float[Array, " D"], Float[Array, ""]],
        ) -> tuple[tuple[PyTree, Float[Array, "M D"]], Float[Array, " D"]]:
            acc_params, acc_X2 = carry
            x1_i, ui = x1_i_ui
            # grads: (params_g, x1_g, x2_g) with leaves (M, ...)
            g_params, g_x1, g_x2 = _row_grads(x1_i, X2)
            # Weight by u_i * v_j and sum over j (M axis)
            w_params = jax.tree.map(
                lambda rg: ui * jnp.tensordot(v, rg, axes=([0], [0])),
                g_params,
            )
            acc_params = jax.tree.map(lambda c, w: c + w, acc_params, w_params)
            # ∂L/∂x1_i = u_i * Σ_j v_j * dk/dx1(x1_i, x2_j)
            # g_x1 has shape (M, D), weight by v and sum over M
            grad_x1_i = ui * jnp.einsum("j,jd->d", v, g_x1)
            # ∂L/∂x2_j += u_i * v_j * dk/dx2(x1_i, x2_j)
            acc_X2 = acc_X2 + ui * (v[:, None] * g_x2)
            return (acc_params, acc_X2), grad_x1_i

        (grad_params, grad_X2), grad_X1 = jax.lax.scan(
            _weighted_row, (zero_params, zero_X2), (X1, u)
        )

        return grad_params, grad_X1, grad_X2, grad_v

    kernel_mv.defvjp(kernel_mv_fwd, kernel_mv_bwd)
    return kernel_mv


# ---------------------------------------------------------------------------
# Operator class
# ---------------------------------------------------------------------------


class KernelOperator(lx.AbstractLinearOperator):
    r"""Kernel matrix operator with efficient hyperparameter gradients.

    Represents the matrix ``K`` where ``K[i, j] = kernel_fn(params, X1[i], X2[j])``.
    The matvec ``K @ v`` is computed via scan (O(N) memory), and a
    ``jax.custom_vjp`` ensures that ``∂(uᵀ K v)/∂θ`` is obtained through the
    bilinear derivative trick — without ever materializing ``∂K/∂θ``.

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
            normalized_tags = normalized_tags | {lx.symmetric_tag}
        self.tags = normalized_tags
        self._kernel_mv = _make_kernel_mv(kernel_fn)

    def mv(self, vector: Float[Array, " M"]) -> Float[Array, " N"]:
        """Compute ``K @ v`` via scan with custom VJP for param gradients."""
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
