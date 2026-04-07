"""Interpolated (SKI / KISS-GP) linear operator: K ~ W K_uu W^T."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset


class InterpolatedOperator(lx.AbstractLinearOperator):
    r"""Structured Kernel Interpolation: ``K \approx W K_{uu} W^T``.

    ``W`` is a sparse interpolation matrix with ``p`` nonzeros per row
    (e.g. cubic interpolation weights).  The base operator ``K_{uu}``
    acts on the inducing grid (typically Toeplitz for stationary
    kernels).

    Total matvec cost: ``O(n p + m log m)`` when the base is Toeplitz,
    essentially linear in ``n``.

    Args:
        base_operator: The inducing-point kernel ``K_{uu}``, shape ``(m, m)``.
        interp_indices: Integer indices into the inducing grid,
            shape ``(n, p)`` where ``p`` is the interpolation order.
        interp_values: Interpolation weights, shape ``(n, p)``.
    """

    base_operator: lx.AbstractLinearOperator
    interp_indices: Int[Array, "n p"]
    interp_values: Float[Array, "n p"]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _m: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        base_operator: lx.AbstractLinearOperator,
        interp_indices: Int[Array, "n p"],
        interp_values: Float[Array, "n p"],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if interp_indices.shape != interp_values.shape:
            raise ValueError(
                f"interp_indices and interp_values must have the same shape, "
                f"got {interp_indices.shape} and {interp_values.shape}."
            )
        m = base_operator.in_size()
        n = interp_indices.shape[0]
        self.base_operator = base_operator
        self.interp_indices = jnp.asarray(interp_indices)
        self.interp_values = jnp.asarray(interp_values)
        self._in_size = n
        self._out_size = n
        self._m = m
        self._dtype = _resolve_dtype(base_operator)
        self.tags = _to_frozenset(tags)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        # W^T v: scatter into inducing space
        wt_v = jnp.zeros(self._m, dtype=vector.dtype)
        wt_v = wt_v.at[self.interp_indices].add(self.interp_values * vector[:, None])
        # K_uu (W^T v)
        k_wt_v = self.base_operator.mv(wt_v)
        # W (K_uu W^T v): gather back
        return jnp.sum(self.interp_values * k_wt_v[self.interp_indices], axis=-1)

    def as_matrix(self) -> Float[Array, "n n"]:
        W = self._build_W()
        K_uu = self.base_operator.as_matrix()
        return W @ K_uu @ W.T

    def transpose(self) -> InterpolatedOperator:
        # W K_uu W^T is symmetric when K_uu is symmetric, so self-transpose.
        # In general, (W K_uu W^T)^T = W K_uu^T W^T.
        return InterpolatedOperator(
            self.base_operator.T,
            self.interp_indices,
            self.interp_values,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))

    def _build_W(self) -> Float[Array, "n m"]:
        """Build the dense interpolation matrix W (n x m)."""
        n = self._in_size
        W = jnp.zeros((n, self._m), dtype=jnp.dtype(self._dtype))
        rows = jnp.arange(n)[:, None]  # (n, 1) broadcast with (n, p)
        W = W.at[rows, self.interp_indices].add(self.interp_values)
        return W
