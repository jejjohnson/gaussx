"""Symmetric Toeplitz operator with FFT-based matvec."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset


class Toeplitz(lx.AbstractLinearOperator):
    r"""Symmetric Toeplitz matrix from its first column.

    ``K_{ij} = c_{|i-j|}``.  Stored as O(n) with O(n log n) matvec
    via circulant embedding and FFT.

    For stationary kernels on regular 1-D grids the full kernel matrix
    is Toeplitz, so this gives an asymptotic win over dense storage.

    Args:
        column: First column of the Toeplitz matrix, shape ``(n,)``.
    """

    column: Float[Array, " n"]
    _size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        column: Float[Array, " n"],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        self.column = jnp.asarray(column)
        self._size = self.column.shape[0]
        self._dtype = str(self.column.dtype)
        self.tags = _to_frozenset(tags) | {lx.symmetric_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        return _toeplitz_mv(self.column, vector)

    def as_matrix(self) -> Float[Array, "n n"]:
        n = self._size
        indices = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
        return self.column[indices]

    def transpose(self) -> Toeplitz:
        # Symmetric Toeplitz is self-transpose.
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))


def _toeplitz_mv(
    column: Float[Array, " n"],
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    r"""Toeplitz matvec via circulant embedding + FFT.

    Embeds the n x n Toeplitz matrix into a 2n circulant matrix, then
    uses the convolution theorem:  ``y = IFFT(FFT(c_embed) * FFT(v_embed))[:n]``.
    """
    n = column.shape[0]
    # For n <= 2 the circulant embedding degenerates; use dense fallback.
    if n <= 2:
        indices = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
        return column[indices] @ vector
    # Circulant embedding: [c_0, c_1, ..., c_{n-1}, c_{n-1}, ..., c_1]
    m = 2 * n - 2
    c_embed = jnp.concatenate([column, column[-2:0:-1]])
    # Zero-pad the vector
    v_embed = jnp.concatenate([vector, jnp.zeros(n - 2, dtype=vector.dtype)])
    # FFT-based circular convolution
    result = jnp.fft.irfft(jnp.fft.rfft(c_embed) * jnp.fft.rfft(v_embed), n=m)
    return result[:n].real
