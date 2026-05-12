"""Symmetric Toeplitz operator with FFT-based matvec."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
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


class ToeplitzCholesky(lx.AbstractLinearOperator):
    """Circulant-embedding sample factor for a symmetric positive Toeplitz matrix.

    The operator has shape ``(n, embedding_factor * n)`` and satisfies
    ``L @ L.T == Toeplitz(column)`` — it is a rectangular sample factor,
    *not* a traditional lower-triangular Cholesky factor. Applying it to
    standard normal white noise gives samples from
    ``𝒩(0, Toeplitz(column))`` when the Wood--Chan condition holds. The
    Wood--Chan non-negativity check is performed eagerly at construction
    time (so the constructor is not JIT-friendly; use :func:`toeplitz_sample`
    if a runtime fallback is desired).

    Args:
        column: First column of the Toeplitz matrix, shape ``(n,)``.
        embedding_factor: Circulant embedding size as a multiple of ``n``.
    """

    sqrt_spectrum: Float[Array, " m"]
    _size: int = eqx.field(static=True)
    _embedding_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        column: Float[Array, " n"],
        *,
        embedding_factor: int = 2,
    ) -> None:
        column = _as_floating_column(column)
        self.sqrt_spectrum = _circulant_sqrt_spectrum(
            column,
            embedding_factor=embedding_factor,
        )
        self._size = column.shape[0]
        self._embedding_size = embedding_factor * self._size
        self._dtype = str(column.dtype)
        self.tags = frozenset()

    def mv(self, vector: Float[Array, " m"]) -> Float[Array, " n"]:
        return _circulant_sqrt_mv(self.sqrt_spectrum, vector, self._size)

    def as_matrix(self) -> Float[Array, "n m"]:
        eye = jnp.eye(self._embedding_size, dtype=jnp.dtype(self._dtype))
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(eye)

    def transpose(self) -> _ToeplitzCholeskyTranspose:
        return _ToeplitzCholeskyTranspose(self)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._embedding_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))


class _ToeplitzCholeskyTranspose(lx.AbstractLinearOperator):
    factor: ToeplitzCholesky
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(self, factor: ToeplitzCholesky) -> None:
        self.factor = factor
        self.tags = frozenset()

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        padding = jnp.zeros(
            self.factor._embedding_size - self.factor._size,
            dtype=vector.dtype,
        )
        padded = jnp.concatenate([vector, padding])
        return _circulant_sqrt_mv(
            self.factor.sqrt_spectrum,
            padded,
            self.factor._embedding_size,
        )

    def as_matrix(self) -> Float[Array, "m n"]:
        eye = jnp.eye(self.factor._size, dtype=jnp.dtype(self.factor._dtype))
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(eye)

    def transpose(self) -> ToeplitzCholesky:
        return self.factor

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.factor.out_structure()

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.factor.in_structure()


for _check in (
    lx.is_symmetric,
    lx.is_diagonal,
    lx.is_positive_semidefinite,
    lx.is_negative_semidefinite,
    lx.is_tridiagonal,
    lx.is_lower_triangular,
    lx.is_upper_triangular,
):

    @_check.register(ToeplitzCholesky)
    @_check.register(_ToeplitzCholeskyTranspose)
    def _(_operator, check=_check):
        return False


def toeplitz_sample(
    column: Float[Array, " n"],
    *,
    key: jax.Array,
    num_samples: int = 1,
    embedding_factor: int = 2,
) -> Float[Array, "num_samples n"]:
    """Sample from ``𝒩(0, Toeplitz(column))`` via FFT circulant embedding.

    Args:
        column: First column of the covariance matrix.
        key: JAX PRNG key used to draw white noise.
        num_samples: Number of independent samples to draw.
        embedding_factor: Circulant embedding size as a multiple of ``n``.

    Returns:
        Samples with shape ``(num_samples, n)``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    column = _as_floating_column(column)
    try:
        factor = ToeplitzCholesky(column, embedding_factor=embedding_factor)
    except ValueError as error:
        warnings.warn(
            f"Wood-Chan embedding failed: {error} "
            "Falling back to dense Cholesky sampling.",
            RuntimeWarning,
            stacklevel=2,
        )
        L = jnp.linalg.cholesky(Toeplitz(column).as_matrix())
        noise = jr.normal(key, (num_samples, column.shape[0]), dtype=column.dtype)
        return noise @ L.T

    noise = jr.normal(key, (num_samples, factor.in_size()), dtype=column.dtype)
    return jax.vmap(factor.mv)(noise)


def _as_floating_column(column: Float[Array, " n"]) -> Float[Array, " n"]:
    column = jnp.asarray(column)
    if column.ndim != 1:
        raise ValueError(f"Toeplitz column must be rank 1, got shape {column.shape}.")
    if column.shape[0] == 0:
        raise ValueError("Toeplitz column must be non-empty.")
    dtype = jnp.result_type(column.dtype, jnp.float32)
    return column.astype(dtype)


def _circulant_embedding(
    column: Float[Array, " n"],
    *,
    embedding_factor: int,
) -> Float[Array, " m"]:
    if embedding_factor < 2:
        raise ValueError("embedding_factor must be at least 2.")
    n = column.shape[0]
    m = embedding_factor * n
    zeros = jnp.zeros(m - (2 * n - 1), dtype=column.dtype)
    return jnp.concatenate([column, zeros, column[:0:-1]])


def _circulant_sqrt_spectrum(
    column: Float[Array, " n"],
    *,
    embedding_factor: int,
) -> Float[Array, " m"]:
    spectrum = jnp.fft.rfft(
        _circulant_embedding(column, embedding_factor=embedding_factor),
    ).real
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(spectrum)))
    # Allow roundoff-sized negative FFT eigenvalues while rejecting real failures.
    tolerance = 100 * jnp.finfo(column.dtype).eps * scale
    if jnp.any(spectrum < -tolerance):
        raise ValueError(
            "Circulant embedding failed the Wood-Chan non-negativity condition "
            f"for embedding_factor={embedding_factor}; try "
            f"embedding_factor={2 * embedding_factor}."
        )
    return jnp.sqrt(jnp.clip(spectrum, min=0.0))


def _circulant_sqrt_mv(
    sqrt_spectrum: Float[Array, " m"],
    vector: Float[Array, " m"],
    output_size: int,
) -> Float[Array, " n"]:
    embedding_size = vector.shape[0]
    result = jnp.fft.irfft(
        sqrt_spectrum * jnp.fft.rfft(vector),
        n=embedding_size,
    )
    return result[:output_size]


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
