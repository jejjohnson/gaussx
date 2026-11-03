"""Symmetric Toeplitz operator A = toeplitz(t) with FFT matvec and sampling.

t ∈ ℝ^N is the first column, so [A]_ij = t_|i−j| with zero-based indices
i, j = 0, …, N−1. A circulant matrix C ∈ ℝ^{M × M} with first column c is
diagonalised by the discrete Fourier transform (DFT),
C x = DFT⁻¹(DFT(c) ⊙ DFT(x)), so embedding A in the top-left corner of a
circulant (circulant embedding) gives an O(M log M) matvec. When the DFT of
the embedding, γ = DFT(c), is non-negative (the Wood--Chan condition), the
symmetric root C^½ = DFT⁻¹ diag(√γ) DFT gives the sample factor
L = [C^½]_{0:N, :} with L Lᵀ = A.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset


class Toeplitz(lx.AbstractLinearOperator):
    r"""Symmetric Toeplitz matrix A = toeplitz(t) from its first column t.

        [A]_ij = t_|i−j|,   i, j = 0, …, N−1

    A is stored as t alone, O(N), and applied in O(N log N) by circulant
    embedding and the FFT. A stationary kernel on a regular 1-D grid has a
    Toeplitz kernel matrix, so this beats dense storage asymptotically.

    Args:
        column: First column t ∈ ℝ^N, real; integer columns are promoted to
            floating point.
        tags: Extra lineax tags for A; ``symmetric_tag`` is always added.

    Raises:
        TypeError: If ``column`` is complex.
        ValueError: If ``column`` is not rank 1 or is empty.

    Examples:
        A 3 × 3 Toeplitz matrix from its first column:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = gaussx.Toeplitz(jnp.array([2.0, 0.5, 0.25]))
        >>> A.as_matrix().tolist()
        [[2.0, 0.5, 0.25], [0.5, 2.0, 0.5], [0.25, 0.5, 2.0]]

        The FFT matvec matches the dense product:

        >>> x = jnp.ones(3)
        >>> bool(jnp.allclose(A.mv(x), jnp.array([2.75, 3.0, 2.75]), atol=1e-6))
        True

        Integer columns are promoted, and A is always symmetric:

        >>> B = gaussx.Toeplitz(jnp.array([3, 1]))
        >>> bool(jnp.issubdtype(B.column.dtype, jnp.floating))
        True
        >>> lx.is_symmetric(B)
        True
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
        self.column = _as_floating_column(column)
        # N
        self._size = self.column.shape[0]
        self._dtype = str(self.column.dtype)
        self.tags = _to_frozenset(tags) | {lx.symmetric_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply A to a vector by circulant embedding and the FFT.

            A x = [C [x; 0]]_{0:N},   C x = DFT⁻¹(DFT(c) ⊙ DFT(x))

        Args:
            vector: Input x, of length N.

        Returns:
            A x, of length N.
        """
        return _toeplitz_mv(self.column, vector)

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense A, N × N, with [A]_ij = t_|i−j|.

        Returns:
            The materialised Toeplitz matrix.
        """
        n = self._size
        # i, j = 0, …, N−1; N ≥ 1, so einx sees no empty axis
        positions = jnp.arange(n)
        # |i − j|
        lags = jnp.abs(einx.subtract("row, col -> row col", positions, positions))
        # [A]_ij = t_|i−j|
        return self.column[lags]

    def transpose(self) -> Toeplitz:
        """Aᵀ = A, since a symmetric Toeplitz matrix is its own transpose.

        Returns:
            The operator itself.
        """
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))


class ToeplitzCholesky(lx.AbstractLinearOperator):
    """Circulant-embedding sample factor L, N × M, with L Lᵀ = toeplitz(t).

        L z = [DFT⁻¹(√γ ⊙ DFT(z))]_{0:N},   γ = DFT(c)

    The embedding c ∈ ℝ^M, M = r N for the embedding factor r, is
    [t_0, …, t_N−1, 0, …, 0, t_N−1, …, t_1]. L is the first N rows of the
    symmetric root C^½ of its circulant C, a rectangular sample factor and
    *not* a lower-triangular Cholesky factor: L z with z ~ 𝒩(0, I_M) is a
    draw from 𝒩(0, toeplitz(t)) whenever γ ≥ 0 (the Wood--Chan condition).

    The Wood--Chan check uses ``eqx.error_if``, so it is jit-friendly and
    fires at evaluation time rather than at tracing time. If the spectrum has
    a materially negative eigenvalue, raise ``embedding_factor`` (4 or 8 is
    usually enough for a well-behaved covariance).

    Args:
        column: First column t ∈ ℝ^N of the Toeplitz covariance.
        embedding_factor: Embedding factor r ≥ 2, so the circulant has size
            M = r N.

    Raises:
        TypeError: If ``column`` is complex.
        ValueError: If ``column`` is not rank 1 or is empty, or
            ``embedding_factor`` is below 2.

    Examples:
        A 3 × 6 factor of a 3 × 3 Toeplitz covariance:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> column = jnp.array([1.0, 0.5, 0.1])
        >>> L = gaussx.ToeplitzCholesky(column)
        >>> L.as_matrix().shape
        (3, 6)

        It reproduces the covariance, L Lᵀ = A:

        >>> LLt = L.as_matrix() @ L.T.as_matrix()
        >>> A = gaussx.Toeplitz(column).as_matrix()
        >>> bool(jnp.allclose(LLt, A, atol=1e-5))
        True
    """

    sqrt_spectrum: Float[Array, " m_rfft"]
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
        # √γ on the non-negative DFT frequencies
        self.sqrt_spectrum = _circulant_sqrt_spectrum(
            column,
            embedding_factor=embedding_factor,
        )
        # N
        self._size = column.shape[0]
        # M = r N
        self._embedding_size = embedding_factor * self._size
        self._dtype = str(column.dtype)
        self.tags = frozenset()

    def mv(self, vector: Float[Array, " m"]) -> Float[Array, " n"]:
        """Apply L to a length-M vector.

            L z = [DFT⁻¹(√γ ⊙ DFT(z))]_{0:N}

        Args:
            vector: Input z, of length M.

        Returns:
            L z, of length N.
        """
        return _circulant_sqrt_mv(self.sqrt_spectrum, vector, self._size)

    def as_matrix(self) -> Float[Array, "n m"]:
        """Dense L, N × M, one column L e_m per basis vector.

        Returns:
            The materialised factor.
        """
        # I_M
        eye = jnp.eye(self._embedding_size, dtype=jnp.dtype(self._dtype))
        # [L e_1, …, L e_M]
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(eye)

    def transpose(self) -> _ToeplitzCholeskyTranspose:
        """Lᵀ, M × N, applied as C^½ [y; 0].

        Returns:
            The transposed factor.
        """
        return _ToeplitzCholeskyTranspose(self)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (M,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._embedding_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))


class _ToeplitzCholeskyTranspose(lx.AbstractLinearOperator):
    """Transpose Lᵀ, M × N, of a `ToeplitzCholesky` factor L.

        Lᵀ y = C^½ [y; 0_{M−N}] = DFT⁻¹(√γ ⊙ DFT([y; 0_{M−N}]))

    Attributes:
        factor: The factor L it transposes.
        tags: lineax tags, always empty.
    """

    factor: ToeplitzCholesky
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(self, factor: ToeplitzCholesky) -> None:
        self.factor = factor
        self.tags = frozenset()

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        """Apply Lᵀ to a length-N vector.

            Lᵀ y = C^½ [y; 0_{M−N}]

        Args:
            vector: Input y, of length N.

        Returns:
            Lᵀ y, of length M.
        """
        # 0_{M−N}
        padding = jnp.zeros(
            self.factor._embedding_size - self.factor._size,
            dtype=vector.dtype,
        )
        # [y; 0_{M−N}]
        padded = jnp.concatenate([vector, padding])
        # C^½ [y; 0_{M−N}], all M entries
        return _circulant_sqrt_mv(
            self.factor.sqrt_spectrum,
            padded,
            self.factor._embedding_size,
        )

    def as_matrix(self) -> Float[Array, "m n"]:
        """Dense Lᵀ, M × N, one column Lᵀ e_n per basis vector.

        Returns:
            The materialised transposed factor.
        """
        # I_N
        eye = jnp.eye(self.factor._size, dtype=jnp.dtype(self.factor._dtype))
        # [Lᵀ e_1, …, Lᵀ e_N]
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(eye)

    def transpose(self) -> ToeplitzCholesky:
        """(Lᵀ)ᵀ = L, the original factor.

        Returns:
            The `ToeplitzCholesky` factor.
        """
        return self.factor

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return self.factor.out_structure()

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (M,) and dtype of the output vector."""
        return self.factor.in_structure()


# L and Lᵀ are rectangular, so every structural predicate is False
for _check in (
    lx.is_symmetric,
    lx.is_diagonal,
    lx.is_positive_semidefinite,
    lx.is_negative_semidefinite,
    lx.is_tridiagonal,
    lx.is_lower_triangular,
    lx.is_upper_triangular,
    lx.has_unit_diagonal,
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
    """Draw from 𝒩(0, toeplitz(t)) by circulant embedding and the FFT.

        y⁽ᵉ⁾ = L z⁽ᵉ⁾,   z⁽ᵉ⁾ ~ 𝒩(0, I_M),   L Lᵀ = toeplitz(t)

    Algorithm:
        1. Build the sample factor L of `ToeplitzCholesky` for the embedding
           factor r, so M = r N.
        2. Draw ``num_samples`` white-noise vectors z⁽ᵉ⁾ of length M.
        3. Apply L to each draw.

    The Wood--Chan check is jit-friendly (``eqx.error_if``): if the embedding
    fails for the given ``embedding_factor``, the error fires at evaluation
    time. Raise ``embedding_factor`` (4 or 8) for a difficult covariance
    rather than relying on a runtime fallback.

    Args:
        column: First column t ∈ ℝ^N of the Toeplitz covariance.
        key: PRNG key for the white noise z⁽ᵉ⁾.
        num_samples: Number of independent draws.
        embedding_factor: Embedding factor r ≥ 2, so the circulant has size
            M = r N.

    Returns:
        The draws y⁽ᵉ⁾ stacked as rows, shape (``num_samples``, N).

    Raises:
        ValueError: If ``num_samples`` is below 1, ``column`` is not a
            non-empty rank-1 array, or ``embedding_factor`` is below 2.
        TypeError: If ``column`` is complex.

    Examples:
        Three draws from a 4 × 4 Toeplitz covariance:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> column = jnp.array([1.0, 0.5, 0.2, 0.05])
        >>> draws = gaussx.toeplitz_sample(column, key=jr.key(0), num_samples=3)
        >>> draws.shape
        (3, 4)

        The same key gives the same draws:

        >>> again = gaussx.toeplitz_sample(column, key=jr.key(0), num_samples=3)
        >>> bool(jnp.array_equal(draws, again))
        True
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    column = _as_floating_column(column)
    factor = ToeplitzCholesky(column, embedding_factor=embedding_factor)
    # z⁽ᵉ⁾ ~ 𝒩(0, I_M)
    noise = jr.normal(key, (num_samples, factor.in_size()), dtype=column.dtype)
    # y⁽ᵉ⁾ = L z⁽ᵉ⁾
    return jax.vmap(factor.mv)(noise)


def _as_floating_column(column: Float[Array, " n"]) -> Float[Array, " n"]:
    """Validate a first column t and promote it to a floating dtype.

    Args:
        column: Candidate first column t.

    Returns:
        t as a rank-1 floating array (at least float32).

    Raises:
        ValueError: If t is not rank 1 or is empty.
        TypeError: If t is complex.
    """
    column = jnp.asarray(column)
    if column.ndim != 1:
        raise ValueError(f"Toeplitz column must be rank 1, got shape {column.shape}.")
    if column.shape[0] == 0:
        raise ValueError("Toeplitz column must be non-empty.")
    if jnp.issubdtype(column.dtype, jnp.complexfloating):
        # The FFT matvec uses rfft, and the operator is tagged symmetric
        # (gh-368).
        raise TypeError(
            f"Toeplitz column must be real, got {column.dtype}; complex Toeplitz "
            "operators are not supported (use gaussx.circulant for complex "
            "circulant operators, or a dense operator)."
        )
    dtype = jnp.result_type(column.dtype, jnp.float32)
    return column.astype(dtype)


def _circulant_embedding(
    column: Float[Array, " n"],
    *,
    embedding_factor: int,
) -> Float[Array, " m"]:
    """First column c of the size-M circulant that embeds toeplitz(t).

        c = [t_0, …, t_N−1, 0_{M−2N+1}, t_N−1, …, t_1],   M = r N

    Args:
        column: First column t ∈ ℝ^N.
        embedding_factor: Embedding factor r ≥ 2.

    Returns:
        The embedding c, of length M.

    Raises:
        ValueError: If ``embedding_factor`` is below 2.
    """
    if embedding_factor < 2:
        raise ValueError("embedding_factor must be at least 2.")
    n = column.shape[0]
    # M = r N
    m = embedding_factor * n
    # 0_{M−2N+1}
    zeros = jnp.zeros(m - (2 * n - 1), dtype=column.dtype)
    # c = [t_0, …, t_N−1, 0, …, 0, t_N−1, …, t_1]
    return jnp.concatenate([column, zeros, column[:0:-1]])


def _circulant_sqrt_spectrum(
    column: Float[Array, " n"],
    *,
    embedding_factor: int,
) -> Float[Array, " m_rfft"]:
    """Root √γ of the embedding's spectrum γ = DFT(c), after Wood--Chan.

        γ = Re(DFT(c)),   √γ = √max(γ, 0)

    Algorithm:
        1. Embed t in the circulant first column c of size M = r N.
        2. Take γ on the non-negative frequencies with a real FFT.
        3. Reject γ with an entry below −100 ε max(1, max|γ|) at evaluation
           time, then clip round-off negatives to zero and take the root.

    Args:
        column: First column t ∈ ℝ^N.
        embedding_factor: Embedding factor r ≥ 2.

    Returns:
        √γ on the M // 2 + 1 non-negative frequencies.
    """
    # γ = Re(DFT(c)) on the non-negative frequencies
    spectrum = jnp.fft.rfft(
        _circulant_embedding(column, embedding_factor=embedding_factor),
    ).real
    # max(1, max|γ|)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(spectrum)))
    # Allow roundoff-sized negative FFT eigenvalues while rejecting real
    # failures. Uses eqx.error_if so the check is JIT-friendly (raises at
    # evaluation time rather than tracing time).
    tolerance = 100 * jnp.finfo(column.dtype).eps * scale
    spectrum = eqx.error_if(
        spectrum,
        jnp.any(spectrum < -tolerance),
        f"Circulant embedding failed the Wood-Chan non-negativity condition "
        f"for embedding_factor={embedding_factor}; "
        f"try embedding_factor={2 * embedding_factor}.",
    )
    # √max(γ, 0)
    return jnp.sqrt(jnp.clip(spectrum, min=0.0))


def _circulant_sqrt_mv(
    sqrt_spectrum: Float[Array, " m_rfft"],
    vector: Float[Array, " m"],
    output_size: int,
) -> Float[Array, " n"]:
    """Leading entries of C^½ z, the circulant root applied by the FFT.

        [C^½ z]_{0:K} = [DFT⁻¹(√γ ⊙ DFT(z))]_{0:K}

    Args:
        sqrt_spectrum: √γ on the non-negative frequencies.
        vector: Input z, of length M.
        output_size: Number K of leading entries to keep.

    Returns:
        The first K entries of C^½ z.
    """
    # M
    embedding_size = vector.shape[0]
    # C^½ z = DFT⁻¹(√γ ⊙ DFT(z))
    result = jnp.fft.irfft(
        sqrt_spectrum * jnp.fft.rfft(vector),
        n=embedding_size,
    )
    # [C^½ z]_{0:K}
    return result[:output_size]


def _toeplitz_mv(
    column: Float[Array, " n"],
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Toeplitz matvec A x by circulant embedding and the FFT.

        A x = [DFT⁻¹(DFT(c) ⊙ DFT([x; 0_{N−2}]))]_{0:N}

    Algorithm:
        1. For N ≤ 2 the embedding degenerates, so multiply by the dense A.
        2. Embed t in the circulant first column
           c = [t_0, …, t_N−1, t_N−2, …, t_1] of size M = 2N − 2.
        3. Zero-pad x to length M and convolve it with c by the FFT
           (the convolution theorem).
        4. Keep the first N entries.

    Args:
        column: First column t ∈ ℝ^N.
        vector: Input x, of length N.

    Returns:
        A x, of length N.
    """
    n = column.shape[0]
    # For n <= 2 the circulant embedding degenerates; use dense fallback.
    if n <= 2:
        # i, j = 0, …, N−1; N ≥ 1, so einx sees no empty axis
        positions = jnp.arange(n)
        # |i − j|
        lags = jnp.abs(einx.subtract("row, col -> row col", positions, positions))
        # A x with [A]_ij = t_|i−j|
        return column[lags] @ vector
    # M = 2N − 2
    m = 2 * n - 2
    # c = [t_0, …, t_N−1, t_N−2, …, t_1]
    c_embed = jnp.concatenate([column, column[-2:0:-1]])
    # [x; 0_{N−2}]
    v_embed = jnp.concatenate([vector, jnp.zeros(n - 2, dtype=vector.dtype)])
    # C [x; 0] = DFT⁻¹(DFT(c) ⊙ DFT([x; 0]))
    result = jnp.fft.irfft(jnp.fft.rfft(c_embed) * jnp.fft.rfft(v_embed), n=m)
    # A x = [C [x; 0]]_{0:N}
    return result[:n].real
