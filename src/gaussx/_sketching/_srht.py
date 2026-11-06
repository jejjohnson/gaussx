"""Subsampled randomized Hadamard transform (SRHT) sketch S ∈ ℝ^(d × M).

M₂ = 2^⌈log₂ M⌉ is the padded input size, H the M₂ × M₂ Walsh-Hadamard matrix, P a
random permutation of the M input rows, D a diagonal of random signs and R the
selection of d distinct rows of the padded transform.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum, rearrange
from gaussx._sketching._base import AbstractSketch
from gaussx._sketching._hadamard import hadamard_transform


class SRHTSketch(AbstractSketch):
    """Subsampled randomized Hadamard transform sketch.

        S = √(M₂ / d) R (H / √M₂) D P,   𝔼[Sᵀ S] = I_M

    P is a random permutation of the M input rows, D a diagonal of random signs, the
    result is zero-padded to M₂ = 2^⌈log₂ M⌉ rows, H / √M₂ is the orthonormal
    Walsh-Hadamard transform (`hadamard_transform`) and R selects d distinct rows
    uniformly at random. H D spreads each vector's mass evenly over the coordinates
    (flattens the leverage), so uniform row sampling afterwards is safe. A size
    d = O((N + log M) log N / ε²) suffices for an ε-subspace embedding (Tropp, 2011),
    and applying S to an M × N matrix costs O(M N log M).

    The padding to a power of two costs up to 2× the memory of the input while the
    transform runs.

    Attributes:
        permutation: Input row read by each position, (P x)_i = x_(permutation_i), of
            length M.
        signs: Diagonal of D, ±1, of length M.
        rows: The d distinct rows of the padded transform kept by R, of length d.
        in_size: The input size M.
        out_size: The sketch size d.

    Examples:
        M = 100 pads to M₂ = 128 internally, and the sketch stays d × M:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> sketch = gaussx.SRHTSketch.sample(jr.key(0), d=16, m=100)
        >>> sketch.apply(jnp.ones((100, 3))).shape
        (16, 3)
        >>> sketch.apply_transpose(jnp.ones((16, 3))).shape
        (100, 3)

        With M a power of two and d = M, S is orthogonal up to rounding:

        >>> square = gaussx.SRHTSketch.sample(jr.key(0), d=8, m=8)
        >>> dense = square.apply(jnp.eye(8))
        >>> bool(jnp.allclose(dense @ dense.T, jnp.eye(8), atol=1e-5))
        True
    """

    permutation: Int[Array, " m"]
    signs: Float[Array, " m"]
    rows: Int[Array, " d"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    @classmethod
    def sample(cls, key: jax.Array | None, d: int, m: int) -> SRHTSketch:
        """Draw a d × M SRHT sketch.

            P, a permutation of M rows,   D = diag(±1),   R, d distinct rows of M₂

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Sketch size d (rows of S); at most the padded size M₂.
            m: Input size M (columns of S).

        Returns:
            The sampled `SRHTSketch`.

        Raises:
            ValueError: If d exceeds the padded size M₂.

        Examples:
            d may exceed M as long as it fits the padded size:

            >>> import jax.random as jr
            >>> import gaussx
            >>> gaussx.SRHTSketch.sample(jr.key(0), d=8, m=5).rows.shape
            (8,)
            >>> gaussx.SRHTSketch.sample(jr.key(0), d=9, m=5)
            Traceback (most recent call last):
            ...
            ValueError: SRHTSketch needs d <= 8 (m=5 padded), got 9.
        """
        padded_size = _padded_size(m)
        if d > padded_size:
            raise ValueError(
                f"SRHTSketch needs d <= {padded_size} (m={m} padded), got {d}."
            )
        if key is None:
            key = jax.random.PRNGKey(0)
        perm_key, sign_key, row_key = jax.random.split(key, 3)
        # P
        permutation = jax.random.permutation(perm_key, m)
        # diag(D) = ±1
        signs = jax.random.rademacher(sign_key, (m,), dtype=float)
        # R
        rows = jax.random.choice(row_key, padded_size, (d,), replace=False)
        return cls(
            permutation=permutation,
            signs=signs,
            rows=rows,
            in_size=m,
            out_size=d,
        )

    # Jitted: eagerly, the gather, padding, transform and row selection would
    # each dispatch (and compile) as separate programs.
    @eqx.filter_jit
    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        """Compute S A along the leading axis of ``A``.

            S A = R H [D P A; 0] / √d

        Algorithm:
            1. Permute and sign the rows, D P A.
            2. Zero-pad to M₂ rows and apply H along the padded axis.
            3. Keep the rows R and scale by 1 / √d.

        Args:
            A: Array A whose leading axis has length M.

        Returns:
            The product S A, whose leading axis has length d.
        """
        # in the input's dtype, so a float32 input never meets a float64 sketch
        signs = self.signs.astype(A.dtype)
        # P A
        permuted = A[self.permutation]
        # D P A
        signed = einsum(signs, permuted, "m, m ... -> m ...")
        padded_size = _padded_size(self.in_size)
        padding = jnp.zeros((padded_size - self.in_size, *A.shape[1:]), dtype=A.dtype)
        # [D P A; 0]
        padded = jnp.concatenate([signed, padding])
        rows_last = rearrange(padded, "m ... -> ... m")
        # H [D P A; 0]
        transformed = hadamard_transform(rows_last)
        # R H [D P A; 0]
        selected = transformed[..., self.rows]
        rows_first = rearrange(selected, "... d -> d ...")
        # √(M₂ / d) R (H / √M₂) = R H / √d with the unnormalised transform
        return rows_first / math.sqrt(self.out_size)

    @eqx.filter_jit
    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        """Compute Sᵀ Y along the leading axis of ``Y``.

            Sᵀ Y = Pᵀ D (H Rᵀ Y)_1:M / √d

        Algorithm:
            1. Place Y on the rows R of M₂ zeros, Rᵀ Y, and apply H.
            2. Keep the first M rows, which the padding of `apply` read as zero.
            3. Sign and undo the permutation, then scale by 1 / √d.

        Args:
            Y: Array Y whose leading axis has length d.

        Returns:
            The product Sᵀ Y, whose leading axis has length M.
        """
        padded_size = _padded_size(self.in_size)
        zeros = jnp.zeros((padded_size, *Y.shape[1:]), dtype=Y.dtype)
        # Rᵀ Y
        padded = zeros.at[self.rows].set(Y)
        rows_last = rearrange(padded, "m ... -> ... m")
        # H Rᵀ Y
        transformed = hadamard_transform(rows_last)
        rows_first = rearrange(transformed, "... m -> m ...")
        # in the input's dtype, so a float32 input never meets a float64 sketch
        signs = self.signs.astype(Y.dtype)
        # (H Rᵀ Y)_1:M
        truncated = rows_first[: self.in_size]
        # D (H Rᵀ Y)_1:M
        signed = einsum(signs, truncated, "m, m ... -> m ...")
        permuted_zeros = jnp.zeros_like(signed)
        # Pᵀ D (H Rᵀ Y)_1:M
        unpermuted = permuted_zeros.at[self.permutation].set(signed)
        # Sᵀ Y = Pᵀ D (H Rᵀ Y)_1:M / √d
        return unpermuted / math.sqrt(self.out_size)


def _padded_size(m: int) -> int:
    """Smallest power of two that is at least M (and 1 when M ≤ 0).

        M₂ = 2^⌈log₂ M⌉

    Args:
        m: Input size M.

    Returns:
        The padded size M₂.
    """
    # ⌈log₂ M⌉
    exponent = max(m - 1, 0).bit_length()
    # M₂ = 2^⌈log₂ M⌉
    return 1 << exponent
