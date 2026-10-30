"""Block-tridiagonal operator and its block-bidiagonal triangular factors.

A symmetric block-tridiagonal operator A has B diagonal blocks A_bb and
sub-diagonal blocks A_b,b−1, b = 1, …, B, each d × d, with A_b,b−1ᵀ above
the diagonal; it is the precision of a discretised SDE in state-space GP
inference. Its Cholesky factor L is lower block-bidiagonal with diagonal
blocks L_bb and sub-diagonal blocks L_b,b−1, and Lᵀ is upper
block-bidiagonal with diagonal blocks U_bb and super-diagonal blocks
U_b,b+1. The input splits into consecutive blocks x = [x_1; …; x_B] with
x_b ∈ ℝ^d.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.core import Tracer
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import _to_frozenset
from gaussx._tags import block_tridiagonal_tag


_DEFINITENESS_SWAP: dict[object, object] = {
    lx.positive_semidefinite_tag: lx.negative_semidefinite_tag,
    lx.negative_semidefinite_tag: lx.positive_semidefinite_tag,
}


def _transpose_blocks(blocks: Float[Array, "N d d"]) -> Float[Array, "N d d"]:
    """Transpose every block of a stack, passing an empty stack through.

        [M_1, …, M_K] ↦ [M_1ᵀ, …, M_Kᵀ]

    Args:
        blocks: Stack of square blocks M_k; may hold no blocks.

    Returns:
        The stack of transposed blocks.
    """
    if blocks.shape[0] == 0:
        # einx rejects the empty block axis
        return blocks
    # M_kᵀ, per block
    return rearrange(blocks, "block row col -> block col row")


class BlockTriDiag(lx.AbstractLinearOperator):
    """Block-tridiagonal operator A with a symmetric off-diagonal band.

        [A_11   A_21ᵀ                ]
        [A_21   A_22    A_32ᵀ        ]
        [       A_32    A_33    ⋱    ]
        [               ⋱       A_BB ]

        (A x)_b = A_b,b−1 x_b−1 + A_bb x_b + A_b+1,bᵀ x_b+1

    The B diagonal blocks A_bb and the B − 1 sub-diagonal blocks A_b,b−1 are
    d × d. This is the precision structure of a discretised SDE in
    state-space GP inference, and every primitive (solve, logdet, cholesky,
    diag, trace) exploits the band in O(B d³) instead of O((B d)³).

    The off-diagonal band is symmetric by construction (A_b,b−1 below,
    A_b,b−1ᵀ above), so A is symmetric exactly when every A_bb is. With
    ``symmetric=True`` (the default) it carries ``lineax.symmetric_tag``
    and `gaussx.solve` / `gaussx.logdet` use the block Cholesky, which reads
    only the lower triangle of each A_bb; a concrete non-symmetric A_bb is
    rejected (the check is skipped under tracing, where the flag is a
    claim). Pass ``symmetric=False`` for non-symmetric diagonal blocks: the
    tag is then omitted and the solve and log-determinant fall back to a
    dense LU.

    ``+``, ``-``, unary ``-`` and scalar ``*`` keep the tags that still hold:
    symmetry when both operands are symmetric, positive-semidefiniteness for
    a sum of two PSD operators or a concretely non-negative scale (a negative
    one, or negation, swaps it with negative-semidefiniteness).

    Args:
        diagonal: Diagonal blocks A_11, …, A_BB.
        sub_diagonal: Sub-diagonal blocks A_21, …, A_B,B−1.
        symmetric: Whether every A_bb is symmetric (see above).
        tags: Extra lineax tags for A; ``block_tridiagonal_tag`` is always
            added, and ``symmetric_tag`` when ``symmetric`` is set.

    Raises:
        ValueError: If the blocks are not stacks of square d × d blocks, if
            there are not B − 1 sub-diagonal blocks, or if ``symmetric=True``
            and a concrete A_bb is not symmetric to within √ε of its largest
            entry.

    Examples:
        Three 2 × 2 blocks give a 6 × 6 operator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> diagonal = jnp.stack([3.0 * jnp.eye(2)] * 3)
        >>> sub_diagonal = jnp.stack([0.5 * jnp.eye(2)] * 2)
        >>> A = gaussx.BlockTriDiag(diagonal, sub_diagonal)
        >>> A.as_matrix().shape
        (6, 6)

        The solve runs the block-banded recurrence in O(B d³):

        >>> b = jnp.ones(6)
        >>> x = gaussx.solve(A, b)
        >>> bool(jnp.allclose(A.as_matrix() @ x, b, atol=1e-5))
        True

        Sums of block-tridiagonal operators stay block-tridiagonal:

        >>> type(A + A).__name__
        'BlockTriDiag'
    """

    diagonal: Float[Array, "N d d"]
    sub_diagonal: Float[Array, "Nm1 d d"]
    symmetric: bool = eqx.field(static=True)
    _num_blocks: int = eqx.field(static=True)
    _block_size: int = eqx.field(static=True)
    _size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        diagonal: Float[Array, "N d d"],
        sub_diagonal: Float[Array, "Nm1 d d"],
        *,
        symmetric: bool = True,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if diagonal.ndim != 3:
            raise ValueError(
                f"diagonal must have 3 dimensions (N, d, d), got {diagonal.ndim}."
            )
        if sub_diagonal.ndim != 3:
            raise ValueError(
                f"sub_diagonal must have 3 dimensions (N-1, d, d), "
                f"got {sub_diagonal.ndim}."
            )
        num_blocks, block_size, block_cols = diagonal.shape
        if block_size != block_cols:
            raise ValueError(
                f"Diagonal blocks must be square, got ({block_size}, {block_cols})."
            )
        if sub_diagonal.shape[0] != num_blocks - 1:
            raise ValueError(
                f"sub_diagonal must have {num_blocks - 1} blocks, "
                f"got {sub_diagonal.shape[0]}."
            )
        if sub_diagonal.shape[1] != block_size or sub_diagonal.shape[2] != block_size:
            raise ValueError(
                f"Sub-diagonal blocks must have shape ({block_size}, {block_size}), "
                f"got ({sub_diagonal.shape[1]}, {sub_diagonal.shape[2]})."
            )
        if symmetric and not _is_concretely_symmetric_or_traced(diagonal):
            raise ValueError(
                "BlockTriDiag diagonal blocks are not symmetric; pass "
                "symmetric=False for a non-symmetric operator (gh-344)."
            )
        self.diagonal = diagonal
        self.sub_diagonal = sub_diagonal
        self.symmetric = symmetric
        self._num_blocks = num_blocks
        self._block_size = block_size
        # N = B d
        self._size = num_blocks * block_size
        self._dtype = str(diagonal.dtype)
        structural: set[object] = {block_tridiagonal_tag}
        if symmetric:
            structural.add(lx.symmetric_tag)
        self.tags = _to_frozenset(tags) | structural

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply A to a vector through its three block bands.

            (A x)_b = A_b,b−1 x_b−1 + A_bb x_b + A_b+1,bᵀ x_b+1

        Args:
            vector: Input x = [x_1; …; x_B], of length N = B d.

        Returns:
            A x, of length N.
        """
        num_blocks = self._num_blocks
        block_size = self._block_size
        # [x_1; …; x_B] ↦ rows x_b
        input_blocks = rearrange(
            vector, "(block row) -> block row", block=num_blocks, row=block_size
        )
        # A_bb x_b, every block
        output_blocks = einsum(
            self.diagonal, input_blocks, "block row col, block col -> block row"
        )
        if num_blocks == 1:
            # einx cannot contract the empty (0, d, d) sub-diagonal (gh-304).
            return rearrange(output_blocks, "block row -> (block row)")
        # A_b,b−1 x_b−1, b = 2, …, B
        below_diagonal = einsum(
            self.sub_diagonal,
            input_blocks[:-1],
            "block row col, block col -> block row",
        )
        # + A_b,b−1 x_b−1
        output_blocks = output_blocks.at[1:].add(below_diagonal)
        # A_b+1,bᵀ x_b+1, b = 1, …, B − 1
        above_diagonal = einsum(
            self.sub_diagonal,
            input_blocks[1:],
            "block row col, block row -> block col",
        )
        # + A_b+1,bᵀ x_b+1
        output_blocks = output_blocks.at[:-1].add(above_diagonal)
        # [(A x)_1; …; (A x)_B]
        return rearrange(output_blocks, "block row -> (block row)")

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense block-tridiagonal A, N × N.

        Returns:
            The materialised matrix.
        """
        num_blocks = self._num_blocks
        block_size = self._block_size
        matrix = jnp.zeros((self._size, self._size), dtype=jnp.dtype(self._dtype))
        for block_index in range(num_blocks):
            start = block_index * block_size
            stop = start + block_size
            # A_bb on the diagonal
            matrix = matrix.at[start:stop, start:stop].set(self.diagonal[block_index])
        for block_index in range(num_blocks - 1):
            row_start = (block_index + 1) * block_size
            row_stop = row_start + block_size
            col_start = block_index * block_size
            col_stop = col_start + block_size
            sub_block = self.sub_diagonal[block_index]
            # A_b+1,b below the diagonal
            matrix = matrix.at[row_start:row_stop, col_start:col_stop].set(sub_block)
            # A_b+1,bᵀ above the diagonal
            # jnp, not einx: einx rejects the empty axes of a d = 0 block
            matrix = matrix.at[col_start:col_stop, row_start:row_stop].set(sub_block.T)
        return matrix

    def transpose(self) -> BlockTriDiag:
        """Aᵀ, with diagonal blocks A_bbᵀ and the same band.

            (Aᵀ)_bb = A_bbᵀ,   (Aᵀ)_b,b−1 = (A_b−1,b)ᵀ = A_b,b−1

        The band is A_b,b−1 below and its transpose above, so transposing
        swaps the two and leaves ``sub_diagonal`` unchanged.

        Returns:
            The transposed `BlockTriDiag`, with transposed extra tags.
        """
        # A_bbᵀ, per block
        transposed_diagonal = rearrange(self.diagonal, "block row col -> block col row")
        return BlockTriDiag(
            transposed_diagonal,
            self.sub_diagonal,
            symmetric=self.symmetric,
            tags=lx.transpose_tags(self._extra_tags()),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def _extra_tags(self) -> frozenset[object]:
        """Tags other than the structural ones the constructor adds."""
        return self.tags - {block_tridiagonal_tag, lx.symmetric_tag}

    def _definiteness(self) -> frozenset[object]:
        """The PSD or NSD tag A carries, if any."""
        return self.tags & frozenset(_DEFINITENESS_SWAP)

    def add(self, other: BlockTriDiag) -> BlockTriDiag:
        """Add two block-tridiagonal operators (e.g. prior + likelihood sites).

            (A + A′)_bb = A_bb + A′_bb,   (A + A′)_b,b−1 = A_b,b−1 + A′_b,b−1

        The sum is symmetric if both operands are, and keeps a definiteness
        tag (PSD or NSD) that both operands carry.

        Args:
            other: The second operand A′, with the same block layout.

        Returns:
            A + A′ as a `BlockTriDiag`.
        """
        return BlockTriDiag(
            self.diagonal + other.diagonal,
            self.sub_diagonal + other.sub_diagonal,
            symmetric=self.symmetric and other.symmetric,
            tags=self._definiteness() & other._definiteness(),
        )

    def __add__(self, other):
        """A + A′ blockwise for a `BlockTriDiag`, else a lineax sum."""
        if not isinstance(other, BlockTriDiag):
            # A foreign operand (e.g. ``s * Identity``) becomes a lineax sum
            # instead of crashing on ``other.diagonal`` (gh-337).
            return super().__add__(other)
        return self.add(other)

    def __radd__(self, other: object) -> BlockTriDiag:
        """A′ + A for a `BlockTriDiag`; 0 + A = A, so ``sum`` works."""
        if isinstance(other, BlockTriDiag):
            return other.add(self)
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other):
        """A − A′ blockwise for a `BlockTriDiag`, else a lineax difference.

        The difference keeps symmetry but no definiteness tag.
        """
        if not isinstance(other, BlockTriDiag):
            return super().__sub__(other)
        return BlockTriDiag(
            self.diagonal - other.diagonal,
            self.sub_diagonal - other.sub_diagonal,
            symmetric=self.symmetric and other.symmetric,
        )

    def __neg__(self) -> BlockTriDiag:
        """−A blockwise; a PSD tag becomes NSD and vice versa."""
        return BlockTriDiag(
            -self.diagonal,
            -self.sub_diagonal,
            symmetric=self.symmetric,
            tags=frozenset(_DEFINITENESS_SWAP[tag] for tag in self._definiteness()),
        )

    def __mul__(self, other: object) -> BlockTriDiag:
        """c A blockwise for a rank-0 scalar c.

            (c A)_bb = c A_bb,   (c A)_b,b−1 = c A_b,b−1

        A definiteness tag survives only a concrete c, kept for c ≥ 0 and
        swapped for c < 0; a traced c drops it.

        Raises:
            TypeError: If ``other`` is not a rank-0 scalar.
        """
        scalar = jnp.asarray(other)
        if scalar.ndim != 0:
            msg = "BlockTriDiag can only be multiplied by a scalar"
            raise TypeError(msg)
        # A definiteness tag survives a scale of known sign only.
        tags: frozenset[object] = frozenset()
        if not isinstance(scalar, Tracer):
            value = float(np.real(np.asarray(scalar)))
            if value >= 0:
                tags = self._definiteness()
            else:
                tags = frozenset(_DEFINITENESS_SWAP[t] for t in self._definiteness())
        return BlockTriDiag(
            scalar * self.diagonal,
            scalar * self.sub_diagonal,
            symmetric=self.symmetric,
            tags=tags,
        )

    def __rmul__(self, other: object) -> BlockTriDiag:
        """A c = c A for a rank-0 scalar c."""
        return self.__mul__(other)


def _is_concretely_symmetric_or_traced(diagonal: Float[Array, "N d d"]) -> bool:
    """Whether concrete diagonal blocks A_bb are symmetric; ``True`` if traced.

        max_b ‖A_bb − A_bbᵀ‖_max ≤ √ε max_b ‖A_bb‖_max

    The tolerance is relative to the largest entry, so the round-off
    asymmetry of a computed F Σ Fᵀ passes while a genuinely non-symmetric
    block does not.

    Args:
        diagonal: Diagonal blocks A_11, …, A_BB.

    Returns:
        Whether every block is symmetric to tolerance; ``True`` for traced
        or empty blocks, where nothing can be checked.
    """
    if isinstance(diagonal, Tracer):
        return True
    if diagonal.size == 0:
        return True
    # A_bbᵀ, per block
    transposed = rearrange(diagonal, "block row col -> block col row")
    # max_b ‖A_bb − A_bbᵀ‖_max
    asymmetry = jnp.max(jnp.abs(diagonal - transposed))
    # max_b ‖A_bb‖_max
    scale = jnp.max(jnp.abs(diagonal))
    # √ε max_b ‖A_bb‖_max
    tolerance = np.sqrt(jnp.finfo(diagonal.dtype).eps) * scale
    return bool(asymmetry <= tolerance)


class LowerBlockTriDiag(lx.AbstractLinearOperator):
    """Lower block-bidiagonal operator L, the block Cholesky factor of A.

        [L_11                    ]
        [L_21   L_22             ]
        [       L_32   L_33      ]
        [              ⋱    L_BB ]

        (L x)_b = L_b,b−1 x_b−1 + L_bb x_b

    The diagonal blocks L_bb are d × d and lower-triangular, and the
    sub-diagonal blocks L_b,b−1 are d × d. `gaussx.cholesky` returns one
    for a symmetric positive-definite `BlockTriDiag`, and `gaussx.solve`
    runs block forward substitution on it.

    Args:
        diagonal: Lower-triangular diagonal blocks L_11, …, L_BB.
        sub_diagonal: Sub-diagonal blocks L_21, …, L_B,B−1.
        tags: Extra lineax tags for L; ``lower_triangular_tag`` is always
            added.

    Examples:
        The factor of a block-tridiagonal operator rebuilds it:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> diagonal = jnp.stack([3.0 * jnp.eye(2)] * 3)
        >>> sub_diagonal = jnp.stack([0.5 * jnp.eye(2)] * 2)
        >>> psd = lx.positive_semidefinite_tag
        >>> A = gaussx.BlockTriDiag(diagonal, sub_diagonal, tags=psd)
        >>> L = gaussx.cholesky(A)
        >>> type(L).__name__
        'LowerBlockTriDiag'
        >>> rebuilt = L.as_matrix() @ L.T.as_matrix()
        >>> bool(jnp.allclose(rebuilt, A.as_matrix(), atol=1e-5))
        True

        Its transpose is upper block-bidiagonal:

        >>> type(L.T).__name__
        'UpperBlockTriDiag'
    """

    diagonal: Float[Array, "N d d"]
    sub_diagonal: Float[Array, "Nm1 d d"]
    _num_blocks: int = eqx.field(static=True)
    _block_size: int = eqx.field(static=True)
    _size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        diagonal: Float[Array, "N d d"],
        sub_diagonal: Float[Array, "Nm1 d d"],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        num_blocks, block_size, _ = diagonal.shape
        self.diagonal = diagonal
        self.sub_diagonal = sub_diagonal
        self._num_blocks = num_blocks
        self._block_size = block_size
        # N = B d
        self._size = num_blocks * block_size
        self._dtype = str(diagonal.dtype)
        self.tags = _to_frozenset(tags) | {lx.lower_triangular_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply L to a vector through its two block bands.

            (L x)_b = L_b,b−1 x_b−1 + L_bb x_b

        Args:
            vector: Input x = [x_1; …; x_B], of length N = B d.

        Returns:
            L x, of length N.
        """
        num_blocks = self._num_blocks
        block_size = self._block_size
        # [x_1; …; x_B] ↦ rows x_b
        input_blocks = rearrange(
            vector, "(block row) -> block row", block=num_blocks, row=block_size
        )
        # L_bb x_b, every block
        output_blocks = einsum(
            self.diagonal, input_blocks, "block row col, block col -> block row"
        )
        if num_blocks == 1:
            # einx cannot contract the empty (0, d, d) sub-diagonal (gh-304).
            return rearrange(output_blocks, "block row -> (block row)")
        # L_b,b−1 x_b−1, b = 2, …, B
        below_diagonal = einsum(
            self.sub_diagonal,
            input_blocks[:-1],
            "block row col, block col -> block row",
        )
        # + L_b,b−1 x_b−1
        output_blocks = output_blocks.at[1:].add(below_diagonal)
        # [(L x)_1; …; (L x)_B]
        return rearrange(output_blocks, "block row -> (block row)")

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense lower block-bidiagonal L, N × N.

        Returns:
            The materialised matrix.
        """
        num_blocks = self._num_blocks
        block_size = self._block_size
        matrix = jnp.zeros((self._size, self._size), dtype=jnp.dtype(self._dtype))
        for block_index in range(num_blocks):
            start = block_index * block_size
            stop = start + block_size
            # L_bb on the diagonal
            matrix = matrix.at[start:stop, start:stop].set(self.diagonal[block_index])
        for block_index in range(num_blocks - 1):
            row_start = (block_index + 1) * block_size
            row_stop = row_start + block_size
            col_start = block_index * block_size
            col_stop = col_start + block_size
            # L_b+1,b below the diagonal
            matrix = matrix.at[row_start:row_stop, col_start:col_stop].set(
                self.sub_diagonal[block_index]
            )
        return matrix

    def transpose(self) -> UpperBlockTriDiag:
        """Lᵀ, upper block-bidiagonal.

            (Lᵀ)_bb = L_bbᵀ,   (Lᵀ)_b,b+1 = L_b+1,bᵀ

        Returns:
            The transpose as an `UpperBlockTriDiag`.
        """
        # L_bbᵀ, per block
        transposed_diagonal = rearrange(self.diagonal, "block row col -> block col row")
        # L_b+1,bᵀ, per block; B = 1 has none
        transposed_sub_diagonal = _transpose_blocks(self.sub_diagonal)
        return UpperBlockTriDiag(transposed_diagonal, transposed_sub_diagonal)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))


class UpperBlockTriDiag(lx.AbstractLinearOperator):
    """Upper block-bidiagonal operator U, the transpose of a `LowerBlockTriDiag`.

        [U_11   U_12                 ]
        [       U_22   U_23          ]
        [              ⋱    U_B−1,B  ]
        [                   U_BB     ]

        (U x)_b = U_bb x_b + U_b,b+1 x_b+1

    The diagonal blocks U_bb are d × d and upper-triangular, and the
    super-diagonal blocks U_b,b+1 are d × d. For U = Lᵀ, U_bb = L_bbᵀ and
    U_b,b+1 = L_b+1,bᵀ; `gaussx.solve` runs block backward substitution on
    it.

    Args:
        diagonal: Upper-triangular diagonal blocks U_11, …, U_BB.
        super_diagonal: Super-diagonal blocks U_12, …, U_B−1,B.
        tags: Extra lineax tags for U; ``upper_triangular_tag`` is always
            added.

    Examples:
        Two 2 × 2 blocks, applied to a vector:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> diagonal = jnp.stack([2.0 * jnp.eye(2)] * 2)
        >>> super_diagonal = jnp.ones((1, 2, 2))
        >>> U = gaussx.UpperBlockTriDiag(diagonal, super_diagonal)
        >>> U.mv(jnp.ones(4)).tolist()
        [4.0, 4.0, 2.0, 2.0]
        >>> type(U.T).__name__
        'LowerBlockTriDiag'
    """

    diagonal: Float[Array, "N d d"]
    super_diagonal: Float[Array, "Nm1 d d"]
    _num_blocks: int = eqx.field(static=True)
    _block_size: int = eqx.field(static=True)
    _size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        diagonal: Float[Array, "N d d"],
        super_diagonal: Float[Array, "Nm1 d d"],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        num_blocks, block_size, _ = diagonal.shape
        self.diagonal = diagonal
        self.super_diagonal = super_diagonal
        self._num_blocks = num_blocks
        self._block_size = block_size
        # N = B d
        self._size = num_blocks * block_size
        self._dtype = str(diagonal.dtype)
        self.tags = _to_frozenset(tags) | {lx.upper_triangular_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply U to a vector through its two block bands.

            (U x)_b = U_bb x_b + U_b,b+1 x_b+1

        Args:
            vector: Input x = [x_1; …; x_B], of length N = B d.

        Returns:
            U x, of length N.
        """
        num_blocks = self._num_blocks
        block_size = self._block_size
        # [x_1; …; x_B] ↦ rows x_b
        input_blocks = rearrange(
            vector, "(block row) -> block row", block=num_blocks, row=block_size
        )
        # U_bb x_b, every block
        output_blocks = einsum(
            self.diagonal, input_blocks, "block row col, block col -> block row"
        )
        if num_blocks == 1:
            # einx cannot contract the empty (0, d, d) super-diagonal (gh-304).
            return rearrange(output_blocks, "block row -> (block row)")
        # U_b,b+1 x_b+1, b = 1, …, B − 1
        above_diagonal = einsum(
            self.super_diagonal,
            input_blocks[1:],
            "block row col, block col -> block row",
        )
        # + U_b,b+1 x_b+1
        output_blocks = output_blocks.at[:-1].add(above_diagonal)
        # [(U x)_1; …; (U x)_B]
        return rearrange(output_blocks, "block row -> (block row)")

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense upper block-bidiagonal U, N × N.

        Returns:
            The materialised matrix.
        """
        num_blocks = self._num_blocks
        block_size = self._block_size
        matrix = jnp.zeros((self._size, self._size), dtype=jnp.dtype(self._dtype))
        for block_index in range(num_blocks):
            start = block_index * block_size
            stop = start + block_size
            # U_bb on the diagonal
            matrix = matrix.at[start:stop, start:stop].set(self.diagonal[block_index])
        for block_index in range(num_blocks - 1):
            row_start = block_index * block_size
            row_stop = row_start + block_size
            col_start = (block_index + 1) * block_size
            col_stop = col_start + block_size
            # U_b,b+1 above the diagonal
            matrix = matrix.at[row_start:row_stop, col_start:col_stop].set(
                self.super_diagonal[block_index]
            )
        return matrix

    def transpose(self) -> LowerBlockTriDiag:
        """Uᵀ, lower block-bidiagonal.

            (Uᵀ)_bb = U_bbᵀ,   (Uᵀ)_b+1,b = U_b,b+1ᵀ

        Returns:
            The transpose as a `LowerBlockTriDiag`.
        """
        # U_bbᵀ, per block
        transposed_diagonal = rearrange(self.diagonal, "block row col -> block col row")
        # U_b,b+1ᵀ, per block; B = 1 has none
        transposed_super_diagonal = _transpose_blocks(self.super_diagonal)
        return LowerBlockTriDiag(transposed_diagonal, transposed_super_diagonal)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))
