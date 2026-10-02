"""Block tridiagonal linear operator for state-space GP inference."""

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
    """Transpose each block; an empty stack passes through (einx rejects it)."""
    if blocks.shape[0] == 0:
        return blocks
    return rearrange(blocks, "N i j -> N j i")


class BlockTriDiag(lx.AbstractLinearOperator):
    r"""Block-tridiagonal operator with a symmetric off-diagonal band.

    Represents the structure:

        [D_1  A_1^T              ]
        [A_1  D_2   A_2^T        ]
        [     A_2   D_3   ...    ]
        [               A_{N-1} D_N]

    where ``D_k`` are ``(d, d)`` diagonal blocks and ``A_k`` are
    ``(d, d)`` sub-diagonal blocks. This is the precision matrix
    structure arising from discretized SDEs in state-space GP inference.

    All primitives (solve, logdet, cholesky, diag, trace) exploit the
    banded structure for O(Nd³) cost instead of O((Nd)³).

    The off-diagonal band is symmetric by construction (``A_k`` below,
    ``A_kᵀ`` above), so the operator is symmetric exactly when every ``D_k``
    is. With ``symmetric=True`` (the default) it carries
    ``lineax.symmetric_tag`` and `gaussx.solve` / `gaussx.logdet` use the
    block Cholesky, which reads only the lower triangle of each ``D_k``; a
    concrete non-symmetric ``D_k`` is rejected with a ``ValueError`` (the
    check is skipped under tracing, where the flag is a claim). Pass
    ``symmetric=False`` for non-symmetric diagonal blocks: the tag is then
    omitted and the solve and log-determinant fall back to a dense LU.

    ``+``, ``-``, unary ``-`` and scalar ``*`` keep the tags that still hold:
    symmetry when both operands are symmetric, positive-semidefiniteness for
    a sum of two PSD operators or a concretely non-negative scale (a negative
    one, or negation, swaps it with negative-semidefiniteness).

    Args:
        diagonal: Diagonal blocks, shape ``(N, d, d)``.
        sub_diagonal: Sub-diagonal blocks, shape ``(N-1, d, d)``.
        symmetric: Whether every ``D_k`` is symmetric (see above).
        tags: Additional lineax tags.

    Raises:
        ValueError: If ``symmetric=True`` and a concrete ``D_k`` is not
            symmetric to within ``sqrt(eps)`` of its largest entry.
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
        N, d, d2 = diagonal.shape
        if d != d2:
            raise ValueError(f"Diagonal blocks must be square, got ({d}, {d2}).")
        if sub_diagonal.shape[0] != N - 1:
            raise ValueError(
                f"sub_diagonal must have {N - 1} blocks, got {sub_diagonal.shape[0]}."
            )
        if sub_diagonal.shape[1] != d or sub_diagonal.shape[2] != d:
            raise ValueError(
                f"Sub-diagonal blocks must have shape ({d}, {d}), "
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
        self._num_blocks = N
        self._block_size = d
        self._size = N * d
        self._dtype = str(diagonal.dtype)
        structural: set[object] = {block_tridiagonal_tag}
        if symmetric:
            structural.add(lx.symmetric_tag)
        self.tags = _to_frozenset(tags) | structural

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        N = self._num_blocks
        d = self._block_size
        x = rearrange(vector, "(N d) -> N d", N=N, d=d)
        # Dₖ xₖ for all k
        result = einsum(self.diagonal, x, "N d1 d2, N d2 -> N d1")
        if N == 1:
            # einx cannot contract the empty (0, d, d) sub-diagonal (gh-304).
            return rearrange(result, "N d -> (N d)")
        # Aₖ xₖ₋₁ for k = 1, ..., N-1 (sub-diagonal)
        sub_contrib = einsum(self.sub_diagonal, x[:-1], "N d1 d2, N d2 -> N d1")
        result = result.at[1:].add(sub_contrib)
        # Aₖᵀ xₖ₊₁ for k = 0, ..., N-2 (super-diagonal)
        super_contrib = einsum(self.sub_diagonal, x[1:], "N d1 d2, N d1 -> N d2")
        result = result.at[:-1].add(super_contrib)
        return rearrange(result, "N d -> (N d)")

    def as_matrix(self) -> Float[Array, "n n"]:
        N = self._num_blocks
        d = self._block_size
        n = self._size
        mat = jnp.zeros((n, n), dtype=jnp.dtype(self._dtype))
        for k in range(N):
            r = k * d
            mat = mat.at[r : r + d, r : r + d].set(self.diagonal[k])
        for k in range(N - 1):
            r = (k + 1) * d
            c = k * d
            mat = mat.at[r : r + d, c : c + d].set(self.sub_diagonal[k])
            mat = mat.at[c : c + d, r : r + d].set(self.sub_diagonal[k].T)
        return mat

    def transpose(self) -> BlockTriDiag:
        # as_matrix puts sub[k] at (k+1,k) and sub[k].T at (k,k+1).
        # Transposing: new (k+1,k) = old (k,k+1).T = sub[k].
        # So new sub_diagonal = self.sub_diagonal (unchanged).
        return BlockTriDiag(
            rearrange(self.diagonal, "N i j -> N j i"),
            self.sub_diagonal,
            symmetric=self.symmetric,
            tags=lx.transpose_tags(self._extra_tags()),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def _extra_tags(self) -> frozenset[object]:
        """Tags other than the structural ones the constructor adds."""
        return self.tags - {block_tridiagonal_tag, lx.symmetric_tag}

    def _definiteness(self) -> frozenset[object]:
        return self.tags & frozenset(_DEFINITENESS_SWAP)

    def add(self, other: BlockTriDiag) -> BlockTriDiag:
        """Add two block-tridiagonal operators (e.g. prior + likelihood sites).

        The sum is symmetric if both operands are, and keeps a definiteness
        tag (PSD or NSD) that both operands carry.
        """
        return BlockTriDiag(
            self.diagonal + other.diagonal,
            self.sub_diagonal + other.sub_diagonal,
            symmetric=self.symmetric and other.symmetric,
            tags=self._definiteness() & other._definiteness(),
        )

    def __add__(self, other: BlockTriDiag) -> BlockTriDiag:
        return self.add(other)

    def __radd__(self, other: object) -> BlockTriDiag:
        if isinstance(other, BlockTriDiag):
            return other.add(self)
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: BlockTriDiag) -> BlockTriDiag:
        return BlockTriDiag(
            self.diagonal - other.diagonal,
            self.sub_diagonal - other.sub_diagonal,
            symmetric=self.symmetric and other.symmetric,
        )

    def __neg__(self) -> BlockTriDiag:
        return BlockTriDiag(
            -self.diagonal,
            -self.sub_diagonal,
            symmetric=self.symmetric,
            tags=frozenset(_DEFINITENESS_SWAP[tag] for tag in self._definiteness()),
        )

    def __mul__(self, other: object) -> BlockTriDiag:
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
        return self.__mul__(other)


def _is_concretely_symmetric_or_traced(diagonal: Float[Array, "N d d"]) -> bool:
    """Whether concrete diagonal blocks are symmetric; ``True`` when traced.

    The tolerance is ``sqrt(eps)`` relative to the largest entry, so the
    round-off asymmetry of a computed ``A Q Aᵀ`` passes while a genuinely
    non-symmetric block does not.
    """
    if isinstance(diagonal, Tracer):
        return True
    if diagonal.size == 0:
        return True
    asymmetry = jnp.max(jnp.abs(diagonal - rearrange(diagonal, "N i j -> N j i")))
    scale = jnp.max(jnp.abs(diagonal))
    return bool(asymmetry <= np.sqrt(jnp.finfo(diagonal.dtype).eps) * scale)


class LowerBlockTriDiag(lx.AbstractLinearOperator):
    """Lower triangular block-bidiagonal Cholesky factor.

    Represents:

        [L_1              ]
        [B_1  L_2          ]
        [     B_2  L_3     ]
        [          ...  L_N]

    where ``L_k`` are ``(d, d)`` lower-triangular blocks and ``B_k`` are
    ``(d, d)`` sub-diagonal blocks.

    Args:
        diagonal: Lower-triangular diagonal blocks, shape ``(N, d, d)``.
        sub_diagonal: Sub-diagonal blocks, shape ``(N-1, d, d)``.
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
        N, d, _ = diagonal.shape
        self.diagonal = diagonal
        self.sub_diagonal = sub_diagonal
        self._num_blocks = N
        self._block_size = d
        self._size = N * d
        self._dtype = str(diagonal.dtype)
        self.tags = _to_frozenset(tags) | {lx.lower_triangular_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        N = self._num_blocks
        d = self._block_size
        x = rearrange(vector, "(N d) -> N d", N=N, d=d)
        # Lₖ xₖ
        result = einsum(self.diagonal, x, "N d1 d2, N d2 -> N d1")
        if N == 1:
            return rearrange(result, "N d -> (N d)")
        # Bₖ xₖ₋₁
        sub_contrib = einsum(self.sub_diagonal, x[:-1], "N d1 d2, N d2 -> N d1")
        result = result.at[1:].add(sub_contrib)
        return rearrange(result, "N d -> (N d)")

    def as_matrix(self) -> Float[Array, "n n"]:
        N = self._num_blocks
        d = self._block_size
        n = self._size
        mat = jnp.zeros((n, n), dtype=jnp.dtype(self._dtype))
        for k in range(N):
            r = k * d
            mat = mat.at[r : r + d, r : r + d].set(self.diagonal[k])
        for k in range(N - 1):
            r = (k + 1) * d
            c = k * d
            mat = mat.at[r : r + d, c : c + d].set(self.sub_diagonal[k])
        return mat

    def transpose(self) -> UpperBlockTriDiag:
        """Transpose gives upper block-bidiagonal."""
        return UpperBlockTriDiag(
            rearrange(self.diagonal, "N i j -> N j i"),
            _transpose_blocks(self.sub_diagonal),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))


class UpperBlockTriDiag(lx.AbstractLinearOperator):
    """Upper triangular block-bidiagonal (transpose of LowerBlockTriDiag).

    Represents:

        [U_1  C_1            ]
        [     U_2  C_2        ]
        [          ...   C_{N-1}]
        [               U_N  ]

    where ``U_k`` are upper-triangular diagonal blocks and ``C_k`` are
    super-diagonal blocks.
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
        N, d, _ = diagonal.shape
        self.diagonal = diagonal
        self.super_diagonal = super_diagonal
        self._num_blocks = N
        self._block_size = d
        self._size = N * d
        self._dtype = str(diagonal.dtype)
        self.tags = _to_frozenset(tags) | {lx.upper_triangular_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        N = self._num_blocks
        d = self._block_size
        x = rearrange(vector, "(N d) -> N d", N=N, d=d)
        # Uₖ xₖ
        result = einsum(self.diagonal, x, "N d1 d2, N d2 -> N d1")
        if N == 1:
            return rearrange(result, "N d -> (N d)")
        # Cₖ xₖ₊₁
        super_contrib = einsum(self.super_diagonal, x[1:], "N d1 d2, N d2 -> N d1")
        result = result.at[:-1].add(super_contrib)
        return rearrange(result, "N d -> (N d)")

    def as_matrix(self) -> Float[Array, "n n"]:
        N = self._num_blocks
        d = self._block_size
        n = self._size
        mat = jnp.zeros((n, n), dtype=jnp.dtype(self._dtype))
        for k in range(N):
            r = k * d
            mat = mat.at[r : r + d, r : r + d].set(self.diagonal[k])
        for k in range(N - 1):
            r = k * d
            c = (k + 1) * d
            mat = mat.at[r : r + d, c : c + d].set(self.super_diagonal[k])
        return mat

    def transpose(self) -> LowerBlockTriDiag:
        return LowerBlockTriDiag(
            rearrange(self.diagonal, "N i j -> N j i"),
            _transpose_blocks(self.super_diagonal),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))
