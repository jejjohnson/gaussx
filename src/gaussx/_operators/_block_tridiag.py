"""Block tridiagonal linear operator for state-space GP inference."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset


class BlockTriDiag(lx.AbstractLinearOperator):
    r"""Symmetric block-tridiagonal operator.

    Represents the structure::

        [D_1  A_1^T              ]
        [A_1  D_2   A_2^T        ]
        [     A_2   D_3   ...    ]
        [               A_{N-1} D_N]

    where ``D_k`` are ``(d, d)`` diagonal blocks and ``A_k`` are
    ``(d, d)`` sub-diagonal blocks. This is the precision matrix
    structure arising from discretized SDEs in state-space GP inference.

    All primitives (solve, logdet, cholesky, diag, trace) exploit the
    banded structure for O(Nd³) cost instead of O((Nd)³).

    Args:
        diagonal: Diagonal blocks, shape ``(N, d, d)``.
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
        self.diagonal = diagonal
        self.sub_diagonal = sub_diagonal
        self._num_blocks = N
        self._block_size = d
        self._size = N * d
        self._dtype = str(diagonal.dtype)
        from gaussx._tags import block_tridiagonal_tag

        self.tags = _to_frozenset(tags) | {block_tridiagonal_tag, lx.symmetric_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        N = self._num_blocks
        d = self._block_size
        x = vector.reshape(N, d)
        result = jnp.zeros_like(x)
        # D_k x_k for all k
        result = jax.vmap(jnp.dot)(self.diagonal, x)
        # A_k x_{k-1} for k = 1, ..., N-1 (sub-diagonal contribution)
        sub_contrib = jax.vmap(jnp.dot)(self.sub_diagonal, x[:-1])
        result = result.at[1:].add(sub_contrib)
        # A_k^T x_{k+1} for k = 0, ..., N-2 (super-diagonal contribution)
        super_contrib = jax.vmap(lambda A, v: A.T @ v)(self.sub_diagonal, x[1:])
        result = result.at[:-1].add(super_contrib)
        return result.reshape(-1)

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
            jax.vmap(jnp.transpose)(self.diagonal),
            self.sub_diagonal,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def add(self, other: BlockTriDiag) -> BlockTriDiag:
        """Add two block-tridiagonal operators (e.g. prior + likelihood sites)."""
        return BlockTriDiag(
            self.diagonal + other.diagonal,
            self.sub_diagonal + other.sub_diagonal,
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
        )

    def __neg__(self) -> BlockTriDiag:
        return BlockTriDiag(-self.diagonal, -self.sub_diagonal)

    def __mul__(self, other: object) -> BlockTriDiag:
        scalar = jnp.asarray(other)
        if scalar.ndim != 0:
            msg = "BlockTriDiag can only be multiplied by a scalar"
            raise TypeError(msg)
        return BlockTriDiag(scalar * self.diagonal, scalar * self.sub_diagonal)

    def __rmul__(self, other: object) -> BlockTriDiag:
        return self.__mul__(other)


class LowerBlockTriDiag(lx.AbstractLinearOperator):
    """Lower triangular block-bidiagonal Cholesky factor.

    Represents::

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
        x = vector.reshape(N, d)
        # L_k x_k
        result = jax.vmap(jnp.dot)(self.diagonal, x)
        # B_k x_{k-1}
        sub_contrib = jax.vmap(jnp.dot)(self.sub_diagonal, x[:-1])
        result = result.at[1:].add(sub_contrib)
        return result.reshape(-1)

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
            jax.vmap(jnp.transpose)(self.diagonal),
            jax.vmap(jnp.transpose)(self.sub_diagonal),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))


class UpperBlockTriDiag(lx.AbstractLinearOperator):
    """Upper triangular block-bidiagonal (transpose of LowerBlockTriDiag).

    Represents::

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
        x = vector.reshape(N, d)
        # U_k x_k
        result = jax.vmap(jnp.dot)(self.diagonal, x)
        # C_k x_{k+1}
        super_contrib = jax.vmap(jnp.dot)(self.super_diagonal, x[1:])
        result = result.at[:-1].add(super_contrib)
        return result.reshape(-1)

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
            jax.vmap(jnp.transpose)(self.diagonal),
            jax.vmap(jnp.transpose)(self.super_diagonal),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))
