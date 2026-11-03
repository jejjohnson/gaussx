"""Block-diagonal operator A = blockdiag(A_1, …, A_D) as a lineax operator.

A_d is the d-th diagonal block, M_d × N_d, d = 1, …, D, so A is M × N with
M = ∑_d M_d and N = ∑_d N_d. The input splits into consecutive slices
x = [x_1; …; x_D] with x_d ∈ ℝ^{N_d}, and block d acts on slice d alone.
"""

from __future__ import annotations

from collections.abc import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._tags import block_diagonal_tag


def _resolve_dtype(*operators: lx.AbstractLinearOperator) -> str:
    """Common dtype of the operators' output structures, as a string.

    Args:
        *operators: The operators whose output dtypes are promoted together.

    Returns:
        The promoted dtype name, e.g. ``"float32"``.
    """
    dtypes = []
    for op in operators:
        struct = op.out_structure()
        leaves = jax.tree.leaves(struct)
        dtypes.extend(leaf.dtype for leaf in leaves)
    return str(jnp.result_type(*dtypes))


class BlockDiag(lx.AbstractLinearOperator):
    """Block-diagonal operator A = blockdiag(A_1, …, A_D), one block per slice.

        blockdiag(A_1, …, A_D) [x_1; …; x_D] = [A_1 x_1; …; A_D x_D]

    Each block A_d acts on its own slice x_d of the input, so the matvec,
    transpose, solve, log-determinant and Cholesky factor all decompose block
    by block and never form the M × N matrix.

    Args:
        *operators: The diagonal blocks A_1, …, A_D, one or more lineax
            operators in block order.
        tags: Extra lineax tags for A; ``block_diagonal_tag`` is always
            added.

    Raises:
        ValueError: If no block is given.

    Examples:
        A 5 × 5 operator stored as a 2 × 2 and a 3 × 3 block:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.MatrixLinearOperator(3.0 * jnp.eye(3), psd)
        >>> D = gaussx.BlockDiag(A, B)
        >>> D.as_matrix().shape
        (5, 5)

        The log-determinant is the sum over the blocks:

        >>> logdet_blocks = gaussx.logdet(A) + gaussx.logdet(B)
        >>> bool(jnp.allclose(gaussx.logdet(D), logdet_blocks))
        True

        Primitives keep the structure:

        >>> type(gaussx.cholesky(D)).__name__
        'BlockDiag'
    """

    operators: tuple[lx.AbstractLinearOperator, ...]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        *operators: lx.AbstractLinearOperator,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if len(operators) == 0:
            raise ValueError("BlockDiag requires at least one operator.")
        self.operators = operators
        # N = ∑_d N_d
        self._in_size = sum(op.in_size() for op in operators)
        # M = ∑_d M_d
        self._out_size = sum(op.out_size() for op in operators)
        self._dtype = _resolve_dtype(*operators)

        self.tags = _to_frozenset(tags) | {block_diagonal_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        """Apply A to a vector, one block at a time.

            A x = [A_1 x_1; …; A_D x_D]

        Args:
            vector: Input x = [x_1; …; x_D], of length N.

        Returns:
            A x, of length M.
        """
        block_outputs = []
        offset = 0
        for block in self.operators:
            block_size = block.in_size()
            # x_d, the slice of x that A_d acts on; static offsets keep the
            # slice jit-compatible
            block_input = jax.lax.dynamic_slice(vector, (offset,), (block_size,))
            # A_d x_d
            block_outputs.append(block.mv(block_input))
            offset += block_size
        # [A_1 x_1; …; A_D x_D]
        return jnp.concatenate(block_outputs)

    def as_matrix(self) -> Float[Array, "m n"]:
        """Dense A = blockdiag(A_1, …, A_D), M × N.

        Returns:
            The materialised block-diagonal matrix.
        """
        block_matrices = [op.as_matrix() for op in self.operators]
        # blockdiag(A_1, …, A_D)
        return jax.scipy.linalg.block_diag(*block_matrices)

    def transpose(self) -> BlockDiag:
        """Aᵀ = blockdiag(A_1ᵀ, …, A_Dᵀ), still a `BlockDiag`.

        Returns:
            The transposed operator, with transposed tags.
        """
        return BlockDiag(
            *(op.T for op in self.operators),
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (M,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def _to_frozenset(x: object | frozenset[object]) -> frozenset[object]:
    """Normalise a single tag or a collection of tags to a frozenset.

    Args:
        x: One lineax tag, or an iterable or frozenset of tags.

    Returns:
        The tags as a frozenset.
    """
    if isinstance(x, frozenset):
        return x
    if isinstance(x, Iterable):
        return frozenset(x)
    return frozenset([x])
