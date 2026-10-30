"""Structured sub-matrix A[I, J], dispatched on the operator type of A.

A is an N × N operator, I = (i_1, …, i_R) a list of row indices and
J = (j_1, …, j_C) a list of column indices (repeats allowed, negative entries
counted from the end), and A[I, J] is the dense R × C block with entries
A[I, J]_rc = A_{i_r j_c}. A Kronecker product is A = A_1 ⊗ … ⊗ A_D with
factors A_d of size N_d,row × N_d,col, d = 1, …, D, and a block-diagonal
operator is blockdiag(A_1, …, A_B). A low-rank update is A = A₀ + U D Vᵀ, with
base A₀, N × k factors U and V and D = diag(d).
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import LowRankUpdate


def submatrix(
    operator: lx.AbstractLinearOperator,
    row_idx: Int[Array, " R"],
    col_idx: Int[Array, " C"],
) -> Float[Array, "R C"]:
    """Sub-matrix A[I, J] of an operator, without forming the full matrix.

        A[I, J]_rc = A_{i_r j_c},   r = 1, …, R,   c = 1, …, C

    A structured A gives its block from its parts, so only R × C intermediates
    are formed, never the N × N matrix; the conditional Gaussian uses it to
    pull Σ_AA, Σ_AB and Σ_BB out of a joint covariance.

    Algorithm:
        1. Map negative indices in I and J to their non-negative positions.
        2. Take the first structural rule that matches A: a diagonal keeps
           d_i on matching pairs i_r = j_c, a block-diagonal operator gathers
           from its stacked blocks and zeroes pairs in different blocks, and
           a Kronecker product multiplies its factors' blocks entrywise.
        3. An empty I or J gives an empty block. Otherwise a square low-rank
           update adds U[I] D V[J]ᵀ to A₀[I, J], the identity marks matching
           pairs, and tagged, scaled, negated and summed operators recurse
           into their parts.
        4. Otherwise materialise A and index it.

    Args:
        operator: The N × N operator A.
        row_idx: Row indices I, negative entries counted from the end.
        col_idx: Column indices J, negative entries counted from the end.

    Returns:
        The dense block A[I, J].

    Examples:
        A Kronecker product gives its block from its factors' blocks:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
        >>> A = gaussx.Kronecker(A_1, A_2)  # A_1 ⊗ A_2, never densified
        >>> block = gaussx.submatrix(A, jnp.array([0, 3]), jnp.array([1, 3]))
        >>> block.tolist()
        [[1.0, 0.0], [0.0, 6.0]]

        A low-rank update adds U[I] D V[J]ᵀ to the base block; negative
        indices count from the end:

        >>> base = lx.DiagonalLinearOperator(jnp.array([1.0, 1.0, 1.0]))
        >>> U = jnp.array([[1.0], [2.0], [0.0]])
        >>> A = gaussx.LowRankUpdate(base, U)  # I + U Uᵀ
        >>> gaussx.submatrix(A, jnp.array([-2]), jnp.array([0, 1])).tolist()
        [[2.0, 5.0]]

        An empty index list gives an empty block:

        >>> gaussx.submatrix(A, jnp.array([], dtype=int), jnp.array([0, 1])).shape
        (0, 2)
    """
    size = operator.in_size()
    # i_r ← i_r + N where i_r < 0
    row_idx = _normalize_indices(row_idx, size)
    # j_c ← j_c + N where j_c < 0
    col_idx = _normalize_indices(col_idx, size)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _submatrix_diagonal(operator, row_idx, col_idx)
    if isinstance(operator, BlockDiag):
        return _submatrix_block_diag(operator, row_idx, col_idx)
    if isinstance(operator, Kronecker):
        return _submatrix_kronecker(operator, row_idx, col_idx)
    dtype = _matrix_dtype(operator)
    if row_idx.shape[0] == 0 or col_idx.shape[0] == 0:
        # einx rejects zero-length axes; an empty block needs no structure.
        return jnp.zeros((row_idx.shape[0], col_idx.shape[0]), dtype=dtype)
    if isinstance(operator, LowRankUpdate) and _is_square(operator.base):
        return _submatrix_low_rank(operator, row_idx, col_idx)
    if isinstance(operator, lx.IdentityLinearOperator) and _is_square(operator):
        # I[I, J]_rc = 1[i_r = j_c]
        match = einx.equal("row, col -> row col", row_idx, col_idx)
        return match.astype(dtype)
    if isinstance(operator, lx.TaggedLinearOperator):
        return submatrix(operator.operator, row_idx, col_idx)
    if isinstance(operator, lx.MulLinearOperator):
        inner_block = submatrix(operator.operator, row_idx, col_idx)
        # (c A)[I, J] = c A[I, J]
        return operator.scalar * inner_block
    if isinstance(operator, lx.DivLinearOperator):
        inner_block = submatrix(operator.operator, row_idx, col_idx)
        # (A / c)[I, J] = A[I, J] / c
        return inner_block / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        inner_block = submatrix(operator.operator, row_idx, col_idx)
        # (−A)[I, J] = −A[I, J]
        return -inner_block
    if isinstance(operator, lx.AddLinearOperator):
        first_block = submatrix(operator.operator1, row_idx, col_idx)
        second_block = submatrix(operator.operator2, row_idx, col_idx)
        # (A + B)[I, J] = A[I, J] + B[I, J]
        return first_block + second_block
    dense = operator.as_matrix()
    # A[I, J], dense
    return dense[jnp.ix_(row_idx, col_idx)]


def _matrix_dtype(operator: lx.AbstractLinearOperator):
    """The dtype ``as_matrix`` would have, also for PyTree structures.

    Args:
        operator: The operator A.

    Returns:
        The result dtype of A's input and output structures.
    """
    leaves = jax.tree_util.tree_leaves(
        (operator.in_structure(), operator.out_structure())
    )
    return jnp.result_type(*(leaf.dtype for leaf in leaves))


def _is_square(operator: lx.AbstractLinearOperator) -> bool:
    """Whether A has as many rows as columns.

    Args:
        operator: The operator A.

    Returns:
        ``True`` when A is square.
    """
    return operator.in_size() == operator.out_size()


def _submatrix_low_rank(
    operator: LowRankUpdate,
    row_idx: Int[Array, " R"],
    col_idx: Int[Array, " C"],
) -> Float[Array, "R C"]:
    """Sub-matrix of a low-rank update A₀ + U D Vᵀ (gh-379).

        (A₀ + U D Vᵀ)[I, J] = A₀[I, J] + U[I] D V[J]ᵀ

    Costs O(R C k) on top of the base's own block.

    Args:
        operator: Square low-rank update A₀ + U D Vᵀ.
        row_idx: Non-negative row indices I.
        col_idx: Non-negative column indices J.

    Returns:
        The dense block A[I, J].
    """
    base_block = submatrix(operator.base, row_idx, col_idx)
    if operator.rank == 0:
        # einx rejects a zero-length axis.
        return base_block
    # U[I] D
    weighted_rows = operator.U[row_idx] * operator.d
    # V[J]
    selected_cols = operator.V[col_idx]
    # U[I] D V[J]ᵀ
    update_block = einsum(weighted_rows, selected_cols, "row rank, col rank -> row col")
    # A₀[I, J] + U[I] D V[J]ᵀ
    return base_block + update_block


def _normalize_indices(
    idx: Int[Array, " K"],
    n: int,
) -> Int[Array, " K"]:
    """Map negative indices to their non-negative positions, as NumPy does.

        i ← i + N   where i < 0

    Args:
        idx: Indices into an axis of length N.
        n: The axis length N.

    Returns:
        The indices, each in 0, …, N − 1 when it was in range.
    """
    idx = jnp.asarray(idx)
    # i ← i + N where i < 0
    return jnp.where(idx < 0, idx + n, idx)


def _submatrix_diagonal(
    operator: lx.DiagonalLinearOperator,
    row_idx: Int[Array, " R"],
    col_idx: Int[Array, " C"],
) -> Float[Array, "R C"]:
    """Sub-matrix of a diagonal operator: nonzero only on matching pairs.

        diag(d)[I, J]_rc = d_{i_r} 1[i_r = j_c]

    Args:
        operator: Diagonal operator diag(d).
        row_idx: Non-negative row indices I.
        col_idx: Non-negative column indices J.

    Returns:
        The dense block diag(d)[I, J].
    """
    diagonal = lx.diagonal(operator)
    # 1[i_r = j_c]
    # jnp, not einx: I or J may be empty here, and einx rejects a zero axis.
    match = row_idx[:, None] == col_idx[None, :]
    # d_{i_r} 1[i_r = j_c]
    # jnp, not einx: I may be empty here, and einx rejects a zero axis.
    return jnp.where(
        match, diagonal[row_idx][:, None], jnp.zeros((), dtype=diagonal.dtype)
    )


def _submatrix_kronecker(
    operator: Kronecker,
    row_idx: Int[Array, " R"],
    col_idx: Int[Array, " C"],
) -> Float[Array, "R C"]:
    """Sub-matrix of a Kronecker product as an entrywise product of factor blocks.

        (A_1 ⊗ … ⊗ A_D)[I, J] = A_1[I_1, J_1] ⊙ … ⊙ A_D[I_D, J_D]

    I_d and J_d are the mixed-radix digits of I and J in the factor sizes
    N_d,row and N_d,col. Each factor's R × C block comes from recursive
    dispatch, so only R × C intermediates are formed, never A.

    Args:
        operator: Kronecker product A_1 ⊗ … ⊗ A_D.
        row_idx: Non-negative row indices I.
        col_idx: Non-negative column indices J.

    Returns:
        The dense block A[I, J].
    """
    result = jnp.ones(
        (row_idx.shape[0], col_idx.shape[0]), dtype=operator.in_structure().dtype
    )
    row_quotient, col_quotient = row_idx, col_idx
    # The last factor owns the fastest-varying digit.
    for factor in reversed(operator.operators):
        factor_rows = factor.out_size()
        factor_cols = factor.in_size()
        # I_d = I mod N_d,row
        row_digit = row_quotient % factor_rows
        # J_d = J mod N_d,col
        col_digit = col_quotient % factor_cols
        factor_block = submatrix(factor, row_digit, col_digit)
        # ⊙ A_d[I_d, J_d]
        result = result * factor_block
        # I ← ⌊I / N_d,row⌋
        row_quotient = row_quotient // factor_rows
        # J ← ⌊J / N_d,col⌋
        col_quotient = col_quotient // factor_cols
    return result


def _submatrix_block_diag(
    operator: BlockDiag,
    row_idx: Int[Array, " R"],
    col_idx: Int[Array, " C"],
) -> Float[Array, "R C"]:
    """Sub-matrix of a block-diagonal operator, without forming the full matrix.

        blockdiag(A_1, …, A_B)[I, J]_rc = A_b[i_r − o_b, j_c − o_b] 1[b(i_r) = b(j_c)]

    o_b is the offset of block b and b(i) the block that holds index i.
    Every block is materialised once and padded into one
    B × N_max × N_max stack to gather from, so only the blocks and the stack
    are formed, never the N × N matrix. Block sizes are static, so the
    bookkeeping is jit-compatible. When I and J stay within a few blocks,
    materialising only the touched blocks would be cheaper.

    Args:
        operator: Block-diagonal operator blockdiag(A_1, …, A_B).
        row_idx: Non-negative row indices I.
        col_idx: Non-negative column indices J.

    Returns:
        The dense block A[I, J].
    """
    block_sizes = [block.in_size() for block in operator.operators]
    offsets_list = [0]
    for block_size in block_sizes[:-1]:
        # o_b+1 = o_b + N_b
        offsets_list.append(offsets_list[-1] + block_size)
    offsets = jnp.asarray(offsets_list)
    # N_max, static: computed in Python, which keeps this jit-compatible.
    max_size = max(block_sizes)

    def _block_index(
        idx: Int[Array, " K"],
    ) -> tuple[Int[Array, " K"], Int[Array, " K"]]:
        # b(i) = max{b : o_b ≤ i}
        block_ids = jnp.searchsorted(offsets, idx, side="right") - 1
        # i − o_b(i)
        return block_ids, idx - offsets[block_ids]

    row_block, row_local = _block_index(row_idx)
    col_block, col_local = _block_index(col_idx)
    # 1[b(i_r) = b(j_c)]
    # jnp, not einx: I or J may be empty here, and einx rejects a zero axis.
    same_block = row_block[:, None] == col_block[None, :]

    # Materialise each block once; entries are pulled from the right block.
    block_matrices = [block.as_matrix() for block in operator.operators]
    # [A_1; …; A_B], each zero-padded to N_max × N_max
    padded = jnp.stack(
        [
            jnp.pad(
                matrix,
                ((0, max_size - matrix.shape[0]), (0, max_size - matrix.shape[1])),
            )
            for matrix in block_matrices
        ]
    )

    # A_b(i_r)[i_r − o_b, j_c − o_b], as if b(i_r) = b(j_c); masked below.
    # jnp, not einx: I or J may be empty here, and einx rejects a zero axis.
    gathered = padded[
        row_block[:, None],
        row_local[:, None],
        col_local[None, :],
    ]
    # A_b[i_r − o_b, j_c − o_b] 1[b(i_r) = b(j_c)]
    return jnp.where(same_block, gathered, jnp.zeros((), dtype=gathered.dtype))
