"""Structured lazy Cholesky factor L, A = L Lᵀ, dispatched on the type of A.

A is a square symmetric positive-definite N × N operator and L its lower
Cholesky factor, returned as an operator. A Kronecker product is
A = A_1 ⊗ … ⊗ A_D with square factors A_d, d = 1, …, D, and a
block-diagonal operator is blockdiag(A_1, …, A_D). A symmetric
block-tridiagonal operator has diagonal blocks A_bb and sub-diagonal blocks
A_b,b−1, b = 1, …, B, and its factor is block-bidiagonal with diagonal blocks
L_bb and sub-diagonal blocks L_b,b−1. A sparse operator factors with a
fill-reducing permutation P, P A Pᵀ = L Lᵀ.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag, LowerBlockTriDiag
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._sparse import SparseOperator
from gaussx._operators._sum_kronecker import SumOfKroneckers
from gaussx._primitives._scale import scaled_root
from gaussx._sparse._factor import sparse_cholesky


if TYPE_CHECKING:
    from gaussx._sparse._factor import SparseCholeskyFactor


class DenseFallbackWarning(UserWarning):
    """Warning emitted when a structured primitive materialises an operator.

    Every structured primitive warns with it before it calls ``as_matrix()``
    on an operator that has structure, and the message names the
    matrix-free alternative. Filter it like any warning, or turn it into an
    error in tests that promise a fast path.

    Examples:
        Densifying a non-FFT diagonalised operator warns:

        >>> import warnings
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> eigenvalues = jnp.array([1.0, 2.0])
        >>> factorization = gaussx.EigenDecomposition(
        ...     eigenvalues, jnp.eye(2), jnp.eye(2)
        ... )
        >>> A = gaussx.DiagonalizedOperator.from_eigen_factorization(factorization)
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     _ = gaussx.cholesky(A)
        >>> caught[0].category.__name__
        'DenseFallbackWarning'
    """


# Frames inside gaussx are skipped, so the warning lands on the caller's line
# however deep the (recursive) dispatch that emitted it.
_GAUSSX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def warn_dense_fallback(message: str) -> None:
    """Emit a `DenseFallbackWarning` attributed to the caller's code.

    Args:
        message: What is densified and the matrix-free alternative.
    """
    warnings.warn(
        message,
        DenseFallbackWarning,
        stacklevel=2,
        skip_file_prefixes=(_GAUSSX_DIR + os.sep,),
    )


@overload
def cholesky(operator: SparseOperator) -> SparseCholeskyFactor: ...


@overload
def cholesky(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator: ...


def cholesky(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator | SparseCholeskyFactor:
    """Lower Cholesky factor L of an operator, structured where A is.

        A = L Lᵀ,   L lower triangular

    The result is an operator, not a raw array, and keeps A's structure: a
    Kronecker product factors per factor, a block-diagonal operator per
    block, and a symmetric block-tridiagonal operator gives a
    block-bidiagonal `LowerBlockTriDiag`. A `SparseOperator` returns a
    `SparseCholeskyFactor` of the fill-reducing permutation, P A Pᵀ = L Lᵀ
    (`gaussx.sparse_cholesky` on the cached symbolic analysis of its
    pattern), with ``solve``, ``logdet``, ``solve_lower_transpose`` and
    ``diag_inv``.

    A scalar multiple c A or A / c (c > 0) factors A and scales the factor
    by √c, keeping its structured type. A concretely negative multiple such
    as −A is not PSD and raises.

    Algorithm:
        1. Take the first structural rule that matches A: the identity is
           its own factor, a diagonal takes square roots, block-diagonal and
           Kronecker operators factor per block or factor, a symmetric
           block-tridiagonal operator runs the block-banded recurrence, and
           a sparse operator runs a sparse Cholesky.
        2. A `SumOfKroneckers` or `DiagonalizedOperator` warns with
           `DenseFallbackWarning` and factors densely; tags are peeled, and
           a scalar multiple factors its base and scales the factor by √c.
        3. Otherwise materialise A and run a dense Cholesky.

    Args:
        operator: The symmetric positive-definite operator A.

    Returns:
        The lower-triangular factor L as an operator, or a
        `SparseCholeskyFactor` for a `SparseOperator`.

    Raises:
        ValueError: If ``operator`` is a concretely negative multiple of a
            non-NSD operator, such as ``-A``.

    Examples:
        A Kronecker product factors per factor:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A_1 = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> A_2 = lx.MatrixLinearOperator(jnp.array([[3.0, 0.5], [0.5, 1.0]]), psd)
        >>> A = gaussx.Kronecker(A_1, A_2)
        >>> L = gaussx.cholesky(A)  # chol(A_1) ⊗ chol(A_2)
        >>> type(L).__name__
        'Kronecker'
        >>> bool(jnp.allclose((L @ L.T).as_matrix(), A.as_matrix(), atol=1e-5))
        True

        A scalar multiple keeps the structured type, scaled by √c:

        >>> type(gaussx.cholesky(4.0 * A)).__name__
        'Kronecker'

        A negative multiple is not PSD:

        >>> gaussx.cholesky(-A)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: cholesky requires a positive semi-definite operator
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        # chol(I) = I
        return operator
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _cholesky_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _cholesky_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _cholesky_kronecker(operator)
    if isinstance(operator, BlockTriDiag):
        return _cholesky_block_tridiag(operator)
    if isinstance(operator, SumOfKroneckers):
        return _cholesky_sum_kronecker(operator)
    if isinstance(operator, SparseOperator):
        return _cholesky_sparse(operator)
    if isinstance(operator, DiagonalizedOperator):
        warn_dense_fallback(
            "cholesky(DiagonalizedOperator) materialises the operator and runs "
            "a dense O(n^3) Cholesky. gaussx.sqrt(A) returns a symmetric root S "
            "with S S^T = A in O(n log n), which is enough for sampling."
        )
        return _cholesky_dense(operator)
    if isinstance(operator, lx.TaggedLinearOperator):
        return cholesky(operator.operator)
    if isinstance(
        operator, lx.MulLinearOperator | lx.DivLinearOperator | lx.NegLinearOperator
    ):
        # chol(c A) = √c chol(A)
        return scaled_root(operator, cholesky, _cholesky_dense, "cholesky")
    return _cholesky_dense(operator)


def _cholesky_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> lx.DiagonalLinearOperator:
    """Cholesky factor of a diagonal operator diag(a), entry by entry.

        chol(diag(a)) = diag(√a)

    Args:
        operator: Diagonal operator diag(a) with a ≥ 0.

    Returns:
        The diagonal operator diag(√a).
    """
    diagonal = lx.diagonal(operator)
    # √a_i
    root_diagonal = jnp.sqrt(diagonal)
    return lx.DiagonalLinearOperator(root_diagonal)


def _cholesky_block_diag(operator: BlockDiag) -> BlockDiag:
    """Cholesky factor of a block-diagonal operator, block by block.

        chol(blockdiag(A_1, …, A_D)) = blockdiag(chol(A_1), …, chol(A_D))

    Args:
        operator: Block-diagonal operator with PSD blocks A_d.

    Returns:
        The block-diagonal factor.
    """
    # chol(A_d), per block
    return BlockDiag(*(cholesky(block) for block in operator.operators))


def _cholesky_kronecker(operator: Kronecker) -> Kronecker:
    """Cholesky factor of a Kronecker product, factor by factor.

        chol(A_1 ⊗ … ⊗ A_D) = chol(A_1) ⊗ … ⊗ chol(A_D)

    Args:
        operator: Kronecker product of PSD factors A_d.

    Returns:
        The Kronecker factor.
    """
    # chol(A_d), per factor
    return Kronecker(*(cholesky(factor) for factor in operator.operators))


def _cholesky_block_tridiag(operator: BlockTriDiag) -> LowerBlockTriDiag:
    """Block-banded Cholesky factor of a symmetric block-tridiagonal operator.

        L_b,b−1 = A_b,b−1 L_b−1,b−1⁻ᵀ,
        L_bb = chol(A_bb − L_b,b−1 L_b,b−1ᵀ)

    The recurrence runs over the B blocks of size d in O(B d³), starting from
    L_11 = chol(A_11).

    Algorithm:
        1. Factor the first diagonal block, L_11 = chol(A_11).
        2. Scan b = 2, …, B: solve for the sub-diagonal block L_b,b−1 from
           the previous diagonal factor, then factor the Schur complement
           A_bb − L_b,b−1 L_b,b−1ᵀ.
        3. Stack the diagonal factors into a `LowerBlockTriDiag`.

    Args:
        operator: Symmetric positive-definite block-tridiagonal operator.

    Returns:
        The block-bidiagonal lower factor L.
    """
    num_blocks = operator._num_blocks
    if num_blocks == 1:
        # The scan body below indexes the empty sub-diagonal at trace time.
        # L_11 = chol(A_11)
        first_factor = jnp.linalg.cholesky(operator.diagonal[0])
        # jnp, not einx: einx cannot resolve an empty block (d = 0).
        return LowerBlockTriDiag(first_factor[None], operator.sub_diagonal)

    def scan_fn(carry, block_index):
        previous_factor = carry
        sub_block = operator.sub_diagonal[block_index - 1]
        # jnp, not einx: einx cannot resolve an empty block (d = 0).
        # L_b,b−1ᵀ = L_b−1,b−1⁻¹ A_b,b−1ᵀ
        sub_factor_transpose = jax.scipy.linalg.solve_triangular(
            previous_factor, sub_block.T, lower=True
        )
        # L_b,b−1 = A_b,b−1 L_b−1,b−1⁻ᵀ
        sub_factor = sub_factor_transpose.T
        # L_b,b−1 L_b,b−1ᵀ
        sub_outer = sub_factor @ sub_factor.T
        # A_bb − L_b,b−1 L_b,b−1ᵀ
        schur_complement = operator.diagonal[block_index] - sub_outer
        # L_bb = chol(A_bb − L_b,b−1 L_b,b−1ᵀ)
        diagonal_factor = jnp.linalg.cholesky(schur_complement)
        return diagonal_factor, (diagonal_factor, sub_factor)

    # L_11 = chol(A_11)
    first_factor = jnp.linalg.cholesky(operator.diagonal[0])
    _, (later_factors, sub_factors) = jax.lax.scan(
        scan_fn, first_factor, jnp.arange(1, num_blocks)
    )
    # jnp, not einx: einx cannot resolve an empty block (d = 0).
    # [L_11; L_22; …; L_BB]
    diagonal_factors = jnp.concatenate([first_factor[None], later_factors], axis=0)
    return LowerBlockTriDiag(diagonal_factors, sub_factors)


def _cholesky_sum_kronecker(operator: SumOfKroneckers) -> lx.MatrixLinearOperator:
    """Dense Cholesky factor of a sum of Kronecker products, with a warning.

    Args:
        operator: Sum of Kronecker products A.

    Returns:
        The dense lower factor L.
    """
    warn_dense_fallback(
        "cholesky(SumOfKroneckers) materialises the dense covariance. "
        "Use sum_of_kroneckers_sample(...) or sqrt(SumOfKroneckers) for matrix-free "
        "Lanczos sampling."
    )
    return _cholesky_dense(operator)


def _cholesky_sparse(operator: SparseOperator) -> SparseCholeskyFactor:
    """Sparse Cholesky P A Pᵀ = L Lᵀ on the cached symbolic analysis.

    Args:
        operator: Sparse PSD operator A.

    Returns:
        The `SparseCholeskyFactor` (RCM ordering, JAX backend).
    """
    return sparse_cholesky(operator)


def _cholesky_dense(
    operator: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Dense Cholesky factor of a materialised operator.

    Args:
        operator: The PSD operator A, materialised here.

    Returns:
        The lower factor L as a lower-triangular-tagged matrix operator.
    """
    matrix = operator.as_matrix()
    # A = L Lᵀ
    lower_factor = jax.scipy.linalg.cholesky(matrix, lower=True)
    return lx.MatrixLinearOperator(lower_factor, lx.lower_triangular_tag)
