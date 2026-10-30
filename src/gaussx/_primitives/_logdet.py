"""Structured log-determinant log ∣det A∣, dispatched on the operator type of A.

A is a square N × N operator and L a lower Cholesky factor, A = L Lᵀ. A
Kronecker product is A = A_1 ⊗ … ⊗ A_D with square N_d × N_d factors A_d,
d = 1, …, D, and a Kronecker sum is A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2, whose
eigenvalues are the pairwise sums γ_1,i + γ_2,j of its factors' eigenvalues.
A low-rank update is A = A₀ + U D Vᵀ, with base A₀, N × k factors U and V
and D = diag(d), and C = I + D Vᵀ A₀⁻¹ U is its k × k capacitance scaled by
D. The pseudo-determinant det₊ A is the product of the non-zero eigenvalues
of a symmetric positive-semidefinite A, and B is an N × N_null basis of its
kernel.
"""

from __future__ import annotations

import functools as ft
from typing import TYPE_CHECKING, Literal

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._diagonalised import DiagonalizedOperator, as_diagonalized
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, _eigh_factor
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    orthonormal_scaled_identity,
)
from gaussx._operators._sparse import _PLAN_CACHE_SIZE, SparseOperator, SparsityPattern
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._operators._sum_kronecker import (
    SumOfKroneckers,
    _sum_of_kroneckers_eigen,
)
from gaussx._primitives._cholesky import cholesky, warn_dense_fallback
from gaussx._primitives._solve import _low_rank_capacitance


if TYPE_CHECKING:
    from gaussx._strategies._base import AbstractLogdetStrategy


def cholesky_logdet(L: Float[Array, "N N"]) -> Float[Array, ""]:
    """Log-determinant log det A from a Cholesky factor L of A = L Lᵀ.

        log det A = 2 ∑_i log L_ii

    Args:
        L: Lower-triangular Cholesky factor L, N × N.

    Returns:
        The scalar log det A.

    Examples:
        A diagonal factor gives the sum of the logs of A's diagonal:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> L = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        >>> round(float(gaussx.cholesky_logdet(L)), 4)  # log 36
        3.5835

        It agrees with the dense log-determinant of L Lᵀ:

        >>> L = jnp.array([[2.0, 0.0], [1.0, 3.0]])
        >>> _, dense = jnp.linalg.slogdet(L @ L.T)
        >>> bool(jnp.allclose(gaussx.cholesky_logdet(L), dense, atol=1e-5))
        True
    """
    # L_ii
    diagonal = jnp.diag(L)
    # ∑_i log L_ii
    log_diagonal_sum = jnp.sum(jnp.log(diagonal))
    # log det A = 2 ∑_i log L_ii
    return 2.0 * log_diagonal_sum


def logdet(operator: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Log-determinant log ∣det A∣ with structural dispatch on the operator type.

        log ∣det A∣ = ∑_i log ∣γ_i∣,   γ_i the eigenvalues of A

    Algorithm:
        1. Take the first structural rule that matches A: a diagonal sums
           its logs, a Kronecker product scales its factors' log-determinants,
           a low-rank update applies the matrix determinant lemma, a Kronecker
           sum adds its factors' eigenvalues, a block-tridiagonal operator
           runs a banded Cholesky, and scalar, tagged and composed wrappers
           recurse into their parts.
        2. A large positive-semidefinite `SparseOperator` takes a stochastic
           SLQ estimate.
        3. Fall back to a dense Cholesky (PSD) or ``slogdet``.

    Args:
        operator: The square operator A.

    Returns:
        The scalar log ∣det A∣.

    Examples:
        A Kronecker product of diagonals, from its factors alone:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
        >>> A = gaussx.Kronecker(A_1, A_2)  # diag(4, 5, 8, 10)
        >>> round(float(gaussx.logdet(A)), 4)  # N_2 log ∣A_1∣ + N_1 log ∣A_2∣
        7.3778

        A dense positive-definite matrix takes the Cholesky fallback:

        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> round(float(gaussx.logdet(A)), 4)  # log 11
        2.3979

        A scalar multiple adds N log ∣c∣:

        >>> scaled = 2.0 * lx.DiagonalLinearOperator(jnp.ones(3))
        >>> round(float(gaussx.logdet(scaled)), 4)  # 3 log 2
        2.0794
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        # log det I = 0
        return jnp.zeros((), dtype=operator.in_structure().dtype)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _logdet_diagonal(operator)
    if isinstance(operator, DiagonalizedOperator):
        return _logdet_diagonalised(operator)
    if isinstance(operator, BlockDiag):
        return _logdet_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _logdet_kronecker(operator)
    if isinstance(operator, LowRankUpdate):
        if operator.rank == 0:
            # log ∣det A₀∣ when k = 0
            return logdet(operator.base)
        shift = orthonormal_scaled_identity(operator)
        if shift is not None:
            return _logdet_low_rank_orthonormal(operator, shift)
        return _logdet_low_rank(operator)
    if isinstance(operator, SumOfKroneckers):
        return _logdet_sum_of_kroneckers(operator)
    if isinstance(operator, KroneckerSum):
        return _logdet_kronecker_sum(operator)
    if isinstance(operator, SpectralFunction):
        return operator.logdet()
    if isinstance(operator, BlockTriDiag) and operator.symmetric:
        # Non-symmetric diagonal blocks take the dense slogdet (gh-344).
        return _logdet_block_tridiag(operator)
    if isinstance(operator, LowerBlockTriDiag | UpperBlockTriDiag):
        return _logdet_block_bidiagonal(operator)
    if isinstance(operator, SparseOperator):
        return _logdet_sparse(operator)
    if isinstance(operator, lx.TaggedLinearOperator):
        return logdet(operator.operator)
    if isinstance(operator, lx.MulLinearOperator):
        size = operator.out_size()
        # log ∣det(c A)∣ = N log ∣c∣ + log ∣det A∣
        return size * jnp.log(jnp.abs(operator.scalar)) + logdet(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        size = operator.out_size()
        # log ∣det(A / c)∣ = log ∣det A∣ − N log ∣c∣
        return logdet(operator.operator) - size * jnp.log(jnp.abs(operator.scalar))
    if isinstance(operator, lx.NegLinearOperator):
        # log ∣det(−A)∣ = log ∣det A∣, since ∣(−1)ᴺ∣ = 1
        return logdet(operator.operator)
    if isinstance(operator, lx.ComposedLinearOperator) and (
        operator.operator1.in_size() == operator.operator1.out_size()
        and operator.operator2.in_size() == operator.operator2.out_size()
    ):
        # log ∣det(A B)∣ = log ∣det A∣ + log ∣det B∣, both square
        return logdet(operator.operator1) + logdet(operator.operator2)
    if isinstance(operator, lx.AddLinearOperator):
        factorization = _sum_of_kroneckers_eigen(operator)
        if factorization is not None:
            return factorization.logdet()
    return _logdet_dense(operator)


def _has_structural_logdet(operator: lx.AbstractLinearOperator) -> bool:
    """Whether `logdet` (and `gaussx.solve`) take an exact structural path.

    Mirrors `logdet`'s dispatch: ``True`` when the operator, after unwrapping
    `lineax.TaggedLinearOperator` / ``MulLinearOperator`` /
    ``DivLinearOperator`` / ``NegLinearOperator`` (and composing square
    factors), is one whose log-determinant is computed without materialising
    it and without a stochastic estimate. A large `SparseOperator`'s default
    logdet is a stochastic SLQ estimate, and a `Toeplitz` or a non-reducible
    sum of Kronecker products is materialised, so those are ``False``.

    Kept beside `logdet` so the two lists cannot drift apart; `AutoSolver`
    and `inv_quad_logdet` route on it.

    Args:
        operator: The linear operator.

    Returns:
        Whether the structural path applies.
    """
    if isinstance(
        operator,
        lx.IdentityLinearOperator
        | lx.DiagonalLinearOperator
        | DiagonalizedOperator
        | BlockDiag
        | Kronecker
        | LowRankUpdate
        | KroneckerSum
        | SpectralFunction
        | LowerBlockTriDiag
        | UpperBlockTriDiag,
    ):
        return True
    if isinstance(operator, BlockTriDiag):
        return operator.symmetric
    if isinstance(operator, SumOfKroneckers | lx.AddLinearOperator):
        return _sum_of_kroneckers_eigen(operator) is not None
    if isinstance(
        operator,
        lx.TaggedLinearOperator
        | lx.MulLinearOperator
        | lx.DivLinearOperator
        | lx.NegLinearOperator,
    ):
        return _has_structural_logdet(operator.operator)
    if isinstance(operator, lx.ComposedLinearOperator):
        first, second = operator.operator1, operator.operator2
        return (
            first.in_size() == first.out_size()
            and second.in_size() == second.out_size()
            and _has_structural_logdet(first)
            and _has_structural_logdet(second)
        )
    return False


def _logdet_sparse(operator: SparseOperator) -> Float[Array, ""]:
    """`SLQLogdet` when large and PSD (`AutoSolver` threshold), dense otherwise.

    The SLQ estimate uses the strategy's default fixed key. The exact sparse
    Cholesky path is the `SparseCholeskySolver` strategy, passed explicitly.

    Args:
        operator: Sparse operator A.

    Returns:
        The scalar log ∣det A∣, estimated when A is large and PSD.
    """
    # lazy import, cycle: _strategies._auto -> _primitives._logdet
    from gaussx._strategies._auto import AutoSolver

    # lazy import, cycle: _strategies._slq_logdet -> _primitives._logdet
    from gaussx._strategies._slq_logdet import SLQLogdet

    if operator.in_size() > AutoSolver().size_threshold and (
        lx.is_positive_semidefinite(operator)
    ):
        return SLQLogdet().logdet(operator)
    return _logdet_dense(operator)


def _logdet_diagonal(operator: lx.DiagonalLinearOperator) -> Float[Array, ""]:
    """Log-determinant of a diagonal operator diag(a).

        log ∣det diag(a)∣ = ∑_i log ∣a_i∣

    Args:
        operator: Diagonal operator diag(a).

    Returns:
        The scalar log ∣det diag(a)∣.
    """
    diagonal = lx.diagonal(operator)
    # ∑_i log ∣a_i∣
    return jnp.sum(jnp.log(jnp.abs(diagonal)))


def _logdet_block_diag(operator: BlockDiag) -> Float[Array, ""]:
    """Log-determinant of a block-diagonal operator, block by block.

        log ∣det A∣ = ∑_d log ∣det A_d∣

    Args:
        operator: Block-diagonal operator with blocks A_d.

    Returns:
        The scalar log ∣det A∣.
    """
    # ∑_d log ∣det A_d∣
    return ft.reduce(jnp.add, (logdet(block) for block in operator.operators))


def _logdet_kronecker(operator: Kronecker) -> Float[Array, ""]:
    """Log-determinant of a Kronecker product from its factors.

        log ∣det(A_1 ⊗ … ⊗ A_D)∣ = ∑_d (N / N_d) log ∣det A_d∣

    For two factors, log ∣det(A_1 ⊗ A_2)∣ = N_2 log ∣det A_1∣
    + N_1 log ∣det A_2∣.

    Args:
        operator: Kronecker product A_1 ⊗ … ⊗ A_D.

    Returns:
        The scalar log ∣det A∣.
    """
    total_size = operator.out_size()
    result = jnp.array(0.0)
    for factor in operator.operators:
        factor_size = factor.out_size()
        # + (N / N_d) log ∣det A_d∣
        result = result + (total_size // factor_size) * logdet(factor)
    return result


def _logdet_low_rank(operator: LowRankUpdate) -> Float[Array, ""]:
    """Log-determinant of A₀ + U D Vᵀ by the matrix determinant lemma.

        det(A₀ + U D Vᵀ) = det A₀ det C,   C = I + D Vᵀ A₀⁻¹ U

    C is the k × k capacitance scaled by D, so a zero weight needs no
    log 0 (gh-307).

    Args:
        operator: Low-rank update A₀ + U D Vᵀ.

    Returns:
        The scalar log ∣det(A₀ + U D Vᵀ)∣.
    """
    # log ∣det A₀∣
    base_logdet = logdet(operator.base)
    # C = I + D Vᵀ A₀⁻¹ U
    _, capacitance = _low_rank_capacitance(operator, solver=None)
    # log ∣det C∣
    _, capacitance_logdet = jnp.linalg.slogdet(capacitance)
    # log ∣det A∣ = log ∣det A₀∣ + log ∣det C∣
    return base_logdet + capacitance_logdet


def _logdet_low_rank_orthonormal(
    operator: LowRankUpdate, c: Float[Array, ""]
) -> Float[Array, ""]:
    """Log-determinant of c I + U D Uᵀ in closed form for orthonormal U.

        log ∣det(c I + U D Uᵀ)∣ = (N − k) log ∣c∣ + ∑_i log ∣c + d_i∣,   Uᵀ U = I

    The eigenvalues are c + d_i on the span of U and c on its
    (N − k)-dimensional complement; O(k) (gh-333).

    Args:
        operator: Low-rank update c I + U D Uᵀ with orthonormal U.
        c: The scalar shift c of the base.

    Returns:
        The scalar log ∣det(c I + U D Uᵀ)∣.
    """
    size, rank = operator.in_size(), operator.rank
    # (N − k) log ∣c∣ + ∑_i log ∣c + d_i∣
    return (size - rank) * jnp.log(jnp.abs(c)) + jnp.sum(
        jnp.log(jnp.abs(c + operator.d))
    )


def _logdet_sum_of_kroneckers(operator: SumOfKroneckers) -> Float[Array, ""]:
    """Log-determinant of ∑_j A_j ⊗ B_j by simultaneous diagonalisation.

        log ∣det(A_1 ⊗ B_1 + A_2 ⊗ B_2)∣

    For two terms with one positive definite this is a sum of
    log ∣γ_A,i γ_B,j + 1∣ over the whitened eigenvalues plus the anchor's own
    scaled log-determinants: the same reduction `solve` uses, so the two
    agree by construction. Three or more terms keep the dense fallback;
    `SLQLogdet` estimates that case matrix-free.

    Args:
        operator: Sum of Kronecker products ∑_j A_j ⊗ B_j.

    Returns:
        The scalar log ∣det A∣.
    """
    factorization = _sum_of_kroneckers_eigen(operator)
    if factorization is None:
        warn_dense_fallback(
            "logdet(SumOfKroneckers) has no closed form here (three or more "
            "terms, or no positive-definite anchor) and materialises the "
            "operator; SLQLogdet estimates it matrix-free."
        )
        return _logdet_dense(operator)
    return factorization.logdet()


def _logdet_kronecker_sum(operator: KroneckerSum) -> Float[Array, ""]:
    """Log-determinant of a Kronecker sum from its factors' eigenvalues.

        log ∣det(A_1 ⊕ A_2)∣ = ∑_i,j log ∣γ_1,i + γ_2,j∣

    The eigenvalues of A_1 ⊕ A_2 are γ_1,i + γ_2,j for any square factors.
    Each factor is handled on its own: a symmetric one (tagged, or
    diagonal) uses the shared ``_eigh_factor`` helper, the same routine the
    Kronecker-sum solve path uses, O(N_d) for a diagonal. ``eigh`` reads
    only one triangle, so any other factor takes the general ``eigvals``
    instead (still O(N_d³) in that factor alone; gh-308), mirroring the
    symmetry guard in `solve`. The general eigensolver runs on CPU (LAPACK
    ``geev``) and on CUDA GPUs (cuSOLVER / MAGMA ``geev``).

    Args:
        operator: Kronecker sum A_1 ⊕ A_2.

    Returns:
        The scalar log ∣det(A_1 ⊕ A_2)∣.
    """
    diagonalised = as_diagonalized(operator)
    if diagonalised is not None:
        return _logdet_diagonalised(diagonalised)
    # γ_1
    eigenvalues_1 = _factor_eigvals(operator.A)
    # γ_2
    eigenvalues_2 = _factor_eigvals(operator.B)
    # γ_1,i + γ_2,j
    eigenvalue_sums = einx.add("a, b -> a b", eigenvalues_1, eigenvalues_2)
    # ∑_i,j log ∣γ_1,i + γ_2,j∣
    return jnp.sum(jnp.log(jnp.abs(eigenvalue_sums)))


def _factor_eigvals(operator: lx.AbstractLinearOperator) -> Array:
    """Eigenvalues γ_d of one Kronecker-sum factor: ``eigh`` iff symmetric.

    Args:
        operator: The factor A_d.

    Returns:
        Its eigenvalues γ_d, possibly complex for a non-symmetric factor.
    """
    if lx.is_symmetric(operator):
        return _eigh_factor(operator)[0]
    return jnp.linalg.eigvals(operator.as_matrix())


def _logdet_diagonalised(operator: DiagonalizedOperator) -> Float[Array, ""]:
    """Log-determinant of A = W⁻¹ diag(γ) W from its eigenvalues.

        log ∣det A∣ = ∑_i log ∣γ_i∣

    The ``slogdet`` convention: −∞ when A is singular.

    Args:
        operator: Diagonalised operator with eigenvalues γ.

    Returns:
        The scalar log ∣det A∣.
    """
    # ∑_i log ∣γ_i∣
    return jnp.sum(jnp.log(jnp.abs(operator.eigenvalues)))


def _logdet_block_tridiag(operator: BlockTriDiag) -> Float[Array, ""]:
    """Log-determinant of a symmetric block-tridiagonal A by banded Cholesky.

        log det A = 2 log det L,   A = L Lᵀ

    Args:
        operator: Symmetric positive-definite block-tridiagonal A.

    Returns:
        The scalar log det A.
    """
    # A = L Lᵀ, L lower block-bidiagonal
    factor = cholesky(operator)
    # log det A = 2 log det L
    return 2.0 * logdet(factor)


def _logdet_block_bidiagonal(
    operator: LowerBlockTriDiag | UpperBlockTriDiag,
) -> Float[Array, ""]:
    """Log-determinant of a block-bidiagonal operator with triangular blocks.

        log ∣det A∣ = ∑_k ∑_i log ∣[L_k]_ii∣

    Valid for both lower and upper variants: the determinant is the product
    of the diagonal blocks' determinants, and the blocks L_k are triangular
    (Cholesky factors), so each reduces to its diagonal.

    Args:
        operator: Block-bidiagonal operator with triangular diagonal blocks.

    Returns:
        The scalar log ∣det A∣.
    """
    # ∑_i log ∣[L_k]_ii∣, per block
    block_logdets = jax.vmap(lambda block: jnp.sum(jnp.log(jnp.abs(jnp.diag(block)))))(
        operator.diagonal
    )
    # ∑_k ∑_i log ∣[L_k]_ii∣
    return jnp.sum(block_logdets)


def _logdet_dense(operator: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Dense log-determinant: Cholesky for a PSD operator, else ``slogdet``.

        log det A = 2 ∑_i log L_ii (PSD),   log ∣det A∣ by LU otherwise

    Args:
        operator: The operator A, materialised here.

    Returns:
        The scalar log ∣det A∣, non-finite when A is singular.
    """
    matrix = operator.as_matrix()
    if _is_psd(operator):
        # A Cholesky is half an LU. A singular PSD matrix gives nan here (as
        # numpyro does), not slogdet's -inf: non-finite either way (gh-329).
        return _cholesky_logdet(jnp.linalg.cholesky(matrix))
    # log ∣det A∣ by LU
    _, log_abs_det = jnp.linalg.slogdet(matrix)
    return log_abs_det


def _is_psd(operator: lx.AbstractLinearOperator) -> bool:
    """``lx.is_positive_semidefinite``, False where lineax does not know.

    Args:
        operator: The operator to test.

    Returns:
        Whether A is known to be positive semi-definite.
    """
    try:
        return bool(lx.is_positive_semidefinite(operator))
    except NotImplementedError:
        return False


def _cholesky_logdet(factor: Float[Array, "N N"]) -> Float[Array, ""]:
    """Log-determinant log det(L Lᵀ) from a Cholesky factor L.

        log det(L Lᵀ) = 2 ∑_i log L_ii

    Args:
        factor: Lower Cholesky factor L.

    Returns:
        The scalar log det(L Lᵀ).
    """
    # 2 ∑_i log L_ii
    return 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))


def _dense_psd_matrix(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "N N"] | None:
    """The matrix of a PSD-tagged dense operator, looking through tags.

    Args:
        operator: The operator, possibly tagged.

    Returns:
        The dense matrix, or ``None`` for anything structured, so callers
        keep its dispatch.
    """
    inner = operator
    while isinstance(inner, lx.TaggedLinearOperator):
        inner = inner.operator
    if not isinstance(inner, lx.MatrixLinearOperator) or not _is_psd(operator):
        return None
    return inner.matrix


def pseudo_logdet(
    operator: lx.AbstractLinearOperator,
    *,
    null_space: Float[ArrayLike, "N c"] | Float[ArrayLike, " N"] | None = None,
    structure: Literal["laplacian"] | None = None,
    rcond: float | None = None,
    strategy: AbstractLogdetStrategy | None = None,
) -> Float[Array, ""]:
    """Log pseudo-determinant: the log of the product of the non-zero eigenvalues.

        log det₊ A = ∑_{i: γ_i > 0} log γ_i

    For a symmetric positive-semidefinite A. Intrinsic GMRFs (Besag / ICAR,
    RW1, RW2) have singular structure matrices R, and ½ log det₊ R is their
    normalising constant. Paths, cheapest first:

    - ``structure="laplacian"`` (a weighted graph Laplacian, e.g. Besag or
      RW1): the matrix-tree theorem. Every principal (N_c − 1)-minor of a
      connected component's Laplacian L_c equals its weighted spanning-tree
      count, and det₊ L_c is N_c times it, so
      log det₊ L = ∑_c (log N_c + log det L_c,minor). The minors are taken by
      replacing one node per component with a unit diagonal, which keeps
      the sparsity pattern, so it is **one** sparse Cholesky
      (`SparseCholeskySolver` by default; G4) for a `SparseOperator`, or one
      banded Cholesky for a `BlockTriDiag` with 1 × 1 blocks (a path, as
      from `rw1_structure`). Components are found from the sparsity pattern
      on the host (once per pattern), so every stored off-diagonal weight
      must be non-zero. Gradients are exact. A `KroneckerSum` (a grid
      Laplacian) takes the eigenvalue path below instead.
    - ``null_space`` given: det₊ A = det(A + B Bᵀ) for an orthonormal B
      spanning ker A (each zero eigenvalue becomes 1). Any basis B of the
      kernel works, since det(A + B Bᵀ) = det₊ A det(Bᵀ B) and the second
      factor is subtracted. A + B Bᵀ is a matvec-only lineax sum, so
      ``strategy=gaussx.SLQLogdet()`` estimates it matrix-free; the default
      is `gaussx.logdet` of it (dense). The matrix determinant lemma does not
      apply because A is singular.
    - A `KroneckerSum` (recursively, with dense, diagonal or
      `DiagonalizedOperator` factors) or a `DiagonalizedOperator`: all
      pairwise sums of the factor eigenvalues, with no factorisation of the
      full operator.
    - Anything else: dense ``eigvalsh``.

    The eigenvalue paths drop eigenvalues γ_i ≤ rcond · max_i γ_i.

    **Constants are constant.** log det₊(τ R) = rank(R) log τ + log det₊ R,
    so log det₊ R never depends on hyperparameters: compute it once, outside
    any θ loop, and keep it.

    Algorithm:
        1. With ``null_space``: log det(A + B Bᵀ) − log det(Bᵀ B).
        2. With ``structure="laplacian"``: one Cholesky of the grounded
           minor plus ∑_c log N_c.
        3. Otherwise: the spectrum γ of A, structured where possible, and
           ∑ log γ_i over the eigenvalues above the cut-off.

    Args:
        operator: Symmetric positive-semidefinite operator A.
        null_space: A basis B of ker A, shape ``(N, c)`` (or ``(N,)`` for
            ``c = 1``); for a connected graph Laplacian, 1/√N. kernellib's
            ``graph_null_space`` gives one column per connected component.
        structure: ``"laplacian"`` to use the matrix-tree theorem on a
            weighted graph Laplacian (`SparseOperator`, 1 × 1-block
            `BlockTriDiag` or `KroneckerSum`).
        rcond: Relative cut-off for the eigenvalue paths. Defaults to
            N · eps of the dtype, with N the largest dense eigenproblem
            solved (a factor's size for a `KroneckerSum`), as in
            ``numpy.linalg.matrix_rank``.
        strategy: Log-determinant strategy for the null-space path (of
            A + B Bᵀ) and the Laplacian path (of the minor). Not used by the
            eigenvalue paths.

    Returns:
        The scalar log det₊ A.

    Raises:
        ValueError: If both ``null_space`` and ``structure`` are given, if
            ``strategy`` is given for an eigenvalue path, if ``structure`` is
            unknown, or if ``null_space`` has the wrong number of rows.
        TypeError: If ``structure="laplacian"`` is given an operator it has
            no Laplacian path for.

    Examples:
        ICAR normalising constant on the path graph 0 − 1 − 2 − 3, where
        det₊ R is N times the single spanning tree, so 4:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import numpy as np
        >>> import gaussx
        >>> n = 4
        >>> R = gaussx.SparseOperator.from_coo(
        ...     np.r_[np.arange(n), np.arange(1, n)],
        ...     np.r_[np.arange(n), np.arange(n - 1)],
        ...     jnp.r_[jnp.array([1.0, 2.0, 2.0, 1.0]), -jnp.ones(n - 1)],
        ...     (n, n),
        ...     symmetric=True,
        ... )
        >>> log_pdet = gaussx.pseudo_logdet(R, structure="laplacian")
        >>> bool(jnp.allclose(log_pdet, jnp.log(4.0), atol=1e-5))
        True

        The same through its null space, the constant vector 1/√N:

        >>> ones = jnp.ones(n) / jnp.sqrt(n)
        >>> log_pdet = gaussx.pseudo_logdet(R, null_space=ones)
        >>> bool(jnp.allclose(log_pdet, jnp.log(4.0), atol=1e-5))
        True

        On a grid there is no factorisation, only the non-zero sums
        γ_1,i + γ_2,j of the factor eigenvalues:

        >>> L = gaussx.rw1_structure(5).as_matrix()
        >>> grid = gaussx.KroneckerSum(
        ...     lx.MatrixLinearOperator(L, lx.symmetric_tag),
        ...     lx.MatrixLinearOperator(L, lx.symmetric_tag),
        ... )
        >>> gaussx.pseudo_logdet(grid).shape
        ()
    """
    if structure not in (None, "laplacian"):
        raise ValueError(f"structure must be None or 'laplacian', got {structure!r}.")
    if null_space is not None and structure is not None:
        raise ValueError("Pass either null_space or structure, not both.")
    if isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    if null_space is not None:
        return _pseudo_logdet_null_space(operator, null_space, strategy)
    if structure == "laplacian":
        if isinstance(operator, SparseOperator):
            return _pseudo_logdet_laplacian_sparse(operator, strategy)
        if isinstance(operator, BlockTriDiag):
            return _pseudo_logdet_laplacian_banded(operator, strategy)
        if not isinstance(operator, KroneckerSum | DiagonalizedOperator):
            raise TypeError(
                "structure='laplacian' needs a SparseOperator, a BlockTriDiag "
                "with 1 x 1 blocks or a KroneckerSum, got "
                f"{type(operator).__name__}; pass null_space instead."
            )
    if strategy is not None:
        raise ValueError(
            "strategy is used by the null_space and structure='laplacian' "
            "paths only; the eigenvalue paths are exact."
        )
    # γ, and the largest dense eigenproblem solved
    eigenvalues, size = _spectrum(operator)
    if rcond is None:
        rcond = size * float(jnp.finfo(eigenvalues.dtype).eps)
    # γ_i > rcond · max_i γ_i
    keep = eigenvalues > rcond * jnp.max(eigenvalues)
    # log γ_i on the kept eigenvalues, log 1 = 0 elsewhere
    log_eigenvalues = jnp.log(jnp.where(keep, eigenvalues, 1))
    # log det₊ A = ∑_{i kept} log γ_i
    return jnp.sum(jnp.where(keep, log_eigenvalues, 0))


def _spectrum(operator: lx.AbstractLinearOperator) -> tuple[Float[Array, " n"], int]:
    """Real eigenvalues γ of a symmetric operator, and the largest ``eigh`` size.

    Args:
        operator: Symmetric operator A.

    Returns:
        Tuple ``(eigenvalues, size)``: the flat spectrum γ and the size of
        the largest dense eigenproblem solved for it.
    """
    if isinstance(operator, lx.TaggedLinearOperator):
        return _spectrum(operator.operator)
    if isinstance(operator, KroneckerSum):
        eigenvalues_1, size_1 = _spectrum(operator.A)
        eigenvalues_2, size_2 = _spectrum(operator.B)
        # γ_1,i + γ_2,j
        eigenvalue_sums = einx.add("a, b -> a b", eigenvalues_1, eigenvalues_2)
        flat_sums = rearrange(eigenvalue_sums, "a b -> (a b)")
        return flat_sums, max(size_1, size_2)
    if isinstance(operator, DiagonalizedOperator):
        return jnp.real(operator.eigenvalues_flat()), 1
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator), 1
    return jnp.linalg.eigvalsh(operator.as_matrix()), operator.in_size()


def _pseudo_logdet_null_space(
    operator: lx.AbstractLinearOperator,
    null_space: Float[ArrayLike, "N c"] | Float[ArrayLike, " N"],
    strategy: AbstractLogdetStrategy | None,
) -> Float[Array, ""]:
    """Log pseudo-determinant from a kernel basis B.

        log det₊ A = log det(A + B Bᵀ) − log det(Bᵀ B)

    Args:
        operator: Symmetric positive-semidefinite A.
        null_space: A basis B of ker A, ``(N, c)`` or ``(N,)``.
        strategy: Log-determinant strategy for A + B Bᵀ, or ``None`` for
            `logdet`.

    Returns:
        The scalar log det₊ A.

    Raises:
        ValueError: If B does not have N rows.
    """
    dtype = operator.in_structure().dtype
    basis = jnp.asarray(null_space, dtype=dtype)
    if basis.ndim == 1:
        basis = rearrange(basis, "n -> n 1")
    if basis.ndim != 2 or basis.shape[0] != operator.in_size():
        raise ValueError(
            f"null_space must have shape ({operator.in_size()}, c), got {basis.shape}."
        )

    def project(vector: Float[Array, " N"]) -> Float[Array, " N"]:
        # Bᵀ v
        coefficients = einsum(basis, vector, "n c, n -> c")
        # B Bᵀ v
        return einsum(basis, coefficients, "n c, c -> n")

    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    low_rank = lx.FunctionLinearOperator(project, operator.in_structure(), tags)
    # A + B Bᵀ
    shifted = lx.TaggedLinearOperator(lx.AddLinearOperator(operator, low_rank), tags)
    # log det(A + B Bᵀ)
    shifted_logdet = logdet(shifted) if strategy is None else strategy.logdet(shifted)
    # Bᵀ B
    gram = einsum(basis, basis, "n c, n d -> c d")
    # log det(Bᵀ B)
    _, gram_logdet = jnp.linalg.slogdet(gram)
    # log det₊ A = log det(A + B Bᵀ) − log det(Bᵀ B)
    return shifted_logdet - gram_logdet


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _laplacian_plan(
    pattern: SparsityPattern,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per stored value: keep it, or the minor's unit diagonal; and ∑_c log N_c.

    Connected components by hooking and pointer jumping over the pattern
    (each node ends up pointing at the smallest index in its component, its
    root); the roots are the nodes removed from each component. Runs on the
    host with NumPy, once per pattern.

    Args:
        pattern: Sparsity pattern of the Laplacian L.

    Returns:
        Tuple ``(keep, unit, log_sizes)``: which stored values the minor
        keeps, which stored values become its unit diagonal, and
        ∑_c log N_c over the connected components c.
    """
    size = pattern.shape[0]
    rows, cols = pattern.rows, pattern.cols
    parent = np.arange(size)
    while True:
        # hook every node to the smaller root across each stored edge
        hooked = parent.copy()
        np.minimum.at(hooked, parent[rows], parent[cols])
        np.minimum.at(hooked, parent[cols], parent[rows])
        # pointer jumping until every node points at a root
        while not np.array_equal(hooked[hooked], hooked):
            hooked = hooked[hooked]
        if np.array_equal(hooked, parent):
            break
        parent = hooked
    is_root = parent == np.arange(size)
    touches_root = is_root[rows] | is_root[cols]
    unit = touches_root & (rows == cols)
    # ∑_c log N_c
    component_sizes = np.bincount(parent, minlength=size)[is_root]
    log_sizes = float(np.sum(np.log(component_sizes)))
    return ~touches_root, unit, log_sizes


def _pseudo_logdet_laplacian_sparse(
    operator: SparseOperator, strategy: AbstractLogdetStrategy | None
) -> Float[Array, ""]:
    """Matrix-tree theorem: ∑_c log N_c + log det L_minor by one sparse Cholesky.

        log det₊ L = ∑_c (log N_c + log det L_c,minor)

    Args:
        operator: Sparse weighted graph Laplacian L.
        strategy: Log-determinant strategy for the minor, or ``None`` for
            `SparseCholeskySolver`.

    Returns:
        The scalar log det₊ L.

    Raises:
        ValueError: If L is not square.
    """
    # lazy import, cycle: _strategies._sparse_cholesky -> _strategies ->
    #   _strategies._auto -> _primitives._logdet
    from gaussx._strategies._sparse_cholesky import SparseCholeskySolver

    if operator.pattern.shape[0] != operator.pattern.shape[1]:
        raise ValueError(f"A Laplacian must be square, got {operator.pattern.shape}.")
    keep, unit, log_sizes = _laplacian_plan(operator.pattern)
    values = operator.values
    # L_minor: one node per component replaced by a unit diagonal
    minor_values = jnp.where(
        jnp.asarray(keep), values, jnp.asarray(unit, dtype=values.dtype)
    )
    minor = SparseOperator(
        minor_values,
        operator.pattern,
        tags=operator.tags | {lx.positive_semidefinite_tag},
    )
    solver = SparseCholeskySolver() if strategy is None else strategy
    # log det L_minor + ∑_c log N_c
    return solver.logdet(minor) + jnp.asarray(log_sizes, dtype=values.dtype)


def _pseudo_logdet_laplacian_banded(
    operator: BlockTriDiag, strategy: AbstractLogdetStrategy | None
) -> Float[Array, ""]:
    """Matrix-tree theorem on a path: log N + log det of the grounded minor.

        log det₊ L = log N + log det L_minor

    The minor is the band with the last node replaced by a unit diagonal, so
    it is one banded Cholesky.

    Args:
        operator: Path-graph Laplacian L as a `BlockTriDiag` with 1 × 1
            blocks.
        strategy: Log-determinant strategy for the minor, or ``None`` for
            `logdet`.

    Returns:
        The scalar log det₊ L.

    Raises:
        ValueError: If the blocks are not 1 × 1.
    """
    num_nodes, block_size, _ = operator.diagonal.shape
    if block_size != 1:
        raise ValueError(
            "structure='laplacian' on a BlockTriDiag needs 1 x 1 blocks (a "
            f"weighted path graph), got {block_size} x {block_size}; use a "
            "SparseOperator or null_space."
        )
    # ground the last node: unit diagonal, no coupling
    diagonal = operator.diagonal.at[-1].set(1)
    sub_diagonal = operator.sub_diagonal.at[num_nodes - 2 :].set(0)
    minor = BlockTriDiag(
        diagonal,
        sub_diagonal,
        symmetric=operator.symmetric,
        tags=operator.tags | {lx.positive_semidefinite_tag},
    )
    # log det L_minor
    minor_logdet = logdet(minor) if strategy is None else strategy.logdet(minor)
    # log det₊ L = log det L_minor + log N
    return minor_logdet + jnp.log(jnp.asarray(num_nodes, dtype=diagonal.dtype))
