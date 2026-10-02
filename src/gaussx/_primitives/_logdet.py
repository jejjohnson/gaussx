"""Structured log-determinant with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._diagonalised import DiagonalisedOperator, as_diagonalised
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, _eigh_factor
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sparse import SparseOperator
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._operators._sum_kronecker import (
    SumOfKroneckers,
    _sum_of_kroneckers_eigen,
)


def cholesky_logdet(L: Float[Array, "N N"]) -> Float[Array, ""]:
    """Compute log|A| from Cholesky factor L where A = L Lᵀ.

    Args:
        L: Lower-triangular Cholesky factor, shape ``(N, N)``.

    Returns:
        Scalar log-determinant.
    """
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def logdet(operator: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Compute log |det(A)| with structural dispatch.

    Args:
        operator: The linear operator A.

    Returns:
        Scalar log |det(A)|.
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        return jnp.array(0.0)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _logdet_diagonal(operator)
    if isinstance(operator, DiagonalisedOperator):
        return _logdet_diagonalised(operator)
    if isinstance(operator, BlockDiag):
        return _logdet_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _logdet_kronecker(operator)
    if isinstance(operator, LowRankUpdate):
        if operator.rank == 0:
            return logdet(operator.base)
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
        n = operator.out_size()
        return n * jnp.log(jnp.abs(operator.scalar)) + logdet(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        n = operator.out_size()
        return logdet(operator.operator) - n * jnp.log(jnp.abs(operator.scalar))
    if isinstance(operator, lx.NegLinearOperator):
        # log|det(-A)| = log|det(A)| since |(-1)^n| = 1.
        return logdet(operator.operator)
    if isinstance(operator, lx.ComposedLinearOperator) and (
        operator.operator1.in_size() == operator.operator1.out_size()
        and operator.operator2.in_size() == operator.operator2.out_size()
    ):
        return logdet(operator.operator1) + logdet(operator.operator2)
    if isinstance(operator, lx.AddLinearOperator):
        factorization = _sum_of_kroneckers_eigen(operator)
        if factorization is not None:
            return factorization.logdet()
    return _logdet_dense(operator)


def _logdet_sparse(operator: SparseOperator) -> Float[Array, ""]:
    """`SLQLogdet` when large and PSD (`AutoSolver` threshold), dense otherwise.

    The SLQ estimate uses the strategy's default fixed key. The exact sparse
    Cholesky path is the `SparseCholeskySolver` strategy, passed explicitly.
    """
    from gaussx._strategies._auto import AutoSolver
    from gaussx._strategies._slq_logdet import SLQLogdet

    if operator.in_size() > AutoSolver().size_threshold and (
        lx.is_positive_semidefinite(operator)
    ):
        return SLQLogdet().logdet(operator)
    return _logdet_dense(operator)


def _logdet_diagonal(operator: lx.DiagonalLinearOperator) -> Float[Array, ""]:
    diag = lx.diagonal(operator)
    return jnp.sum(jnp.log(jnp.abs(diag)))


def _logdet_block_diag(operator: BlockDiag) -> Float[Array, ""]:
    return ft.reduce(jnp.add, (logdet(op) for op in operator.operators))


def _logdet_kronecker(operator: Kronecker) -> Float[Array, ""]:
    """logdet(A1 kron A2 kron ... kron Ak).

    For two factors: logdet(A kron B) = n_B * logdet(A) + n_A * logdet(B).
    Generalizes to k factors.
    """
    total_size = operator.out_size()
    result = jnp.array(0.0)
    for op in operator.operators:
        n_i = op.out_size()
        # This factor's logdet is scaled by total_size / n_i
        result = result + (total_size // n_i) * logdet(op)
    return result


def _logdet_low_rank(operator: LowRankUpdate) -> Float[Array, ""]:
    """Matrix determinant lemma: det(L + U D V^T) = det(L) det(K).

    where K = I + D V^T L^{-1} U is the k x k capacitance scaled by D, so
    a zero weight needs no log(0) (gh-307).
    """
    from gaussx._primitives._solve import _low_rank_capacitance

    ld_base = logdet(operator.base)
    _, K = _low_rank_capacitance(operator, solver=None)
    _, ld_K = jnp.linalg.slogdet(K)
    return ld_base + ld_K


def _logdet_sum_of_kroneckers(operator: SumOfKroneckers) -> Float[Array, ""]:
    r"""``logdet(Σ_k A_k ⊗ B_k)`` from the simultaneous diagonalization.

    For two terms with one positive definite this is
    ``Σ_ij log|λ_a[i] λ_b[j] + 1|`` plus the anchor's own scaled logdets —
    the same reduction `solve` uses, so the two agree by construction.
    Three or more terms keep the dense fallback; `SLQLogdet` estimates that
    case matrix-free.
    """
    factorization = _sum_of_kroneckers_eigen(operator)
    if factorization is None:
        return _logdet_dense(operator)
    return factorization.logdet()


def _logdet_kronecker_sum(operator: KroneckerSum) -> Float[Array, ""]:
    """logdet(A (+) B) = sum(log(lambda_A_i + lambda_B_j)).

    Factor eigenvalues come from the shared ``_eigh_factor`` helper
    (structural shortcut for diagonal factors, ``eigh`` otherwise) —
    the same routine the KroneckerSum solve and eigendecomposition
    paths use, so the symmetry assumption is identical everywhere.
    """
    diagonalised = as_diagonalised(operator)
    if diagonalised is not None:
        return _logdet_diagonalised(diagonalised)
    evals_a, _ = _eigh_factor(operator.A)
    evals_b, _ = _eigh_factor(operator.B)
    eig_mat = evals_a[None, :] + evals_b[:, None]
    return jnp.sum(jnp.log(jnp.abs(eig_mat)))


def _logdet_diagonalised(operator: DiagonalisedOperator) -> Float[Array, ""]:
    """``log|det A| = Σ log|λ|`` (``slogdet`` convention; ``−inf`` if singular)."""
    return jnp.sum(jnp.log(jnp.abs(operator.eigenvalues)))


def _logdet_block_tridiag(operator: BlockTriDiag) -> Float[Array, ""]:
    """logdet via banded Cholesky: logdet(A) = 2 * logdet(L)."""
    from gaussx._primitives._cholesky import cholesky

    L = cholesky(operator)
    return 2.0 * logdet(L)


def _logdet_block_bidiagonal(
    operator: LowerBlockTriDiag | UpperBlockTriDiag,
) -> Float[Array, ""]:
    """logdet of a block-bidiagonal operator with triangular diagonal blocks.

    Valid for both lower and upper variants: the determinant is the
    product of the diagonal blocks' determinants, and the blocks are
    triangular (Cholesky factors), so each reduces to its diagonal.
    """
    return jnp.sum(
        jax.vmap(lambda L: jnp.sum(jnp.log(jnp.abs(jnp.diag(L)))))(operator.diagonal)
    )


def _logdet_dense(operator: lx.AbstractLinearOperator) -> Float[Array, ""]:
    mat = operator.as_matrix()
    _, ld = jnp.linalg.slogdet(mat)
    return ld
