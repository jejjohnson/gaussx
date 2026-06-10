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
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, _eigh_factor
from gaussx._operators._low_rank_update import LowRankUpdate


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
    if isinstance(operator, BlockDiag):
        return _logdet_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _logdet_kronecker(operator)
    if isinstance(operator, LowRankUpdate):
        return _logdet_low_rank(operator)
    if isinstance(operator, KroneckerSum):
        return _logdet_kronecker_sum(operator)
    if isinstance(operator, BlockTriDiag):
        return _logdet_block_tridiag(operator)
    if isinstance(operator, LowerBlockTriDiag | UpperBlockTriDiag):
        return _logdet_block_bidiagonal(operator)
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
    """Matrix determinant lemma: det(L + U D V^T) = det(C) det(D) det(L).

    where C = D^{-1} + V^T L^{-1} U is the k x k capacitance matrix.
    So: logdet = logdet(L) + logdet(C) + sum(log|d_i|).
    """
    from gaussx._primitives._solve import _low_rank_capacitance

    ld_base = logdet(operator.base)
    _, C = _low_rank_capacitance(operator, solver=None)
    _, ld_C = jnp.linalg.slogdet(C)
    ld_d = jnp.sum(jnp.log(jnp.abs(operator.d)))
    return ld_base + ld_C + ld_d


def _logdet_kronecker_sum(operator: KroneckerSum) -> Float[Array, ""]:
    """logdet(A (+) B) = sum(log(lambda_A_i + lambda_B_j)).

    Factor eigenvalues come from the shared ``_eigh_factor`` helper
    (structural shortcut for diagonal factors, ``eigh`` otherwise) —
    the same routine the KroneckerSum solve and eigendecomposition
    paths use, so the symmetry assumption is identical everywhere.
    """
    evals_a, _ = _eigh_factor(operator.A)
    evals_b, _ = _eigh_factor(operator.B)
    eig_mat = evals_a[None, :] + evals_b[:, None]
    return jnp.sum(jnp.log(jnp.abs(eig_mat)))


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
