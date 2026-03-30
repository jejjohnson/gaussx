"""Structured log-determinant with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import LowRankUpdate


def logdet(operator: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Compute log |det(A)| with structural dispatch.

    Args:
        operator: The linear operator A.

    Returns:
        Scalar log |det(A)|.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _logdet_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _logdet_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _logdet_kronecker(operator)
    if isinstance(operator, LowRankUpdate):
        return _logdet_low_rank(operator)
    return _logdet_dense(operator)


def _logdet_diagonal(operator: lx.DiagonalLinearOperator) -> jnp.ndarray:
    diag = lx.diagonal(operator)
    return jnp.sum(jnp.log(jnp.abs(diag)))


def _logdet_block_diag(operator: BlockDiag) -> jnp.ndarray:
    return ft.reduce(jnp.add, (logdet(op) for op in operator.operators))


def _logdet_kronecker(operator: Kronecker) -> jnp.ndarray:
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


def _logdet_low_rank(operator: LowRankUpdate) -> jnp.ndarray:
    """Matrix determinant lemma: det(L + U D V^T) = det(C) det(D) det(L).

    where C = D^{-1} + V^T L^{-1} U is the k x k capacitance matrix.
    So: logdet = logdet(L) + logdet(C) + sum(log|d_i|).
    """
    from gaussx._primitives._solve import solve

    U, d, V = operator.U, operator.d, operator.V

    # logdet(L)
    ld_base = logdet(operator.base)

    # L^{-1} U  (n x k)
    Linv_U = jnp.stack(
        [solve(operator.base, U[:, j]) for j in range(U.shape[1])], axis=1
    )

    # Capacitance matrix C = D^{-1} + V^T L^{-1} U
    C = jnp.diag(1.0 / d) + V.T @ Linv_U

    # logdet(C)
    _, ld_C = jnp.linalg.slogdet(C)

    # sum(log|d_i|)
    ld_d = jnp.sum(jnp.log(jnp.abs(d)))

    return ld_base + ld_C + ld_d


def _logdet_dense(operator: lx.AbstractLinearOperator) -> jnp.ndarray:
    mat = operator.as_matrix()
    _, ld = jnp.linalg.slogdet(mat)
    return ld
