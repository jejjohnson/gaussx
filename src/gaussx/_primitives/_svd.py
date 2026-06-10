"""Structured SVD with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import lineax as lx
import matfree.decomp
import matfree.eig
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker


def svd(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """Compute the singular value decomposition ``A = U diag(s) V^T``.

    When ``rank`` is given, computes a partial (truncated) SVD via
    matfree's Golub-Kahan bidiagonalization — no matrix materialization.

    Args:
        operator: A linear operator.
        rank: Number of singular values to compute. If ``None``,
            computes the full SVD (requires materialization).
        key: PRNG key for the initial random vector when using
            partial SVD. If ``None``, uses ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(U, s, Vt)`` where U has shape ``(M, K)``,
        s has shape ``(K,)``, and Vt has shape ``(K, N)``.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _svd_diagonal(operator)
    if rank is not None:
        return _svd_partial(operator, rank, key)
    if isinstance(operator, Kronecker):
        return _svd_kronecker(operator)
    if isinstance(operator, BlockDiag):
        return _svd_block_diag(operator)
    if isinstance(operator, lx.TaggedLinearOperator):
        return svd(operator.operator)
    return _svd_dense(operator)


def _svd_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> tuple[Float[Array, "n n"], Float[Array, " n"], Float[Array, "n n"]]:
    d = lx.diagonal(operator)
    n = d.shape[0]
    s = jnp.abs(d)
    signs = jnp.where(d >= 0, 1.0, -1.0)
    U = jnp.diag(signs)
    Vt = jnp.eye(n, dtype=d.dtype)
    return U, s, Vt


def _svd_partial(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """Partial SVD via matfree bidiagonalization."""
    if key is None:
        key = jr.PRNGKey(0)

    rank = min(rank, operator.in_size(), operator.out_size())
    n = operator.in_size()
    v0 = jr.normal(key, (n,))

    bidiag = matfree.decomp.bidiag(rank)
    svd_fn = matfree.eig.svd_partial(bidiag)

    # matfree returns U: (k, m), s: (k,), Vt: (k, n)
    U, s, Vt = svd_fn(operator.mv, v0)
    return U.T, s, Vt


def _sort_svd_descending(
    U: Float[Array, "m k"],
    s: Float[Array, " k"],
    Vt: Float[Array, "k n"],
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """Reorder an assembled SVD so singular values are descending."""
    order = jnp.argsort(-s)
    return U[:, order], s[order], Vt[order, :]


def _svd_kronecker(
    operator: Kronecker,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """SVD of a Kronecker product is the Kronecker product of factor SVDs.

    Only the (small) factors are decomposed; the full operator is never
    materialized as an input to a decomposition — the returned dense
    ``U``/``Vt`` are inherent to the SVD output format.
    """
    factor_svds = [svd(op) for op in operator.operators]
    U = ft.reduce(jnp.kron, (u for u, _, _ in factor_svds))
    s = ft.reduce(jnp.kron, (sv for _, sv, _ in factor_svds))
    Vt = ft.reduce(jnp.kron, (vt for _, _, vt in factor_svds))
    return _sort_svd_descending(U, s, Vt)


def _svd_block_diag(
    operator: BlockDiag,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """SVD of a block-diagonal operator from per-block SVDs."""
    block_svds = [svd(op) for op in operator.operators]
    U = jax.scipy.linalg.block_diag(*(u for u, _, _ in block_svds))
    s = jnp.concatenate([sv for _, sv, _ in block_svds])
    Vt = jax.scipy.linalg.block_diag(*(vt for _, _, vt in block_svds))
    return _sort_svd_descending(U, s, Vt)


def _svd_dense(
    operator: lx.AbstractLinearOperator,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    mat = operator.as_matrix()
    U, s, Vt = jnp.linalg.svd(mat, full_matrices=False)  # type: ignore[arg-type]
    return U, s, Vt
