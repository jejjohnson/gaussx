"""Structured eigendecomposition with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import matfree.decomp
import matfree.eig
from jax.scipy.linalg import block_diag as _block_diag
from jaxtyping import Array

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker


def eig(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    key: jax.Array | None = None,
) -> tuple[Array, Array]:
    """Compute eigenvalues and eigenvectors.

    For symmetric operators returns real eigenvalues via ``eigh``.
    When ``rank`` is given, computes a partial eigendecomposition
    via matfree Lanczos (symmetric) — no matrix materialization.

    Args:
        operator: A square linear operator.
        rank: Number of eigenvalues to compute. If ``None``,
            computes the full eigendecomposition.
        key: PRNG key for the initial random vector when using
            partial eig. If ``None``, uses ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` where eigenvalues has
        shape ``(K,)`` and eigenvectors has shape ``(N, K)``.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _eig_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _eig_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _eig_kronecker(operator)
    if rank is not None:
        return _eig_partial(operator, rank, key)
    return _eig_dense(operator)


def eigvals(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    key: jax.Array | None = None,
) -> Array:
    """Compute eigenvalues only.

    When ``rank`` is given, returns the top-k eigenvalues via
    matfree Lanczos without matrix materialization.

    Args:
        operator: A square linear operator.
        rank: Number of eigenvalues to compute.
        key: PRNG key for partial eigendecomposition.

    Returns:
        Eigenvalues array of shape ``(K,)``.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        return jnp.concatenate([eigvals(op) for op in operator.operators])
    if isinstance(operator, Kronecker):
        return _eigvals_kronecker(operator)
    if rank is not None:
        vals, _ = _eig_partial(operator, rank, key)
        return vals
    return _eigvals_dense(operator)


def _eig_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> tuple[Array, Array]:
    d = lx.diagonal(operator)
    n = d.shape[0]
    return d, jnp.eye(n, dtype=d.dtype)


def _eig_block_diag(
    operator: BlockDiag,
) -> tuple[Array, Array]:
    vals_list = []
    vecs_list = []
    for op in operator.operators:
        v, V = eig(op)
        vals_list.append(v)
        vecs_list.append(V)
    vals = jnp.concatenate(vals_list)
    vecs = _block_diag(*vecs_list)
    return vals, vecs


def _eig_kronecker(
    operator: Kronecker,
) -> tuple[Array, Array]:
    """eig(A kron B) = (kron(eigvals), kron(eigvecs))."""
    factor_eigs = [eig(op) for op in operator.operators]
    vals = ft.reduce(jnp.kron, (v for v, _ in factor_eigs))
    vecs = ft.reduce(jnp.kron, (V for _, V in factor_eigs))
    return vals, vecs


def _eigvals_kronecker(operator: Kronecker) -> Array:
    """eigvals(A kron B) = kron(eigvals(A), eigvals(B))."""
    return ft.reduce(jnp.kron, (eigvals(op) for op in operator.operators))


def _eig_partial(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
) -> tuple[Array, Array]:
    """Partial eigendecomposition via matfree Lanczos."""
    if key is None:
        key = jr.PRNGKey(0)

    n = operator.in_size()
    rank = min(rank, n)
    v0 = jr.normal(key, (n,))

    tridiag = matfree.decomp.tridiag_sym(rank, reortho="full")
    eigh_fn = matfree.eig.eigh_partial(tridiag)

    # matfree returns vals: (k,), vecs: (k, n)
    vals, vecs = eigh_fn(operator.mv, v0)
    return vals, vecs.T


def _eig_dense(
    operator: lx.AbstractLinearOperator,
) -> tuple[Array, Array]:
    mat = operator.as_matrix()
    if lx.is_symmetric(operator):
        return jnp.linalg.eigh(mat)
    return jnp.linalg.eig(mat)


def _eigvals_dense(operator: lx.AbstractLinearOperator) -> Array:
    mat = operator.as_matrix()
    if lx.is_symmetric(operator):
        return jnp.linalg.eigvalsh(mat)
    return jnp.linalg.eigvals(mat)
