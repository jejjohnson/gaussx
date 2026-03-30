"""Structured SVD with dispatch on operator type."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import matfree.decomp
import matfree.eig


def svd(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    key: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    return _svd_dense(operator)


def _svd_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    key: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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


def _svd_dense(
    operator: lx.AbstractLinearOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mat = operator.as_matrix()
    U, s, Vt = jnp.linalg.svd(mat, full_matrices=False)  # type: ignore[arg-type]
    return U, s, Vt
