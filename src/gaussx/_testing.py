"""Test utilities for gaussx.

Provides helper functions for generating random structured operators
and comparing results. Intended for use in the gaussx test suite and
other internal tests; not part of the public, stable gaussx API.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def tree_allclose(
    x, y, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool | jnp.ndarray:
    """PyTree-aware approximate equality check.

    Wraps ``eqx.tree_equal`` with tolerance support. Matches the
    pattern used in the lineax test suite.
    """
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Random operator generators
# ---------------------------------------------------------------------------


def random_pd_matrix(key: jax.Array, n: int, *, dtype=jnp.float64) -> jnp.ndarray:
    """Generate a random positive-definite n x n matrix."""
    A = jr.normal(key, (n, n), dtype=dtype)
    return A @ A.T + 0.1 * jnp.eye(n, dtype=dtype)


def random_pd_operator(
    key: jax.Array, n: int, *, dtype=jnp.float64
) -> lx.MatrixLinearOperator:
    """Generate a random PSD MatrixLinearOperator."""
    mat = random_pd_matrix(key, n, dtype=dtype)
    return lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)


def random_kronecker_pd(
    key: jax.Array,
    sizes: tuple[int, ...],
    *,
    dtype=jnp.float64,
) -> Kronecker:
    """Generate a Kronecker product of random PSD matrices."""
    keys = jr.split(key, len(sizes))
    ops = tuple(
        random_pd_operator(k, n, dtype=dtype) for k, n in zip(keys, sizes, strict=True)
    )
    return Kronecker(*ops)


def random_block_diag_pd(
    key: jax.Array,
    sizes: tuple[int, ...],
    *,
    dtype=jnp.float64,
) -> BlockDiag:
    """Generate a BlockDiag of random PSD matrices."""
    keys = jr.split(key, len(sizes))
    ops = tuple(
        random_pd_operator(k, n, dtype=dtype) for k, n in zip(keys, sizes, strict=True)
    )
    return BlockDiag(*ops)


def random_low_rank_update(
    key: jax.Array,
    n: int,
    rank: int,
    *,
    dtype=jnp.float64,
) -> LowRankUpdate:
    """Generate a random LowRankUpdate with positive diagonal base."""
    k1, k2, k3 = jr.split(key, 3)
    d = jnp.abs(jr.normal(k1, (n,), dtype=dtype)) + 0.5
    U = jr.normal(k2, (n, rank), dtype=dtype) * 0.3
    diag_vals = jnp.abs(jr.normal(k3, (rank,), dtype=dtype)) + 0.1
    base = lx.DiagonalLinearOperator(d)
    return LowRankUpdate(base, U, diag_vals)


# ---------------------------------------------------------------------------
# Dense reference computations
# ---------------------------------------------------------------------------


def dense_solve(op: lx.AbstractLinearOperator, v: jnp.ndarray) -> jnp.ndarray:
    """Solve via dense materialization (reference implementation)."""
    return jnp.linalg.solve(op.as_matrix(), v)


def dense_logdet(op: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Log-determinant via dense materialization."""
    return jnp.linalg.slogdet(op.as_matrix())[1]


def dense_inv(op: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Inverse via dense materialization."""
    return jnp.linalg.inv(op.as_matrix())


def dense_diag(op: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Diagonal via dense materialization."""
    return jnp.diag(op.as_matrix())


def dense_trace(op: lx.AbstractLinearOperator) -> jnp.ndarray:
    """Trace via dense materialization."""
    return jnp.trace(op.as_matrix())
