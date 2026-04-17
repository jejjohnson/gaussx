"""Tests for woodbury_solve sugar operation."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import woodbury_solve
from gaussx._testing import tree_allclose


def test_woodbury_solve_basic(getkey):
    """woodbury_solve should match dense solve of (diag + U D U^T)."""
    N, k = 6, 2
    d_base = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
    U = jr.normal(getkey(), (N, k)) * 0.3
    D = jnp.abs(jr.normal(getkey(), (k,))) + 0.1
    b = jr.normal(getkey(), (N,))

    base = lx.DiagonalLinearOperator(d_base)
    result = woodbury_solve(base, U, D, b)

    # Build dense and solve directly
    A_dense = jnp.diag(d_base) + U @ jnp.diag(D) @ U.T
    expected = jnp.linalg.solve(A_dense, b)

    assert tree_allclose(result, expected, rtol=1e-4)


def test_woodbury_solve_identity_base(getkey):
    """Woodbury with identity base."""
    N, k = 5, 2
    U = jr.normal(getkey(), (N, k)) * 0.3
    D = jnp.abs(jr.normal(getkey(), (k,))) + 0.1
    b = jr.normal(getkey(), (N,))

    base = lx.DiagonalLinearOperator(jnp.ones(N))
    result = woodbury_solve(base, U, D, b)

    A_dense = jnp.eye(N) + U @ jnp.diag(D) @ U.T
    expected = jnp.linalg.solve(A_dense, b)

    assert tree_allclose(result, expected, rtol=1e-4)


def test_woodbury_solve_rank_one(getkey):
    """Rank-1 Woodbury: (diag + d * u u^T)^{-1} b."""
    N = 4
    d_base = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
    u = jr.normal(getkey(), (N, 1)) * 0.3
    d = jnp.array([2.0])
    b = jr.normal(getkey(), (N,))

    base = lx.DiagonalLinearOperator(d_base)
    result = woodbury_solve(base, u, d, b)

    A_dense = jnp.diag(d_base) + d[0] * (u @ u.T)
    expected = jnp.linalg.solve(A_dense, b)

    assert tree_allclose(result, expected, rtol=1e-4)
