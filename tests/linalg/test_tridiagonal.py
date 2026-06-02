"""Tests for the tridiagonal solver."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from gaussx import solve_tridiagonal, solve_tridiagonal_batched
from gaussx._testing import tree_allclose


def _dense_tridiagonal(lower, diag, upper):
    return jnp.diag(diag) + jnp.diag(lower, -1) + jnp.diag(upper, 1)


def test_solve_tridiagonal_matches_dense(getkey):
    n = 8
    diag = jnp.abs(jr.normal(getkey(), (n,))) + 4.0  # diagonally dominant
    lower = jr.normal(getkey(), (n - 1,))
    upper = jr.normal(getkey(), (n - 1,))
    rhs = jr.normal(getkey(), (n,))

    x = solve_tridiagonal(lower, diag, upper, rhs)
    mat = _dense_tridiagonal(lower, diag, upper)
    assert tree_allclose(x, jnp.linalg.solve(mat, rhs), rtol=1e-5)


def test_solve_tridiagonal_batched(getkey):
    batch, n = 5, 6
    diag = jnp.abs(jr.normal(getkey(), (batch, n))) + 4.0
    lower = jr.normal(getkey(), (batch, n - 1))
    upper = jr.normal(getkey(), (batch, n - 1))
    rhs = jr.normal(getkey(), (batch, n))

    x = solve_tridiagonal_batched(lower, diag, upper, rhs)
    assert x.shape == (batch, n)
    for k in range(batch):
        mat = _dense_tridiagonal(lower[k], diag[k], upper[k])
        assert tree_allclose(x[k], jnp.linalg.solve(mat, rhs[k]), rtol=1e-5)
