"""Tests for svd primitive."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._primitives._svd import svd
from gaussx._testing import tree_allclose


def test_svd_diagonal(getkey):
    d = jnp.array([3.0, -1.0, 2.0])
    op = lx.DiagonalLinearOperator(d)
    U, s, Vt = svd(op)
    # s should be abs(d)
    assert tree_allclose(s, jnp.abs(d))
    # Reconstruct
    reconstructed = U @ jnp.diag(s) @ Vt
    assert tree_allclose(reconstructed, jnp.diag(d), rtol=1e-5)


def test_svd_dense(getkey):
    mat = jr.normal(getkey(), (4, 4)) + 2 * jnp.eye(4)
    op = lx.MatrixLinearOperator(mat)
    U, s, Vt = svd(op)
    reconstructed = U @ jnp.diag(s) @ Vt
    assert tree_allclose(reconstructed, mat, rtol=1e-4)


def test_svd_rectangular(getkey):
    mat = jr.normal(getkey(), (3, 5))
    op = lx.MatrixLinearOperator(mat)
    U, s, Vt = svd(op)
    reconstructed = U @ jnp.diag(s) @ Vt
    assert tree_allclose(reconstructed, mat, rtol=1e-4)
