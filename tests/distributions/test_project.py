"""Tests for project sugar operation."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import lineax as lx

from gaussx import project
from gaussx._testing import random_pd_matrix, tree_allclose


def test_project_dense(getkey):
    """project(K_XZ, chol(K_ZZ)) should equal K_XZ @ K_ZZ^{-1}."""
    M = 4
    B = 6
    K_ZZ = random_pd_matrix(getkey(), M)
    K_XZ = jr.normal(getkey(), (B, M))

    L = jax.scipy.linalg.cholesky(K_ZZ, lower=True)
    L_op = lx.MatrixLinearOperator(L, lx.lower_triangular_tag)

    result = project(K_XZ, L_op)
    expected = K_XZ @ jnp.linalg.inv(K_ZZ)

    assert tree_allclose(result, expected, rtol=1e-4)


def test_project_diagonal(getkey):
    """project with diagonal K_ZZ."""
    M = 3
    B = 5
    d = jnp.abs(jr.normal(getkey(), (M,))) + 0.5
    K_XZ = jr.normal(getkey(), (B, M))

    L_d = jnp.sqrt(d)
    L_op = lx.MatrixLinearOperator(jnp.diag(L_d), lx.lower_triangular_tag)

    result = project(K_XZ, L_op)
    expected = K_XZ / d[None, :]

    assert tree_allclose(result, expected, rtol=1e-4)
