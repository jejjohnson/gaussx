"""Tests for project sugar operation."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import lineax as lx
import pytest

from gaussx import project
from gaussx._testing import random_pd_matrix, tree_allclose


@pytest.mark.slow
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


def _project_inputs():
    K_ZZ = jnp.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.5]])
    K_XZ = jnp.array([[0.5, 0.1, 0.0], [0.2, 0.3, 0.4]])
    return K_ZZ, K_XZ


def test_project_accepts_an_untagged_cholesky_factor():
    """gh-347: a factor from jnp.linalg.cholesky carries no lineax tag."""
    K_ZZ, K_XZ = _project_inputs()
    L = jnp.linalg.cholesky(K_ZZ)
    expected = K_XZ @ jnp.linalg.inv(K_ZZ)
    untagged = project(K_XZ, lx.MatrixLinearOperator(L))
    tagged = project(K_XZ, lx.MatrixLinearOperator(L, lx.lower_triangular_tag))
    assert jnp.allclose(untagged, expected, atol=1e-12)
    assert jnp.allclose(untagged, tagged, atol=1e-12)
    jitted = jax.jit(lambda m: project(K_XZ, lx.MatrixLinearOperator(m)))(L)
    assert jnp.allclose(jitted, expected, atol=1e-12)


def test_project_rejects_a_mismatched_factor():
    _, K_XZ = _project_inputs()
    with pytest.raises(ValueError, match=r"\(3, 3\).*\(2, 2\)"):
        project(K_XZ, lx.MatrixLinearOperator(jnp.eye(2), lx.lower_triangular_tag))
