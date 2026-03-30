"""Tests for schur_complement and conditional_variance sugar operations."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._sugar import conditional_variance, schur_complement
from gaussx._testing import random_pd_matrix, tree_allclose


def test_schur_complement_dense(getkey):
    """Schur complement should match K_XX - K_XZ K_ZZ^{-1} K_ZX."""
    N, M = 6, 3
    K_XX_mat = random_pd_matrix(getkey(), N)
    K_ZZ_mat = random_pd_matrix(getkey(), M)
    K_XZ = jr.normal(getkey(), (N, M)) * 0.3

    K_XX = lx.MatrixLinearOperator(K_XX_mat, lx.symmetric_tag)
    K_ZZ = lx.MatrixLinearOperator(K_ZZ_mat, lx.positive_semidefinite_tag)

    result = schur_complement(K_XX, K_XZ, K_ZZ)
    expected = K_XX_mat - K_XZ @ jnp.linalg.solve(K_ZZ_mat, K_XZ.T)

    assert tree_allclose(result.as_matrix(), expected, rtol=1e-4)


def test_schur_complement_returns_low_rank_update(getkey):
    """schur_complement should return a LowRankUpdate operator."""
    from gaussx._operators import LowRankUpdate

    N, M = 5, 2
    K_XX = lx.MatrixLinearOperator(random_pd_matrix(getkey(), N))
    K_ZZ = lx.MatrixLinearOperator(random_pd_matrix(getkey(), M))
    K_XZ = jr.normal(getkey(), (N, M)) * 0.3

    result = schur_complement(K_XX, K_XZ, K_ZZ)
    assert isinstance(result, LowRankUpdate)


def test_schur_complement_diagonal_kzz(getkey):
    """Schur complement with diagonal K_ZZ."""
    N, M = 5, 3
    K_XX_mat = random_pd_matrix(getkey(), N)
    d = jnp.abs(jr.normal(getkey(), (M,))) + 1.0
    K_XZ = jr.normal(getkey(), (N, M)) * 0.3

    K_XX = lx.MatrixLinearOperator(K_XX_mat, lx.symmetric_tag)
    K_ZZ = lx.DiagonalLinearOperator(d)

    result = schur_complement(K_XX, K_XZ, K_ZZ)
    expected = K_XX_mat - K_XZ @ jnp.diag(1.0 / d) @ K_XZ.T

    assert tree_allclose(result.as_matrix(), expected, rtol=1e-4)


def test_conditional_variance_basic(getkey):
    """conditional_variance should match manual computation."""
    N, M = 8, 3
    K_XX_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
    A_X = jr.normal(getkey(), (N, M))
    S_mat = random_pd_matrix(getkey(), M)
    S_u = lx.MatrixLinearOperator(S_mat, lx.symmetric_tag)

    result = conditional_variance(K_XX_diag, A_X, S_u)
    expected = K_XX_diag + jnp.diag(A_X @ S_mat @ A_X.T)

    assert tree_allclose(result, expected, rtol=1e-5)


def test_conditional_variance_identity_su(getkey):
    """With S_u = I, should give K_XX_diag + diag(A A^T)."""
    N, M = 6, 2
    K_XX_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
    A_X = jr.normal(getkey(), (N, M))
    S_u = lx.MatrixLinearOperator(jnp.eye(M))

    result = conditional_variance(K_XX_diag, A_X, S_u)
    expected = K_XX_diag + jnp.sum(A_X**2, axis=1)

    assert tree_allclose(result, expected, rtol=1e-5)
