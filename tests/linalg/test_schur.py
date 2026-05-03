"""Tests for schur_complement and conditional_variance sugar operations."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import conditional_variance, schur_complement
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


def test_conditional_variance_no_su(getkey):
    """Without S_u, conditional_variance returns the Schur complement diagonal.

    Cross-validates against the full schur_complement matrix diagonal.
    """
    N, M = 8, 3
    K_XX_mat = random_pd_matrix(getkey(), N)
    K_ZZ_mat = random_pd_matrix(getkey(), M)
    K_XZ = jr.normal(getkey(), (N, M)) * 0.3
    # A_X = K_XZ K_ZZ^{-1}; compute via solve rather than inv for numerical
    # stability (jnp.linalg.solve(K_ZZ, K_XZ.T).T avoids forming K_ZZ^{-1}).
    A_X = jnp.linalg.solve(K_ZZ_mat, K_XZ.T).T
    K_XX_diag = jnp.diag(K_XX_mat)

    result = conditional_variance(K_XX_diag, K_XZ, A_X)

    # Reference: diagonal of the full Schur complement matrix
    K_XX_op = lx.MatrixLinearOperator(K_XX_mat, lx.positive_semidefinite_tag)
    K_ZZ_op = lx.MatrixLinearOperator(K_ZZ_mat, lx.positive_semidefinite_tag)
    schur_mat = schur_complement(K_XX_op, K_XZ, K_ZZ_op).as_matrix()
    expected = jnp.clip(jnp.diag(schur_mat), 0.0)

    assert tree_allclose(result, expected, rtol=1e-4)


def test_conditional_variance_with_su(getkey):
    """With S_u, conditional_variance adds variational correction to Schur diagonal."""
    N, M = 8, 3
    K_XX_diag = jnp.abs(jr.normal(getkey(), (N,))) + 2.0
    K_XZ = jr.normal(getkey(), (N, M)) * 0.3
    A_X = jr.normal(getkey(), (N, M)) * 0.3
    S_mat = random_pd_matrix(getkey(), M)
    S_u = lx.MatrixLinearOperator(S_mat, lx.symmetric_tag)

    result = conditional_variance(K_XX_diag, K_XZ, A_X, S_u)
    schur_diag = jnp.clip(K_XX_diag - jnp.sum(A_X * K_XZ, axis=1), 0.0)
    expected = schur_diag + jnp.diag(A_X @ S_mat @ A_X.T)

    assert tree_allclose(result, expected, rtol=1e-5)


def test_conditional_variance_identity_su(getkey):
    """With S_u = I, should give Schur diagonal + diag(A A^T)."""
    N, M = 6, 2
    K_XX_diag = jnp.abs(jr.normal(getkey(), (N,))) + 2.0
    K_XZ = jr.normal(getkey(), (N, M)) * 0.3
    A_X = jr.normal(getkey(), (N, M)) * 0.3
    S_u = lx.MatrixLinearOperator(jnp.eye(M))

    result = conditional_variance(K_XX_diag, K_XZ, A_X, S_u)
    schur_diag = jnp.clip(K_XX_diag - jnp.sum(A_X * K_XZ, axis=1), 0.0)
    expected = schur_diag + jnp.sum(A_X**2, axis=1)

    assert tree_allclose(result, expected, rtol=1e-5)


def test_conditional_variance_legacy_3arg_call_emits_deprecation(getkey):
    """Old 3-positional ``conditional_variance(base_diag, A_X, S_u)`` keeps
    working with a DeprecationWarning, returning
    ``base_diag + diag(A_X S_u A_X^T)``.
    """
    import pytest

    from gaussx import conditional_variance as cv

    N, M = 6, 3
    base_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
    A_X = jr.normal(getkey(), (N, M)) * 0.3
    S_mat = random_pd_matrix(getkey(), M)
    S_u = lx.MatrixLinearOperator(S_mat, lx.symmetric_tag)

    with pytest.warns(DeprecationWarning, match="conditional_variance"):
        result = cv(base_diag, A_X, S_u)

    expected = jnp.clip(base_diag, 0.0) + jnp.diag(A_X @ S_mat @ A_X.T)
    assert tree_allclose(result, expected, rtol=1e-5)
