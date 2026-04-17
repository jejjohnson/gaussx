"""Tests for unwhiten and whiten_covariance sugar operations."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import lineax as lx

from gaussx import unwhiten, whiten_covariance
from gaussx._testing import random_pd_matrix, tree_allclose


def test_unwhiten_basic(getkey):
    """unwhiten(m_tilde, L) should equal L @ m_tilde."""
    M = 4
    K_ZZ = random_pd_matrix(getkey(), M)
    L = jax.scipy.linalg.cholesky(K_ZZ, lower=True)
    L_op = lx.MatrixLinearOperator(L, lx.lower_triangular_tag)

    m_tilde = jr.normal(getkey(), (M,))
    result = unwhiten(m_tilde, L_op)
    expected = L @ m_tilde

    assert tree_allclose(result, expected)


def test_whiten_covariance_basic(getkey):
    """whiten_covariance(L, S_tilde) should equal L @ S_tilde @ L^T."""
    M = 4
    K_ZZ = random_pd_matrix(getkey(), M)
    L = jax.scipy.linalg.cholesky(K_ZZ, lower=True)
    L_op = lx.MatrixLinearOperator(L, lx.lower_triangular_tag)

    S_tilde_mat = random_pd_matrix(getkey(), M)
    S_tilde_op = lx.MatrixLinearOperator(S_tilde_mat, lx.symmetric_tag)

    result = whiten_covariance(L_op, S_tilde_op)
    expected = L @ S_tilde_mat @ L.T

    assert tree_allclose(result.as_matrix(), expected, rtol=1e-4)


def test_whiten_covariance_identity(getkey):
    """Unwhitening with identity S_tilde should give L @ L^T = K_ZZ."""
    M = 3
    K_ZZ = random_pd_matrix(getkey(), M)
    L = jax.scipy.linalg.cholesky(K_ZZ, lower=True)
    L_op = lx.MatrixLinearOperator(L, lx.lower_triangular_tag)

    I_op = lx.MatrixLinearOperator(jnp.eye(M), lx.symmetric_tag)
    result = whiten_covariance(L_op, I_op)

    assert tree_allclose(result.as_matrix(), K_ZZ, rtol=1e-5)
