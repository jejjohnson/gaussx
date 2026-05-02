"""Tests for linear algebra sugar operations."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import cov_transform, diag_conditional_variance, trace_product
from gaussx._testing import random_pd_matrix, tree_allclose


def test_cov_transform_identity(getkey):
    """J=I should return Sigma unchanged."""
    N = 4
    mat = random_pd_matrix(getkey(), N)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    J = jnp.eye(N)
    result = cov_transform(J, op)
    assert tree_allclose(result.as_matrix(), mat, rtol=1e-5)


def test_cov_transform_rectangular(getkey):
    """J @ Sigma @ J^T for rectangular J."""
    N, M = 5, 3
    mat = random_pd_matrix(getkey(), N)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    J = jr.normal(getkey(), (M, N))
    result = cov_transform(J, op)
    expected = J @ mat @ J.T
    assert result.as_matrix().shape == (M, M)
    assert tree_allclose(result.as_matrix(), expected, rtol=1e-5)


def test_cov_transform_symmetric(getkey):
    """Result should be symmetric when input is."""
    N = 4
    mat = random_pd_matrix(getkey(), N)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    J = jr.normal(getkey(), (3, N))
    result = cov_transform(J, op)
    r = result.as_matrix()
    assert tree_allclose(r, r.T, atol=1e-10)


def test_trace_product_matches_dense(getkey):
    """tr(A @ B) should match jnp.trace(A @ B)."""
    N = 5
    A_mat = jr.normal(getkey(), (N, N))
    B_mat = jr.normal(getkey(), (N, N))
    A = lx.MatrixLinearOperator(A_mat)
    B = lx.MatrixLinearOperator(B_mat)

    result = trace_product(A, B)
    expected = jnp.trace(A_mat @ B_mat)
    assert tree_allclose(result, expected, rtol=1e-5)


def test_trace_product_symmetric(getkey):
    """tr(Sigma @ Lambda) for PSD matrices."""
    N = 4
    S = random_pd_matrix(getkey(), N)
    L = random_pd_matrix(getkey(), N)
    result = trace_product(lx.MatrixLinearOperator(S), lx.MatrixLinearOperator(L))
    expected = jnp.trace(S @ L)
    assert tree_allclose(result, expected, rtol=1e-5)


def test_trace_product_identity(getkey):
    """tr(A @ I) = tr(A)."""
    N = 4
    A_mat = jr.normal(getkey(), (N, N))
    A = lx.MatrixLinearOperator(A_mat)
    I = lx.MatrixLinearOperator(jnp.eye(N))
    assert tree_allclose(trace_product(A, I), jnp.trace(A_mat), rtol=1e-6)


def test_diag_conditional_variance_basic(getkey):
    N, M = 6, 3
    K_XX_diag = jnp.abs(jr.normal(getkey(), (N,))) + 2.0
    K_XZ = jr.normal(getkey(), (N, M)) * 0.3
    A_X = jr.normal(getkey(), (N, M)) * 0.3

    result = diag_conditional_variance(K_XX_diag, K_XZ, A_X)
    expected = jnp.clip(K_XX_diag - jnp.sum(A_X * K_XZ, axis=1), 0.0)
    assert tree_allclose(result, expected)


def test_diag_conditional_variance_nonneg(getkey):
    """Result should always be non-negative."""
    N, M = 10, 5
    K_XX_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1
    K_XZ = jr.normal(getkey(), (N, M))
    A_X = jr.normal(getkey(), (N, M))
    result = diag_conditional_variance(K_XX_diag, K_XZ, A_X)
    assert jnp.all(result >= 0.0)


def test_diag_conditional_variance_zero_projection(getkey):
    """With A_X=0, should return K_XX_diag."""
    N, M = 5, 3
    K_XX_diag = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    K_XZ = jr.normal(getkey(), (N, M))
    A_X = jnp.zeros((N, M))
    result = diag_conditional_variance(K_XX_diag, K_XZ, A_X)
    assert tree_allclose(result, K_XX_diag)


def test_trace_product_diagonal_diagonal(getkey):
    """Both diagonal: should hit the structural path and match dense."""
    N = 4
    da = jr.normal(getkey(), (N,))
    db = jr.normal(getkey(), (N,))
    A = lx.DiagonalLinearOperator(da)
    B = lx.DiagonalLinearOperator(db)
    result = trace_product(A, B)
    expected = jnp.trace(jnp.diag(da) @ jnp.diag(db))
    assert tree_allclose(result, expected, rtol=1e-6)


def test_trace_product_diagonal_full(getkey):
    """Diagonal × general: contract via diag(B)."""
    N = 5
    d = jr.normal(getkey(), (N,))
    Bmat = jr.normal(getkey(), (N, N))
    A = lx.DiagonalLinearOperator(d)
    B = lx.MatrixLinearOperator(Bmat)
    result = trace_product(A, B)
    expected = jnp.trace(jnp.diag(d) @ Bmat)
    assert tree_allclose(result, expected, rtol=1e-6)


def test_trace_product_block_diag_matched(getkey):
    """Matched BlockDiag: sum of per-block trace_product."""
    from gaussx import BlockDiag

    A = BlockDiag(
        lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
        lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4))),
    )
    B = BlockDiag(
        lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
        lx.MatrixLinearOperator(jr.normal(getkey(), (4, 4))),
    )
    expected = jnp.trace(A.as_matrix() @ B.as_matrix())
    assert tree_allclose(trace_product(A, B), expected, rtol=1e-5)


def test_trace_product_kronecker_matched(getkey):
    """Matched Kronecker: product of per-factor trace_product."""
    from gaussx import Kronecker

    A = Kronecker(
        lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2))),
        lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
    )
    B = Kronecker(
        lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2))),
        lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3))),
    )
    expected = jnp.trace(A.as_matrix() @ B.as_matrix())
    assert tree_allclose(trace_product(A, B), expected, rtol=1e-5)


def test_cov_transform_diagonal_avoids_full_materialization(getkey):
    """DiagonalLinearOperator covariance: J diag(d) J^T should match dense."""
    N, M = 6, 3
    d = jnp.abs(jr.normal(getkey(), (N,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    J = jr.normal(getkey(), (M, N))
    expected = J @ jnp.diag(d) @ J.T
    assert tree_allclose(cov_transform(J, op).as_matrix(), expected, rtol=1e-5)
