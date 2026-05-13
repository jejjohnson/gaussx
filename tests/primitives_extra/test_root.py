"""Tests for root and inverse-root decomposition primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

import gaussx
from gaussx._gp._love import love_cache, love_variance
from gaussx._primitives import root_decomposition, root_inv_decomposition


def test_root_decomposition_truncated_diagonal():
    diag = jnp.array([1.0, 4.0, 2.0, 3.0])
    op = lx.DiagonalLinearOperator(diag)

    root = root_decomposition(op, rank=2, method="lanczos")
    approx = root.root @ root.root.T

    expected = jnp.diag(jnp.array([0.0, 4.0, 0.0, 3.0]))
    assert root.root.shape == (4, 2)
    assert root.rank == 2
    assert jnp.allclose(approx, expected, atol=1e-8)


def test_root_inv_decomposition_truncated_diagonal():
    diag = jnp.array([1.0, 4.0, 2.0, 3.0])
    op = lx.DiagonalLinearOperator(diag)

    root = root_inv_decomposition(op, rank=2, method="lanczos")
    approx = root.root @ root.root.T

    expected = jnp.diag(jnp.array([0.0, 1.0 / 4.0, 0.0, 1.0 / 3.0]))
    assert root.root.shape == (4, 2)
    assert jnp.allclose(approx, expected, atol=1e-8)


def test_cholesky_root_is_exact(getkey):
    A = jax.random.normal(getkey(), (5, 5))
    A = A @ A.T + 0.5 * jnp.eye(5)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)

    root = root_decomposition(op, rank=2, method="cholesky")
    inv_root = root_inv_decomposition(op, rank=2, method="cholesky")

    assert root.root.shape == (5, 5)
    assert jnp.allclose(root.root @ root.root.T, A, atol=1e-8)
    assert jnp.allclose(inv_root.root @ inv_root.root.T, jnp.linalg.inv(A), atol=1e-8)


def test_pivoted_cholesky_root_diagonal():
    diag = jnp.array([1.0, 4.0, 2.0, 3.0])
    op = lx.DiagonalLinearOperator(diag)

    root = root_decomposition(op, rank=2, method="pivoted_cholesky")
    approx = root.root @ root.root.T

    expected = jnp.diag(jnp.array([0.0, 4.0, 0.0, 3.0]))
    assert jnp.allclose(approx, expected, atol=1e-8)


def test_pivoted_cholesky_inverse_root_dense(getkey):
    """Inverse pivoted Cholesky at full rank reconstructs A^{-1}."""
    A = jax.random.normal(getkey(), (5, 5))
    A = A @ A.T + 0.5 * jnp.eye(5)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)

    inv_root = root_inv_decomposition(op, rank=5, method="pivoted_cholesky")
    reconstructed = inv_root.root @ inv_root.root.T

    assert inv_root.root.shape == (5, 5)
    assert jnp.allclose(reconstructed, jnp.linalg.inv(A), atol=1e-6)


def test_root_decomposition_cholesky_ignores_rank_sentinel(getkey):
    """``method='cholesky'`` is exact full-rank; ``rank`` is ignored."""
    A = jax.random.normal(getkey(), (4, 4))
    A = A @ A.T + 0.1 * jnp.eye(4)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)

    # rank=0 is a sentinel callers may pass when switching methods
    # dynamically — it must not raise for the cholesky path.
    root = root_decomposition(op, rank=0, method="cholesky")
    inv_root = root_inv_decomposition(op, rank=0, method="cholesky")

    assert root.root.shape == (4, 4)
    assert inv_root.root.shape == (4, 4)
    assert jnp.allclose(root.root @ root.root.T, A, atol=1e-8)


def test_root_matmul_sampling_covariance(getkey):
    num_samples = 4096
    empirical_cov_tolerance = 0.12
    diag = jnp.array([1.0, 2.0, 3.0])
    op = lx.DiagonalLinearOperator(diag)
    root = root_decomposition(op, rank=3, method="svd")

    eps = jax.random.normal(getkey(), (num_samples, root.rank))
    samples = root.matmul(eps)
    centered = samples - jnp.mean(samples, axis=0)
    empirical = centered.T @ centered / (samples.shape[0] - 1)

    assert samples.shape == (num_samples, 3)
    assert jnp.allclose(
        empirical,
        jnp.diag(diag),
        rtol=empirical_cov_tolerance,
        atol=empirical_cov_tolerance,
    )


def test_whitening_reconstructs_identity(getkey):
    diag = jnp.array([4.0, 9.0, 16.0])
    op = lx.DiagonalLinearOperator(diag)
    root = root_decomposition(op, rank=3, method="lanczos")
    inv_root = root_inv_decomposition(op, rank=3, method="lanczos")
    eps = jax.random.normal(getkey(), (root.rank,))

    residual = root.matmul(eps)
    whitened = inv_root.root.T @ residual

    assert jnp.allclose(whitened, eps, atol=1e-8)


def test_love_cache_matches_inverse_root(getkey):
    A = jax.random.normal(getkey(), (8, 8))
    A = A @ A.T + 0.3 * jnp.eye(8)
    op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
    k_star = jax.random.normal(getkey(), (8,))

    cache = love_cache(op, lanczos_order=8, key=getkey())
    inverse_root = root_inv_decomposition(op, rank=8, method="lanczos", key=getkey())
    expected = jnp.sum((inverse_root.root.T @ k_star) ** 2)

    assert jnp.allclose(love_variance(cache, k_star), expected, atol=1e-8)


def test_root_decomposition_top_level_export_and_jit_grad():
    diag = jnp.array([1.0, 2.0, 4.0])

    @jax.jit
    def root_sum(d):
        op = lx.DiagonalLinearOperator(d)
        return jnp.sum(gaussx.root_decomposition(op, rank=2, method="svd").root)

    value = root_sum(diag)
    grad = jax.grad(root_sum)(diag)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))
