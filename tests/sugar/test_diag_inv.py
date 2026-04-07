"""Tests for diag_inv."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import diag_inv


class TestDiagInv:
    def test_cholesky_matches_dense(self, getkey):
        """Cholesky method matches jnp.diag(jnp.linalg.inv(A))."""
        N = 12
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = diag_inv(op, method="cholesky")
        expected = jnp.diag(jnp.linalg.inv(K))
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_hutchinson_converges(self, getkey):
        """Hutchinson estimate is close with many probes."""
        N = 12
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = diag_inv(op, method="hutchinson", num_probes=1000, seed=42)
        expected = jnp.diag(jnp.linalg.inv(K))
        assert jnp.allclose(result, expected, rtol=0.2, atol=0.05)

    def test_auto_selects_cholesky_small(self, getkey):
        """Auto mode uses cholesky for small N."""
        N = 8
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = diag_inv(op, method="auto")
        expected = jnp.diag(jnp.linalg.inv(K))
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_output_shape(self, getkey):
        """Output shape is (N,)."""
        N = 10
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = diag_inv(op)
        assert result.shape == (N,)

    def test_invalid_method_raises(self, getkey):
        """Unknown method raises ValueError."""
        N = 5
        A = jax.random.normal(getkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        with pytest.raises(ValueError, match="Unknown method"):
            diag_inv(op, method="bogus")
