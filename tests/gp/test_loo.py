"""Tests for leave-one-out cross-validation."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import leave_one_out_cv
from gaussx._strategies import DenseSolver
from gaussx._testing import key_sequence


class TestLeaveOneOutCV:
    @pytest.mark.slow
    def test_loo_means_shape(self):
        """LOO means have shape (N,)."""
        nextkey = key_sequence(0)
        N = 10
        A = jax.random.normal(nextkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        y = jax.random.normal(nextkey(), (N,))
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = leave_one_out_cv(op, y)
        assert result.loo_means.shape == (N,)
        assert result.loo_variances.shape == (N,)
        assert result.loo_log_likelihood.shape == ()

    def test_loo_variances_positive(self):
        """LOO variances should be positive."""
        nextkey = key_sequence(0)
        N = 10
        A = jax.random.normal(nextkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        y = jax.random.normal(nextkey(), (N,))
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = leave_one_out_cv(op, y)
        assert jnp.all(result.loo_variances > 0)

    @pytest.mark.slow
    def test_loo_matches_brute_force(self):
        """LOO means match brute-force leave-one-out on a small problem."""
        nextkey = key_sequence(0)
        N = 6
        A = jax.random.normal(nextkey(), (N, N))
        K = A @ A.T + 0.5 * jnp.eye(N)
        y = jax.random.normal(nextkey(), (N,))
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = leave_one_out_cv(op, y)

        # Brute force: for each i, remove point i and predict
        K_inv = jnp.linalg.inv(K)
        alpha = K_inv @ y
        diag_Kinv = jnp.diag(K_inv)
        expected_means = y - alpha / diag_Kinv
        expected_vars = 1.0 / diag_Kinv

        assert jnp.allclose(result.loo_means, expected_means, atol=1e-4)
        assert jnp.allclose(result.loo_variances, expected_vars, atol=1e-4)

    def test_loo_log_likelihood_finite(self):
        """LOO log-likelihood is finite."""
        nextkey = key_sequence(0)
        N = 8
        A = jax.random.normal(nextkey(), (N, N))
        K = A @ A.T + jnp.eye(N)
        y = jax.random.normal(nextkey(), (N,))
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = leave_one_out_cv(op, y)
        assert jnp.isfinite(result.loo_log_likelihood)

    def test_loo_with_explicit_diag_inv_solver(self):
        """The diagonal-inverse path can reuse an explicit solve strategy."""
        nextkey = key_sequence(0)
        N = 6
        A = jax.random.normal(nextkey(), (N, N))
        K = A @ A.T + 0.5 * jnp.eye(N)
        y = jax.random.normal(nextkey(), (N,))
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        result = leave_one_out_cv(
            op,
            y,
            solver=DenseSolver(),
            diag_inv_method="solve",
        )

        K_inv = jnp.linalg.inv(K)
        alpha = K_inv @ y
        diag_Kinv = jnp.diag(K_inv)
        expected_means = y - alpha / diag_Kinv
        expected_vars = 1.0 / diag_Kinv

        assert jnp.allclose(result.loo_means, expected_means, atol=1e-4)
        assert jnp.allclose(result.loo_variances, expected_vars, atol=1e-4)
