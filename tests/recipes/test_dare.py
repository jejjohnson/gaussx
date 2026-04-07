"""Tests for DARE solver."""

import jax
import jax.numpy as jnp

from gaussx import dare


class TestDARE:
    def test_converges_stable_system(self, getkey):
        """DARE converges for a stable system."""
        D, M = 3, 2
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        result = dare(A, H, Q, R)
        assert result.converged
        assert result.P_inf.shape == (D, D)
        assert result.K_inf.shape == (D, M)

    def test_satisfies_dare(self, getkey):
        """P_inf satisfies the DARE fixed-point equation."""
        D, M = 3, 2
        A = 0.8 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        result = dare(A, H, Q, R, max_iter=200)

        P = result.P_inf
        # One predict-update step from P should return P (fixed point).
        P_pred = A @ P @ A.T + Q
        S = H @ P_pred @ H.T + R
        K = jnp.linalg.solve(S, H @ P_pred).T
        P_updated = (jnp.eye(D) - K @ H) @ P_pred
        assert jnp.allclose(P, P_updated, atol=1e-6)

    def test_p_inf_symmetric(self, getkey):
        """Steady-state covariance is symmetric."""
        D, M = 4, 2
        A = 0.85 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.3
        Q = 0.2 * jnp.eye(D)
        R = jnp.eye(M)
        result = dare(A, H, Q, R)
        assert jnp.allclose(result.P_inf, result.P_inf.T, atol=1e-8)

    def test_p_inf_positive_definite(self, getkey):
        """Steady-state covariance is positive definite."""
        D, M = 3, 2
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        result = dare(A, H, Q, R)
        eigvals = jnp.linalg.eigvalsh(result.P_inf)
        assert jnp.all(eigvals > 0)

    def test_jit_compatible(self, getkey):
        """Works under jax.jit."""
        D, M = 2, 1
        A = 0.9 * jnp.eye(D)
        H = jnp.ones((M, D))
        Q = 0.1 * jnp.eye(D)
        R = jnp.eye(M)
        result = jax.jit(dare)(A, H, Q, R)
        assert result.converged

    def test_custom_p_init(self, getkey):
        """Custom P_init does not affect convergence."""
        D, M = 3, 2
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        P_init = jnp.eye(D)
        result = dare(A, H, Q, R, P_init=P_init)
        assert result.converged
