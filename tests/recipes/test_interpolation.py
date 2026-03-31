"""Tests for conditional interpolation between time points."""

import jax
import jax.numpy as jnp

from gaussx._recipes._interpolation import conditional_interpolate


class TestConditionalInterpolate:
    def test_shapes(self, getkey):
        """Output should be (d,) mean and (d, d) covariance."""
        d = 3
        A_fwd = 0.9 * jnp.eye(d)
        Q_fwd = 0.1 * jnp.eye(d)
        A_bwd = 0.9 * jnp.eye(d)
        Q_bwd = 0.1 * jnp.eye(d)
        mu_prev = jax.random.normal(getkey(), (d,))
        P_prev = 0.5 * jnp.eye(d)
        mu_next = jax.random.normal(getkey(), (d,))
        P_next = 0.5 * jnp.eye(d)

        m, P = conditional_interpolate(
            A_fwd, Q_fwd, A_bwd, Q_bwd, mu_prev, P_prev, mu_next, P_next
        )
        assert m.shape == (d,)
        assert P.shape == (d, d)

    def test_uncertainty_reduction(self, getkey):
        """Fused estimate should have less uncertainty than forward-only."""
        d = 2
        A_fwd = jnp.eye(d)
        Q_fwd = 0.5 * jnp.eye(d)
        A_bwd = jnp.eye(d)
        Q_bwd = 0.5 * jnp.eye(d)
        mu_prev = jnp.zeros(d)
        P_prev = jnp.eye(d)
        mu_next = jnp.ones(d)
        P_next = jnp.eye(d)

        _, P_fused = conditional_interpolate(
            A_fwd, Q_fwd, A_bwd, Q_bwd, mu_prev, P_prev, mu_next, P_next
        )

        # Forward-only prediction covariance
        P_fwd = A_fwd @ P_prev @ A_fwd.T + Q_fwd

        # Fused should be tighter
        assert jnp.trace(P_fused) < jnp.trace(P_fwd)

    def test_symmetric_case(self, getkey):
        """Symmetric inputs should give mean at midpoint."""
        d = 2
        A = jnp.eye(d)
        Q = 0.1 * jnp.eye(d)
        mu_prev = jnp.array([0.0, 0.0])
        mu_next = jnp.array([2.0, 2.0])
        P = jnp.eye(d)

        m, _ = conditional_interpolate(A, Q, A, Q, mu_prev, P, mu_next, P)

        # With identical dynamics and noise, mean should be near midpoint
        assert jnp.allclose(m, jnp.array([1.0, 1.0]), atol=0.3)

    def test_psd_covariance(self, getkey):
        """Output covariance should be positive definite."""
        d = 3
        A_fwd = 0.8 * jnp.eye(d) + 0.1 * jax.random.normal(getkey(), (d, d))
        Q_fwd = jax.random.normal(getkey(), (d, d))
        Q_fwd = Q_fwd @ Q_fwd.T + 0.1 * jnp.eye(d)
        A_bwd = 0.8 * jnp.eye(d) + 0.1 * jax.random.normal(getkey(), (d, d))
        Q_bwd = jax.random.normal(getkey(), (d, d))
        Q_bwd = Q_bwd @ Q_bwd.T + 0.1 * jnp.eye(d)
        mu_prev = jax.random.normal(getkey(), (d,))
        P_prev = jax.random.normal(getkey(), (d, d))
        P_prev = P_prev @ P_prev.T + 0.1 * jnp.eye(d)
        mu_next = jax.random.normal(getkey(), (d,))
        P_next = jax.random.normal(getkey(), (d, d))
        P_next = P_next @ P_next.T + 0.1 * jnp.eye(d)

        _, P_out = conditional_interpolate(
            A_fwd, Q_fwd, A_bwd, Q_bwd, mu_prev, P_prev, mu_next, P_next
        )
        eigvals = jnp.linalg.eigvalsh(P_out)
        assert jnp.all(eigvals > -1e-6)

    def test_finite(self, getkey):
        """All outputs should be finite."""
        d = 2
        A = 0.9 * jnp.eye(d)
        Q = 0.2 * jnp.eye(d)
        mu = jax.random.normal(getkey(), (d,))
        P = 0.5 * jnp.eye(d)

        m, P_out = conditional_interpolate(A, Q, A, Q, mu, P, mu, P)
        assert jnp.all(jnp.isfinite(m))
        assert jnp.all(jnp.isfinite(P_out))
