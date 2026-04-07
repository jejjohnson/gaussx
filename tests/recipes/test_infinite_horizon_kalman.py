"""Tests for infinite-horizon Kalman filter and smoother."""

import jax
import jax.numpy as jnp

from gaussx import (
    dare,
    infinite_horizon_filter,
    infinite_horizon_smoother,
    kalman_filter,
)


class TestInfiniteHorizonFilter:
    def _stable_system(self, getkey):
        """Create a stable system for testing."""
        D, M = 3, 2
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        return A, H, Q, R

    def test_output_shapes(self, getkey):
        """Output shapes match time steps and state dimension."""
        A, H, Q, R = self._stable_system(getkey)
        D, M, T = A.shape[0], H.shape[0], 50
        obs = jax.random.normal(getkey(), (T, M))
        state = infinite_horizon_filter(A, H, Q, R, obs)
        assert state.filtered_means.shape == (T, D)
        assert state.filtered_covs.shape == (T, D, D)
        assert state.predicted_means.shape == (T, D)
        assert state.predicted_covs.shape == (T, D, D)
        assert state.log_likelihood.shape == ()

    def test_log_likelihood_finite(self, getkey):
        """Log-likelihood is finite."""
        A, H, Q, R = self._stable_system(getkey)
        T = 50
        obs = jax.random.normal(getkey(), (T, H.shape[0]))
        state = infinite_horizon_filter(A, H, Q, R, obs)
        assert jnp.isfinite(state.log_likelihood)

    def test_matches_standard_kf_at_convergence(self, getkey):
        """Filtered means converge to standard KF for long sequences."""
        A, H, Q, R = self._stable_system(getkey)
        D, M, T = A.shape[0], H.shape[0], 500
        obs = jax.random.normal(getkey(), (T, M))

        # Standard KF
        init_mean = jnp.zeros(D)
        init_cov = jnp.eye(D)
        kf_state = kalman_filter(A, H, Q, R, obs, init_mean, init_cov)

        # Infinite-horizon KF
        ih_state = infinite_horizon_filter(A, H, Q, R, obs)

        # Compare last 100 time steps (after convergence)
        assert jnp.allclose(
            kf_state.filtered_means[-100:],
            ih_state.filtered_means[-100:],
            atol=0.1,
        )

    def test_precomputed_dare(self, getkey):
        """Accepts precomputed DARE result."""
        A, H, Q, R = self._stable_system(getkey)
        T = 50
        obs = jax.random.normal(getkey(), (T, H.shape[0]))
        dare_result = dare(A, H, Q, R)
        state = infinite_horizon_filter(
            A,
            H,
            Q,
            R,
            obs,
            dare_result=dare_result,
        )
        assert jnp.isfinite(state.log_likelihood)

    def test_jit_compatible(self, getkey):
        """Works under jax.jit."""
        A, H, Q, R = self._stable_system(getkey)
        T = 30
        obs = jax.random.normal(getkey(), (T, H.shape[0]))
        dare_result = dare(A, H, Q, R)
        state = jax.jit(
            lambda o: infinite_horizon_filter(
                A,
                H,
                Q,
                R,
                o,
                dare_result=dare_result,
            )
        )(obs)
        assert jnp.isfinite(state.log_likelihood)


class TestInfiniteHorizonSmoother:
    def test_output_shapes(self, getkey):
        """Smoothed outputs have correct shapes."""
        D, M, T = 3, 2, 50
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        obs = jax.random.normal(getkey(), (T, M))
        dare_result = dare(A, H, Q, R)
        filt = infinite_horizon_filter(
            A,
            H,
            Q,
            R,
            obs,
            dare_result=dare_result,
        )
        s_means, s_covs = infinite_horizon_smoother(filt, A, dare_result, Q)
        assert s_means.shape == (T, D)
        assert s_covs.shape == (T, D, D)

    def test_smoother_jit_compatible(self, getkey):
        """Smoother works under jax.jit."""
        D, M, T = 3, 2, 30
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        obs = jax.random.normal(getkey(), (T, M))
        dare_result = dare(A, H, Q, R)
        filt = infinite_horizon_filter(
            A,
            H,
            Q,
            R,
            obs,
            dare_result=dare_result,
        )

        @jax.jit
        def smooth(filt):
            return infinite_horizon_smoother(filt, A, dare_result, Q)

        s_means, _s_covs = smooth(filt)
        assert jnp.all(jnp.isfinite(s_means))
